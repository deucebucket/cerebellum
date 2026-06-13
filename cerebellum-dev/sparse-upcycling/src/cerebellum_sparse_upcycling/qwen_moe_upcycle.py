from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file


LAYER_MLP_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
MTP_RE = re.compile(r"^mtp\.")


@dataclass(frozen=True)
class UpcycleSpec:
    name: str
    source_model: str
    layers: int
    hidden_size: int
    dense_intermediate_size: int
    num_experts: int
    top_k: int
    expert_intermediate_size: int
    shared_expert_intermediate_size: int
    output_dir: Path
    max_shard_size_gb: float = 4.0
    disable_mtp: bool = True
    expert_init: str = "sliced_dense_ffn"
    shared_expert_init: str = "dense_slice"

    @classmethod
    def from_config(cls, cfg: dict) -> "UpcycleSpec":
        moe = cfg["moe"]
        checkpoint = cfg.get("checkpoint", {})
        return cls(
            name=cfg["name"],
            source_model=cfg["source_model"],
            layers=cfg["layers"],
            hidden_size=cfg["hidden_size"],
            dense_intermediate_size=cfg["dense_intermediate_size"],
            num_experts=moe["num_experts"],
            top_k=moe["top_k"],
            expert_intermediate_size=moe["expert_intermediate_size"],
            shared_expert_intermediate_size=moe["shared_expert_intermediate_size"],
            output_dir=Path(checkpoint.get("output_dir", f"/var/home/deucebucket/games/models/{cfg['name']}")),
            max_shard_size_gb=checkpoint.get("max_shard_size_gb", 4.0),
            disable_mtp=checkpoint.get(
                "disable_mtp",
                checkpoint.get("disable_mtp_for_v1", checkpoint.get("disable_mtp_for_v0", True)),
            ),
            expert_init=moe.get("expert_init", "sliced_dense_ffn"),
            shared_expert_init=moe.get("shared_expert_init", "dense_slice"),
        )

    @property
    def max_shard_bytes(self) -> int:
        return int(self.max_shard_size_gb * 1024**3)

    def validate(self) -> None:
        valid_expert_inits = {"sliced_dense_ffn", "zero_residual_experts"}
        valid_shared_inits = {"dense_slice", "dense_ffn_scaled_down_projection_2x"}
        if self.expert_init not in valid_expert_inits:
            raise ValueError(f"unknown expert_init {self.expert_init!r}; expected one of {sorted(valid_expert_inits)}")
        if self.shared_expert_init not in valid_shared_inits:
            raise ValueError(
                f"unknown shared_expert_init {self.shared_expert_init!r}; expected one of {sorted(valid_shared_inits)}"
            )
        if self.dense_intermediate_size != self.num_experts * self.expert_intermediate_size:
            raise ValueError(
                "sparse upcycling requires dense_intermediate_size == "
                "num_experts * expert_intermediate_size"
            )
        if (
            self.shared_expert_init == "dense_ffn_scaled_down_projection_2x"
            and self.shared_expert_intermediate_size != self.dense_intermediate_size
        ):
            raise ValueError(
                "dense_ffn_scaled_down_projection_2x requires "
                "shared_expert_intermediate_size == dense_intermediate_size"
            )
        if self.shared_expert_init == "dense_slice" and self.shared_expert_intermediate_size > self.dense_intermediate_size:
            raise ValueError("dense_slice cannot exceed dense_intermediate_size")
        if self.top_k > self.num_experts:
            raise ValueError("top_k must be <= num_experts")


def load_source_index(source_model: str) -> dict:
    return json.loads(Path(hf_hub_download(source_model, "model.safetensors.index.json")).read_text())


def load_source_config(source_model: str) -> dict:
    return json.loads(Path(hf_hub_download(source_model, "config.json")).read_text())


def build_moe_config(source_config: dict, spec: UpcycleSpec) -> dict:
    cfg = json.loads(json.dumps(source_config))
    cfg["model_type"] = "qwen3_5_moe"
    cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    cfg["transformers_version"] = cfg.get("transformers_version", "4.57.0.dev0")

    text = cfg["text_config"]
    text["model_type"] = "qwen3_5_moe_text"
    text.pop("intermediate_size", None)
    text["moe_intermediate_size"] = spec.expert_intermediate_size
    text["num_experts"] = spec.num_experts
    text["num_experts_per_tok"] = spec.top_k
    text["shared_expert_intermediate_size"] = spec.shared_expert_intermediate_size
    text.setdefault("router_aux_loss_coef", 0.001)
    if spec.disable_mtp:
        text["mtp_num_hidden_layers"] = 0

    if "vision_config" in cfg:
        cfg["vision_config"]["model_type"] = "qwen3_5_moe"
        cfg["vision_config"]["out_hidden_size"] = spec.hidden_size

    return cfg


def is_source_dense_mlp(key: str) -> bool:
    return LAYER_MLP_RE.match(key) is not None


def should_skip_source_key(key: str, spec: UpcycleSpec) -> bool:
    if spec.disable_mtp and MTP_RE.match(key):
        return True
    return is_source_dense_mlp(key)


def convert_mlp_layer(
    layer: int,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    spec: UpcycleSpec,
) -> dict[str, torch.Tensor]:
    e = spec.expert_intermediate_size
    num = spec.num_experts

    if gate.shape != (spec.dense_intermediate_size, spec.hidden_size):
        raise ValueError(f"layer {layer} gate shape {tuple(gate.shape)}")
    if up.shape != (spec.dense_intermediate_size, spec.hidden_size):
        raise ValueError(f"layer {layer} up shape {tuple(up.shape)}")
    if down.shape != (spec.hidden_size, spec.dense_intermediate_size):
        raise ValueError(f"layer {layer} down shape {tuple(down.shape)}")

    prefix = f"model.language_model.layers.{layer}.mlp"
    router = torch.zeros((spec.num_experts, spec.hidden_size), dtype=gate.dtype)
    shared_gate = torch.zeros((1, spec.hidden_size), dtype=gate.dtype)

    if spec.expert_init == "zero_residual_experts":
        if spec.shared_expert_init != "dense_ffn_scaled_down_projection_2x":
            raise ValueError("zero_residual_experts requires dense_ffn_scaled_down_projection_2x shared expert init")
        return {
            f"{prefix}.experts.gate_up_proj": torch.zeros(
                (num, 2 * e, spec.hidden_size),
                dtype=gate.dtype,
            ),
            f"{prefix}.experts.down_proj": torch.zeros((num, spec.hidden_size, e), dtype=down.dtype),
            f"{prefix}.gate.weight": router,
            f"{prefix}.shared_expert.gate_proj.weight": gate.contiguous(),
            f"{prefix}.shared_expert.up_proj.weight": up.contiguous(),
            f"{prefix}.shared_expert.down_proj.weight": (down * 2).contiguous(),
            f"{prefix}.shared_expert_gate.weight": shared_gate,
        }

    if spec.shared_expert_init != "dense_slice":
        raise ValueError("sliced_dense_ffn requires dense_slice shared expert init")

    gate_experts = gate.reshape(num, e, spec.hidden_size)
    up_experts = up.reshape(num, e, spec.hidden_size)
    down_experts = down.reshape(spec.hidden_size, num, e).permute(1, 0, 2).contiguous()
    gate_up = torch.cat([gate_experts, up_experts], dim=1).contiguous()

    shared_slice = slice(0, spec.shared_expert_intermediate_size)
    return {
        f"{prefix}.experts.gate_up_proj": gate_up,
        f"{prefix}.experts.down_proj": down_experts,
        f"{prefix}.gate.weight": router,
        f"{prefix}.shared_expert.gate_proj.weight": gate[shared_slice].contiguous(),
        f"{prefix}.shared_expert.up_proj.weight": up[shared_slice].contiguous(),
        f"{prefix}.shared_expert.down_proj.weight": down[:, shared_slice].contiguous(),
        f"{prefix}.shared_expert_gate.weight": shared_gate,
    }


def iter_source_shards(index: dict) -> Iterable[tuple[str, list[str]]]:
    by_file: dict[str, list[str]] = {}
    for key, filename in index["weight_map"].items():
        by_file.setdefault(filename, []).append(key)
    for filename in sorted(by_file):
        yield filename, sorted(by_file[filename])


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class ShardWriter:
    def __init__(self, output_dir: Path, max_shard_bytes: int):
        self.output_dir = output_dir
        self.max_shard_bytes = max_shard_bytes
        self.current: dict[str, torch.Tensor] = {}
        self.current_bytes = 0
        self.shard_index = 0
        self.weight_map: dict[str, str] = {}
        self.total_size = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        size = tensor_nbytes(tensor)
        if self.current and self.current_bytes + size > self.max_shard_bytes:
            self.flush()
        self.current[name] = tensor.contiguous()
        self.current_bytes += size

    def flush(self) -> None:
        if not self.current:
            return
        self.shard_index += 1
        filename = f"model-{self.shard_index:05d}-of-PLACEHOLDER.safetensors"
        save_file(self.current, str(self.output_dir / filename), metadata={"format": "pt"})
        for key, tensor in self.current.items():
            self.weight_map[key] = filename
            self.total_size += tensor_nbytes(tensor)
        self.current = {}
        self.current_bytes = 0

    def finalize(self) -> dict:
        self.flush()
        final_map: dict[str, str] = {}
        total = self.shard_index
        for old in sorted(set(self.weight_map.values())):
            new = old.replace("PLACEHOLDER", f"{total:05d}")
            if old != new:
                (self.output_dir / old).rename(self.output_dir / new)
        for key, filename in self.weight_map.items():
            final_map[key] = filename.replace("PLACEHOLDER", f"{total:05d}")
        return {
            "metadata": {"total_size": self.total_size},
            "weight_map": dict(sorted(final_map.items())),
        }


def copy_repo_sidecars(source_dir: Path, output_dir: Path) -> None:
    skip_suffixes = {".safetensors"}
    skip_names = {"model.safetensors.index.json", "config.json"}
    for path in source_dir.iterdir():
        if path.name in skip_names or path.suffix in skip_suffixes:
            continue
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)


def ensure_source_snapshot(source_model: str) -> Path:
    return Path(snapshot_download(source_model, allow_patterns=["*"]))


def upcycle_checkpoint(config_path: Path, dry_run: bool = False, overwrite: bool = False) -> Path:
    cfg = json.loads(config_path.read_text())
    spec = UpcycleSpec.from_config(cfg)
    spec.validate()

    source_config = load_source_config(spec.source_model)
    source_index = load_source_index(spec.source_model)
    output_dir = spec.output_dir

    if dry_run:
        return output_dir

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    source_dir = ensure_source_snapshot(spec.source_model)
    copy_repo_sidecars(source_dir, output_dir)
    (output_dir / "config.json").write_text(json.dumps(build_moe_config(source_config, spec), indent=2) + "\n")

    writer = ShardWriter(output_dir, spec.max_shard_bytes)
    pending_mlp: dict[int, dict[str, torch.Tensor]] = {}
    manifest = {
        "name": spec.name,
        "source_model": spec.source_model,
        "source_snapshot": str(source_dir),
        "output_dir": str(output_dir),
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
        "expert_intermediate_size": spec.expert_intermediate_size,
        "shared_expert_intermediate_size": spec.shared_expert_intermediate_size,
        "expert_init": spec.expert_init,
        "shared_expert_init": spec.shared_expert_init,
        "disable_mtp": spec.disable_mtp,
        "converted_layers": [],
        "copied_tensors": 0,
    }

    for filename, keys in iter_source_shards(source_index):
        shard = load_file(str(source_dir / filename), device="cpu")
        for key in keys:
            if key not in shard:
                continue
            match = LAYER_MLP_RE.match(key)
            if match:
                layer = int(match.group(1))
                part = match.group(2)
                pending_mlp.setdefault(layer, {})[part] = shard[key]
                if {"gate_proj", "up_proj", "down_proj"} <= pending_mlp[layer].keys():
                    tensors = convert_mlp_layer(
                        layer,
                        pending_mlp[layer]["gate_proj"],
                        pending_mlp[layer]["up_proj"],
                        pending_mlp[layer]["down_proj"],
                        spec,
                    )
                    for out_key, tensor in tensors.items():
                        writer.add(out_key, tensor)
                    del pending_mlp[layer]
                    manifest["converted_layers"].append(layer)
                continue
            if should_skip_source_key(key, spec):
                continue
            writer.add(key, shard[key])
            manifest["copied_tensors"] += 1
        del shard

    if pending_mlp:
        raise RuntimeError(f"incomplete MLP layers: {sorted(pending_mlp)}")

    index = writer.finalize()
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    manifest["converted_layers"] = sorted(manifest["converted_layers"])
    manifest["total_size"] = index["metadata"]["total_size"]
    (output_dir / "cerebellum_sparse_upcycle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output_dir


def dry_run_summary(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    spec = UpcycleSpec.from_config(cfg)
    spec.validate()
    source_config = load_source_config(spec.source_model)
    source_index = load_source_index(spec.source_model)
    moe_config = build_moe_config(source_config, spec)

    source_keys = sorted(source_index["weight_map"])
    dense_mlp_keys = [k for k in source_keys if is_source_dense_mlp(k)]
    skipped_mtp = [k for k in source_keys if spec.disable_mtp and MTP_RE.match(k)]
    output_mlp_keys = []
    for layer in range(spec.layers):
        prefix = f"model.language_model.layers.{layer}.mlp"
        output_mlp_keys.extend(
            [
                f"{prefix}.experts.gate_up_proj",
                f"{prefix}.experts.down_proj",
                f"{prefix}.gate.weight",
                f"{prefix}.shared_expert.gate_proj.weight",
                f"{prefix}.shared_expert.up_proj.weight",
                f"{prefix}.shared_expert.down_proj.weight",
                f"{prefix}.shared_expert_gate.weight",
            ]
        )

    copied = [k for k in source_keys if not should_skip_source_key(k, spec)]
    return {
        "name": spec.name,
        "source_model": spec.source_model,
        "output_dir": str(spec.output_dir),
        "model_type": moe_config["model_type"],
        "text_model_type": moe_config["text_config"]["model_type"],
        "mtp_num_hidden_layers": moe_config["text_config"].get("mtp_num_hidden_layers"),
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
        "expert_init": spec.expert_init,
        "shared_expert_init": spec.shared_expert_init,
        "expert_intermediate_size": spec.expert_intermediate_size,
        "shared_expert_intermediate_size": spec.shared_expert_intermediate_size,
        "disable_mtp": spec.disable_mtp,
        "dense_mlp_source_keys": len(dense_mlp_keys),
        "skipped_mtp_keys": len(skipped_mtp),
        "copied_source_keys": len(copied),
        "generated_moe_mlp_keys": len(output_mlp_keys),
        "estimated_output_keys": len(copied) + len(output_mlp_keys),
        "expert_gate_up_shape": [
            spec.num_experts,
            2 * spec.expert_intermediate_size,
            spec.hidden_size,
        ],
        "expert_down_shape": [
            spec.num_experts,
            spec.hidden_size,
            spec.expert_intermediate_size,
        ],
        "shared_expert_gate_shape": [
            spec.shared_expert_intermediate_size,
            spec.hidden_size,
        ],
        "shared_expert_up_shape": [
            spec.shared_expert_intermediate_size,
            spec.hidden_size,
        ],
        "shared_expert_down_shape": [
            spec.hidden_size,
            spec.shared_expert_intermediate_size,
        ],
    }
