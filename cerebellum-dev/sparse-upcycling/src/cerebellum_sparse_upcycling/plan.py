from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseMlpShape:
    hidden_size: int
    intermediate_size: int
    layers: int
    mlp_type: str = "swiglu"

    @property
    def params_per_layer(self) -> int:
        if self.mlp_type != "swiglu":
            raise ValueError(f"unsupported mlp_type: {self.mlp_type}")
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def total_params(self) -> int:
        return self.params_per_layer * self.layers


@dataclass(frozen=True)
class MoeMlpShape:
    hidden_size: int
    layers: int
    num_experts: int
    top_k: int
    expert_intermediate_size: int
    shared_expert_intermediate_size: int = 0
    mlp_type: str = "swiglu"

    @property
    def routed_params_per_layer(self) -> int:
        if self.mlp_type != "swiglu":
            raise ValueError(f"unsupported mlp_type: {self.mlp_type}")
        return 3 * self.hidden_size * self.expert_intermediate_size * self.num_experts

    @property
    def shared_params_per_layer(self) -> int:
        return 3 * self.hidden_size * self.shared_expert_intermediate_size

    @property
    def router_params_per_layer(self) -> int:
        return self.hidden_size * self.num_experts

    @property
    def total_params_per_layer(self) -> int:
        return self.routed_params_per_layer + self.shared_params_per_layer + self.router_params_per_layer

    @property
    def total_params(self) -> int:
        return self.total_params_per_layer * self.layers

    @property
    def active_params_per_layer(self) -> int:
        active_expert_params = 3 * self.hidden_size * self.expert_intermediate_size * self.top_k
        return active_expert_params + self.shared_params_per_layer + self.router_params_per_layer

    @property
    def active_params(self) -> int:
        return self.active_params_per_layer * self.layers


def format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return str(value)


def summarize_plan(dense: DenseMlpShape, moe: MoeMlpShape) -> list[str]:
    dense_total = dense.total_params
    moe_total = moe.total_params
    active_total = moe.active_params
    total_ratio = moe_total / dense_total
    active_ratio = active_total / dense_total

    return [
        f"dense_mlp_total_params: {format_count(dense_total)}",
        f"moe_mlp_total_params: {format_count(moe_total)} ({total_ratio:.3f}x dense MLP)",
        f"moe_active_mlp_params: {format_count(active_total)} ({active_ratio:.3f}x dense MLP)",
        f"dense_params_per_layer: {format_count(dense.params_per_layer)}",
        f"moe_total_params_per_layer: {format_count(moe.total_params_per_layer)}",
        f"moe_active_params_per_layer: {format_count(moe.active_params_per_layer)}",
        f"router_params_per_layer: {format_count(moe.router_params_per_layer)}",
        f"routed_width_total: {moe.num_experts * moe.expert_intermediate_size}",
        f"routed_width_active: {moe.top_k * moe.expert_intermediate_size}",
        f"shared_width_active: {moe.shared_expert_intermediate_size}",
    ]
