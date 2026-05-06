#!/usr/bin/env python3
"""Patch a qwen-3.5-style chat template to bake a VADUGWI steering line into
the system role.

Behavior after patching:
- If the runtime caller supplies a system message, the VADUGWI steering is
  prepended to it (still one system block, never two).
- If the runtime caller supplies no system message, a synthetic system block
  is emitted containing only the steering. Native chat-template behavior —
  the model can't tell the steering came from outside.

This is the runtime-equivalent of writing the patched template back into the
GGUF's `tokenizer.chat_template` metadata field. We use --chat-template-file
on llama-server to test it without actually rewriting the GGUF.
"""
import argparse
from pathlib import Path


HERE = Path(__file__).parent


def patch_template(orig: str, steering: str) -> str:
    """Insert steering content into the qwen-3.5 chat template's no-tools
    system-block emission section."""
    # Original block (lines 61-66 in the qwen 3.5 9b template):
    needle = (
        "{%- else %}\n"
        "    {%- if messages[0].role == 'system' %}\n"
        "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n"
        "        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}\n"
        "    {%- endif %}\n"
        "{%- endif %}\n"
    )
    if needle not in orig:
        raise SystemExit(
            "patch needle not found in template — verify this is qwen 3.5 9b's "
            "chat template, or update the needle to match a new revision."
        )

    # Steering must survive being embedded as a jinja string literal.
    # Backslash-escape backslashes and single quotes; convert real newlines
    # to escaped \n so the template's quoted form keeps them.
    safe = (
        steering
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
    )

    replacement = (
        "{%- else %}\n"
        f"    {{%- set vadugwi_steering = '{safe}' %}}\n"
        "    {%- if messages[0].role == 'system' %}\n"
        "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n"
        "        {{- '<|im_start|>system\\n' + vadugwi_steering + '\\n\\n' + content + '<|im_end|>\\n' }}\n"
        "    {%- else %}\n"
        "        {{- '<|im_start|>system\\n' + vadugwi_steering + '<|im_end|>\\n' }}\n"
        "    {%- endif %}\n"
        "{%- endif %}\n"
    )
    return orig.replace(needle, replacement, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-template", default="/tmp/qwen_template_orig.jinja")
    ap.add_argument("--steering-file", required=True,
                    help="Path to a steering .txt (e.g. bad_dog.txt) — its content becomes the synthetic system block prefix.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    orig = Path(args.orig_template).read_text()
    steering = Path(args.steering_file).read_text().strip()
    patched = patch_template(orig, steering)
    Path(args.out).write_text(patched)
    print(f"patched template written to {args.out} ({len(patched)} chars; "
          f"steering injection: {len(steering)} chars)")


if __name__ == "__main__":
    main()
