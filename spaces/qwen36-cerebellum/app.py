import subprocess
import sys
import os

os.environ["CMAKE_ARGS"] = "-DGGML_CUDA=ON"
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "llama-cpp-python==0.3.19",
    "--no-cache-dir", "--quiet"
])

import spaces
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_REPO = "deucebucket/Qwen3.6-27B-Cerebellum-v4-GGUF"
MODEL_FILE = "Qwen3.6-27B-Cerebellum-v4-Q2_K_Mixed.gguf"
MODEL_PATH = None
LLM = None


def ensure_model():
    global MODEL_PATH
    if MODEL_PATH is None:
        MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return MODEL_PATH


@spaces.GPU
def generate(message, history, system_prompt, max_tokens, temperature, top_p):
    global LLM

    model_path = ensure_model()

    if LLM is None:
        LLM = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = LLM.create_chat_completion(
        messages=messages,
        max_tokens=int(max_tokens),
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    partial = ""
    for chunk in response:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            partial += delta["content"]
            yield partial


DESCRIPTION = """# Qwen 3.6 27B — Cerebellum v4 (12 GB)

Ablation-guided mixed-precision quantization. 181 per-tensor overrides.
**75% HumanEval** · **95% ARC** · **91% HellaSwag** · **77% MMLU-Redux** at 12 GB.

[Model Card](https://huggingface.co/deucebucket/Qwen3.6-27B-Cerebellum-v4-GGUF) · [Method](https://github.com/deucebucket/cerebellum)

*ZeroGPU — first message loads the model, subsequent messages are fast.*
"""

demo = gr.ChatInterface(
    fn=generate,
    title="Qwen 3.6 27B — Cerebellum v4",
    description=DESCRIPTION,
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant.", label="System Prompt", lines=2),
        gr.Slider(64, 2048, value=512, step=64, label="Max Tokens"),
        gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top P"),
    ],
    examples=[
        ["Write a Python function that finds the longest palindromic substring."],
        ["Explain the difference between TCP and UDP like I'm 10 years old."],
        ["What are the key differences between transformers and state space models?"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
