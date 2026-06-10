from llama_cpp import Llama
import sys

def smoke_test(model_path):
    print(f"[*] Loading model: {model_path}")
    llm = Llama(model_path=model_path, n_ctx=512, n_gpu_layers=-1)
    
    print("\n[1] Basic Coherence Test:")
    prompt = "Explain why the sky is blue in one sentence."
    print(f"Prompt: {prompt}")
    output = llm(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", max_tokens=60, stop=["<|im_end|>"])
    text = output['choices'][0]['text']
    print(f"Response: {text.strip()}")
    
    print("\n[2] Reasoning Test (Math):")
    prompt = "If I have 3 apples and you give me 5 more, but I eat 2, how many do I have?"
    print(f"Prompt: {prompt}")
    output = llm(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", max_tokens=60, stop=["<|im_end|>"])
    text = output['choices'][0]['text']
    print(f"Response: {text.strip()}")

if __name__ == "__main__":
    smoke_test("qwen2.5-3b-unrolled.gguf")
