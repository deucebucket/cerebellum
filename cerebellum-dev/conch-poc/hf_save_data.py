import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

if not token:
    print("[-] Error: HF_TOKEN not found in .env")
    exit(1)

api = HfApi(token=token)
repo_id = "deucebucket/cerebellum-brainloop"

files_to_upload = [
    "checkpoints-fusion-13k/fused_refiners.pt",
    "python_13k_rag.bin",
    "python_13k_deltas.pt",
    "humaneval_samples_conch.jsonl",
    "eval_rag.log"
]

print(f"[*] Uploading data to HF Dataset: {repo_id}")
for fpath in files_to_upload:
    if os.path.exists(fpath):
        print(f"[*] Uploading {fpath}...")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=os.path.basename(fpath),
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"[+] {fpath} uploaded.")
    else:
        print(f"[!] Warning: {fpath} not found.")

print("[+] All data saved to Hugging Face.")
