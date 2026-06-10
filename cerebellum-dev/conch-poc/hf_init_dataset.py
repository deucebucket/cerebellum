import os
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

if not token:
    print("[-] Error: HF_TOKEN not found in .env")
    exit(1)

api = HfApi(token=token)

repo_id = "deucebucket/cerebellum-brainloop"

try:
    print(f"[*] Creating dataset repository: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="dataset", private=False)
    print(f"[+] Successfully created dataset repo: https://huggingface.co/datasets/{repo_id}")
except Exception as e:
    if "already exists" in str(e):
        print(f"[!] Dataset repo {repo_id} already exists.")
    else:
        print(f"[-] Error creating repo: {e}")

# Optional: Upload the 13k text file as an initial commit
try:
    print("[*] Uploading python_stdlib_13k.txt...")
    api.upload_file(
        path_or_fileobj="python_stdlib_13k.txt",
        path_in_repo="python_stdlib_13k.txt",
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("[+] Upload complete.")
except Exception as e:
    print(f"[-] Error uploading file: {e}")
