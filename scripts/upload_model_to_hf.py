"""
Upload a local folder (trained model) to Hugging Face Model Hub. Script used by GitHub Actions 'release' job.
"""
import argparse
import os
import sys
from huggingface_hub import HfApi, Repository, upload_folder

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-folder", required=True, help="Local folder containing model (saved via model.save_pretrained())")
    p.add_argument("--repo-id", required=True, help="HF model repo id like username/repo-name")
    p.add_argument("--token", required=True, help="HF token")
    args = p.parse_args()

    model_folder = args.model_folder
    repo_id = args.repo_id
    token = args.token

    api = HfApi()
    # Create model repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, token=token)
        print(f"Created model repo: {repo_id}")
    except Exception as e:
        print(f"Repo probably exists or creation failed: {e}")

    print(f"Uploading folder {model_folder} to {repo_id} ...")
    upload_folder(
        folder_path=model_folder,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        path_in_repo=""
    )
    print("Upload finished.")

if __name__ == "__main__":
    main()
