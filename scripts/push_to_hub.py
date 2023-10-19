import json
import os

from accelerate.state import PartialState
from huggingface_hub import HfApi

from autotrain import logger

TARGET_MODEL_PATH = "/home/abhinand/projects/Chinese-LLaMA-Alpaca-2/models/llms/llama2-7b-pt-600k"
REPO_ID = "abhinand/llama2-7b-tamil-600k-pt-hf"
HF_TOKEN = "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"

if PartialState().process_index == 0:
    logger.info("Pushing model to hub...")
    if os.path.exists(f"{TARGET_MODEL_PATH}/training_params.json"):
        training_params = json.load(open(f"{TARGET_MODEL_PATH}/training_params.json"))
        # training_params.pop("token")
        json.dump(training_params, open(f"{TARGET_MODEL_PATH}/training_params.json", "w"))
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=True)
    api.upload_folder(folder_path=TARGET_MODEL_PATH, repo_id=REPO_ID, repo_type="model")