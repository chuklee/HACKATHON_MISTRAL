from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder
from logging_config import logger
import os
import shutil


def push_model_to_hugging_face(model_path, repo_id):
    """
    Save the model to the specified path and push to the Hugging Face Hub.

    Parameters:
    - model_path: The local path where the model should be saved.
    - repo_name: The repository name on Hugging Face Model Hub.

    Returns:
    - None
    """
    try:
        if not os.path.exists(os.path.join(model_path, "README.md")):
            readme_path = os.path.join(os.path.dirname(__file__), "README.md")
            shutil.copy(readme_path, model_path)
        # Ensure the repository exists on the Hugging Face Hub
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True)
        logger.info("Created repository: %s", repo_id)
        # Upload the model files to the repository
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            commit_message="Add new model version",
        )
        logger.info("Model uploaded successfully")
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))


def save_model_locally(model, tokenizer, model_path):
    """
    Save the model and tokenizer to the specified path.

    Parameters:
    - model: The model to be saved.
    - tokenizer: The tokenizer to be saved.
    - model_path: The local path where the model and tokenizer should be saved.

    Returns:
    - None
    """
    try:
        logger.info("Saving model and tokenizer to: %s", model_path)
        # Save the model and tokenizer
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        logger.info("Model and tokenizer saved successfully")
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))


def save_model_locally_and_push_to_hugging_face(
    model, tokenizer, model_path, repo_name
):
    """
    Save the model and tokenizer to the specified path and push to the Hugging Face Hub.

    Parameters:
    - model: The model to be saved.
    - tokenizer: The tokenizer to be saved.
    - model_path: The local path where the model and tokenizer should be saved.
    - repo_name: The repository name on Hugging Face Model Hub.

    Returns:
    - None
    """
    try:
        logger.info("Saving model and tokenizer locally")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        logger.info("Saving model and tokenizer locally")
        save_model_locally(model, tokenizer, model_path)

        # Push the model to the Hugging Face Hub
        logger.info("Pushing model to Hugging Face Hub")
        push_model_to_hugging_face(model_path, repo_name)
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
