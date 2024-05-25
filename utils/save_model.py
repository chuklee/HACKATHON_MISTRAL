from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder
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
        print(f"Created repository: {repo_id}")
        # Upload the model files to the repository
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            commit_message="Add new model version",
        )
    except Exception as e:
        print(f"An error occurred: {str(e)}")


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
        # Save the model and tokenizer
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


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
        # Save the model locally
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        save_model_locally(model, tokenizer, model_path)

        # Push the model to the Hugging Face Hub
        print("pushing model to hugging face")
        push_model_to_hugging_face(model_path, repo_name)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
