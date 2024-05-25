from transformers import AutoModel, AutoTokenizer
import os
from utils.save_model import save_model_locally_and_push_to_hugging_face


def test_save_and_push_model():
    model_name = "keeeeenw/MicroLlama"
    model_path = "./test-model"
    repo_name = "cvmistralparis/smol"

    # Load a simple model and tokenizer
    print("getting model")
    model = AutoModel.from_pretrained(model_name)
    print("getting tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure the model directory exists
    print("ensuring model directory exists")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Save the model and tokenizer locally and push to the Hugging Face Hub
    print("saving model locally and pushing to hugging face")
    save_model_locally_and_push_to_hugging_face(model, tokenizer, model_path, repo_name)
