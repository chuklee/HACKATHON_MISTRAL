"""
This is the main file for the API.
"""

import os
import logging
import json
from datetime import datetime
from flask import Flask, request, jsonify, Response, render_template_string
from logging_config import logger
from generate_dataset import create_dataset
from utils.wait_function import wait_10_seconds
from dpo import fine_tune
from utils.save_model import push_model_to_hugging_face
app = Flask(__name__)


@app.route("/create_model", methods=["GET", "POST"])
def main():
    """
    Main function for the API.
    It will be run when the server is called with a GET or POST request on the address '/'.
    """
    with open("models.json", "r", encoding="utf-8") as file:
        data = json.load(file)
        list_oracle = data["oracle"]
        list_student = data["student"]
    
    if request.method == "POST":
        try:
            data = request.get_json(force=True)
            theme = data.get(
                "theme",
                "Function Implementation of DataStructre and Algorithms in Python",
            )

            conditions = 'Each question must present only the function signature formatted as follows: `def name_of_the_function(parameter_of_the_function):\\n"""docstring"""'
            example_question = '''
            from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: """ Check if in given list of numbers, are any two numbers closer to each other than given threshold. """
            '''
            example_answer = """
            for idx, elem in enumerate(numbers): for idx2, elem2 in enumerate(numbers): if idx != idx2: distance = abs(elem - elem2) if distance < threshold: return True return False
            """
            oracle = data.get("oracle", "groq_llama3-70b-8192")
            student_model = data.get("student_model", "hf_mistralai/Mistral-7B-v0.1")
            condition = data.get("condition", conditions)
            question_example = data.get("question_example", example_question)
            answer_example = data.get("answer_example", example_answer)
            logging.info("Received POST request with data: %s", data)

            if oracle not in list_oracle:
                error_message = f"Oracle model {oracle} not found"
                logging.error(error_message)
                return jsonify(error=error_message), 400

            if student_model not in list_student:
                error_message = f"Student model {student_model} not found"
                logging.error(error_message)
                return jsonify(error=error_message), 400
            dataset_path = create_dataset(theme, oracle, student_model, condition, question_example, answer_example)
            logging.info("Dataset created successfully at %s", dataset_path)
            fine_tune(student_model, "mistralai/Mistral-7B-v0.1", dataset_path, condition, question_example, answer_example)
            logging.info("Model fine-tuned successfully")
            response_message = f"Dataset created successfully at {dataset_path}"
            logging.info("Response: %s", response_message)
            return jsonify(message=response_message)

        except Exception as e:
            logging.exception("Exception occurred while processing POST request")
            return jsonify(error=str(e)), 400

    logging.info("Received GET request")
    return "Hello World!"


@app.route("/get_response", methods=["GET"])
def get_response():
    logging.info("Received GET request at /get_response")
    return "Hello World!"


@app.route("/update_env", methods=["POST"])
def update_env():
    """
    Update the environment variables.
    """
    data = request.get_json(force=True)
    logging.info("Received POST request with data: %s", data)
    if not os.path.exists(".env"):
        with open(".env", "w", encoding="utf-8") as file:
            for key, value in data.items():
                file.write(f"{key}={value}\n")
    else:
        # Read the existing environment variables and update the values
        with open(".env", "r", encoding="utf-8") as file:
            lines = file.readlines()
        with open(".env", "w", encoding="utf-8") as file:
            for line in lines:
                key, _ = line.split("=")
                if key in data:
                    file.write(f"{key}={data[key]}\n")
                else:
                    file.write(line)
    return jsonify(message="Environment variables updated successfully")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=105, debug=False)