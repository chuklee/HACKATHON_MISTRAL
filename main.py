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
            oracle = data.get("oracle", "groq_llama3-70b-8192")
            student_model = data.get("student_model", "groq_gemma-7b-it")
            condition = data.get("condition", "")
            question_example = data.get("question_example", "")
            answer_example = data.get("answer_example", "")
            logging.info("Received POST request with data: %s", data)

            if oracle not in list_oracle:
                error_message = f"Oracle model {oracle} not found"
                logging.error(error_message)
                return jsonify(error=error_message), 400

            if student_model not in list_student:
                error_message = f"Student model {student_model} not found"
                logging.error(error_message)
                return jsonify(error=error_message), 400

<<<<<<< HEAD
            #dataset_path = create_dataset(theme, oracle, student_model, condition, question_example, answer_example)
            dataset_path = wait_10_seconds()
            response_message = f"Dataset created successfully at {dataset_path}"
=======
            #dataset_path = create_dataset(theme, oracle, student_model)
           #response_message = f"Dataset created successfully at {dataset_path}"
            response_message = f"Dataset created successfully at "
>>>>>>> a0e2d765f43817404ff7c41e5ea691ba0d757703
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=105, debug=True)