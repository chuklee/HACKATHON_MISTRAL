"""
This is the main file for the API.
"""

import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# Set up logging
LOG_FOLDER = "log"
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

log_filename = os.path.join(LOG_FOLDER, datetime.now().strftime("%Y-%m-%d") + ".log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@app.route("/create_model", methods=["GET", "POST"])
def main():
    """
    Main function for the API.
    It will be run when the server is called with a GET or POST request on the address '/'.
    """
    list_oracle = ["groq_llama3-70b-8192", "groq_mixtral-8x7b-32768"]
    list_student = ["groq_gemma-7b-it", "groq_llama3-8b-8192"]
    if request.method == "POST":
        try:
            data = request.get_json(force=True)
            theme = data.get(
                "theme", "Coding Data Structures and Algorithms exercises in Python"
            )
            oracle = data.get("model", "groq_llama3-70b-8192")
            student_model = data.get("student_model", "groq_gemma-7b-it")

            logging.info("Received POST request with data: %s", data)

            if oracle not in list_oracle:
                error_message = f"Oracle model {oracle} not found"
                logging.error(error_message)
                return jsonify(error=error_message), 400

            if student_model not in list_student:
                error_message = f"Student model {student_model} not found"
                logging.error(error_message)
                return jsonify(error=error_message), 400

            response_message = f"Hello World {theme}!"
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
    app.run(host="0.0.0.0", port=105)
