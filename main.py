from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    list_oracle = ['groq_llama3-70b-8192', 'groq_mixtral-8x7b-32768']
    list_student = ['groq_gemma-7b-it', 'groq_llama3-8b-8192']
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            theme = data.get('theme', 'Coding Data Structures and Algorithms exerices in Python')
            oracle = data.get('model', 'groq_llama3-70b-8192')
            student_model = data.get('student_model', 'groq_gemma-7b-it')
            if oracle not in list_oracle:
                return jsonify(error=f"Oracle model {oracle} not found"), 400
            if student_model not in list_student:
                return jsonify(error=f"Student model {student_model} not found"), 400
            return jsonify(message=f"Hello World {theme}!")
        except Exception as e:
            return jsonify(error=str(e)), 400
    return "Hello World!"

@app.route('/get_response', methods=['GET'])
def get_response():
    return "Hello World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)