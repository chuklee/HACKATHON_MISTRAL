from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            theme = data.get('theme', 'Coding Data Structures and Algorithms exerices in Python')
            oracle = data.get('model', 'MistralLarge') #TODO Vérifier que c'est bien dans une liste
            student_model = data.get('student_model', 'Phi')
            return jsonify(message=f"Hello World {theme}!")
        except Exception as e:
            return jsonify(error=str(e)), 400
    return "Hello World!"

@app.route('/get_response', methods=['GET'])
def get_response():


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)