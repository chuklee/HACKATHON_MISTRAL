import json

MODEL_PATH = "../models.json"

def read_json(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
        return data
    
def load_models():
    models = read_json(MODEL_PATH)
    oracles = models["oracle"]
    students = models["student"]
    return oracles, students

load_models()

class Row:
    def __init__(self, theme_input, oracle_input, student_model_input):
        self.theme_input = theme_input
        self.oracle_input = oracle_input
        self.student_model_input = student_model_input

    def to_dict(self):
        return {
            "Theme": self.theme_input,
            "Oracle": self.oracle_input,
            "Student": self.student_model_input,
            "Link": f"https://example.com/{self.theme_input}",
        }
