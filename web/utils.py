class Row:
    def __init__(self, theme_input, oracle_input, student_model_input):
        self.theme_input = theme_input
        self.oracle_input = oracle_input
        self.student_model_input = student_model_input

    def to_dict(self):
        return {
            "Theme": self.theme_input,
            "Oracle": self.oracle_input,
            "Student Model": self.student_model_input,
            "Link": f"https://example.com/{self.theme_input}",
        }
