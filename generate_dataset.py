import dotenv

dotenv.load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableParallel
import torch

from langchain_core.pydantic_v1 import BaseModel, Field
import json
import concurrent.futures
import hashlib
from typing import Optional
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


MODEL_PATH = "models.json"
with open(MODEL_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for category in data.values():
    for model_name in category:
        modified_model_name = "_".join(model_name.split("_")[1:])
        provider = model_name.split("_")[0]
        if provider == "groq":
            models[model_name] = ChatGroq(model=modified_model_name)
        elif provider == "hf":
            models[model_name] = HuggingFacePipeline.from_model_id(
                model_id="mistralai/Mistral-7B-v0.1",
                task="text-generation",
                device=0,
                model_kwargs={"do_sample": True},
                batch_size=4,
                pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.2, "repetition_penalty": 1.1},
            )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "A theme is given, and you need to provide subcategories that are related to the main theme.",
        ),
        (
            "human",
            "Give 2 diversified subcategories of the following main theme:  {text}",
        ),
    ]
)

prompt_data_generation = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a synthetic data generator. Your task is to generate a dataset based on a given theme and category.
Create 8 questions/answer within the specified category, ensuring they gradually increase in complexity.""",
        ),
        (
            "human",
            """Generate a synthetic dataset with the following theme: {text}. Please be sure to respect these {conditions}.
---
Question are as follow: {example_question}
---
Answer are as follow: {example_answer}
""",
        ),
    ]
)

# prompt_data_generation_no_condition = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """You are a synthetic data generator. Your task is to generate a dataset based on a given theme and category.
#        Create 2 questions within the specified category, ensuring they gradually increase in complexity. The last question should be very challenging.""",
#         ),
#         ("human", "Generate a synthetic dataset with the following theme: {text}."),
#     ]
# )


class SubCategories(BaseModel):
    subcategories: list[str] = Field(description="Names of the subcategories")


class DatasetExample(BaseModel):
    question: str = Field(description="The question to ask")
    answer: str = Field(description="The answer to the question")


class DatasetExamples(BaseModel):
    examples: list[DatasetExample] = Field(description="List of examples")


class FinalDatasetExemple(BaseModel):
    prompt: str
    chosen: str
    rejected: str


def generate_rejected(prompts: list[str], student_model: BaseChatModel):
    # rejected = []
    # runnables = {
    #     f"{i}": (ChatPromptTemplate.from_template(prompt) | student_model)
    #     for i, prompt in enumerate(prompts)
    # }
    # map_chain = RunnableParallel(**runnables)  # type: ignore
    # outputs = map_chain.invoke({})
    # rejected = [output for output in outputs.values()] if isinstance(student_model, HuggingFacePipeline) else [output.content for output in outputs.values()] 
    rejected = student_model.batch([{"question": prompt for prompt in prompts}])
    return rejected
    for prompt in prompts:
        runnable = ChatPromptTemplate.from_template(prompt) | student_model
        rejected.append(runnable.invoke({}))
    return rejected


def generate_category(
    theme: str,
    category: str,
    dataset: list[FinalDatasetExemple],
    oracle_model: BaseChatModel,
    student_model: BaseChatModel,
    conditions: Optional[str],
    example_question: Optional[str],
    example_answer: Optional[str],
):

    runnable_dataset_generation = (
        prompt_data_generation
        | oracle_model.with_structured_output(schema=DatasetExamples)
    )
    try:
        print(f"Generating Dataset Question for category: {category}")
        generated_examples: DatasetExamples = runnable_dataset_generation.invoke(
            {
                "text": f"Theme: {theme}, Category: {category}",
                "conditions": conditions,
                "example_question": example_question,
                "example_answer": example_answer,
            }
        )  # type: ignore
        print(f"Generating Rejected for category: {category}")
        rejecteds = generate_rejected(
            [example.question for example in generated_examples.examples], student_model
        )
        for example, rejected in zip(generated_examples.examples, rejecteds):
            dataset.append(
                FinalDatasetExemple(
                    prompt=example.question,
                    chosen=example.answer,
                    rejected=rejected,  # type: ignore
                )
            )
        print(f"Generated dataset for category: {category}")
    except Exception as e:

        print(f"Failed to generate dataset for category: {category}, Error: {e}")


def generate_dataset(
    theme: str,
    oracle_model_id: str,
    student_model_id: str,
    conditions: str,
    example_question: str,
    example_answer: str,
) -> list[FinalDatasetExemple]:
    oracle_model = models[oracle_model_id]
    student_model = models[student_model_id]
    print("Start")
    print(student_model.invoke("Fait une fonction palidrome"))
    runnable = prompt | oracle_model.with_structured_output(schema=SubCategories)
    categories: SubCategories = runnable.invoke({"text": theme})  # type: ignore
    print(categories.subcategories)
    dataset: list[FinalDatasetExemple] = []

    # for category in categories.subcategories:
    #     print("Generating dataset for category: ", category)
    #     generate_category(theme, category, dataset, oracle_model, student_model)
    def worker(category):
        return generate_category(
            theme,
            category,
            dataset,
            oracle_model,
            student_model,
            conditions,
            example_question,
            example_answer,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(worker, categories.subcategories)

    return dataset


def dump_dataset(
    dataset: list[FinalDatasetExemple], oracle_model_id: str, student_model_id: str
) -> str:
    final_dataset = json.dumps(
        [{"id": i} | example.dict() for i, example in enumerate(dataset)], indent=4
    )
    # Generate a hash of the final dataset
    dataset_hash = hashlib.sha256(final_dataset.encode()).hexdigest()
    dataset_uuid = dataset_hash[:32]
    oracle_model_id = oracle_model_id.replace("/", "_")
    student_model_id = student_model_id.replace("/", "_")
    dataset_file_path = (
        f"datasets/{oracle_model_id}_{student_model_id}_{dataset_uuid}.json"
    )
    with open(dataset_file_path, "w") as f:
        f.write(final_dataset)
        f.close()
    return dataset_file_path


def create_dataset(
    theme,
    oracle_model_id,
    student_model_id,
    conditions,
    example_question,
    example_answer,
):
    dataset = generate_dataset(
        theme,
        oracle_model_id,
        student_model_id,
        conditions,
        example_question,
        example_answer,
    )
    return dump_dataset(dataset, oracle_model_id, student_model_id)


if __name__ == "__main__":
    theme = """Python Coding Interview Exercises on Data Structures and Algorithms"""
    conditions = 'Each question must present only the function signature formatted as follows: `def name_of_the_function(parameter_of_the_function):\\n"""docstring"""'
    example_question = '''
    from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: """ Check if in given list of numbers, are any two numbers closer to each other than given threshold. """
    '''
    example_answer = """
    for idx, elem in enumerate(numbers): for idx2, elem2 in enumerate(numbers): if idx != idx2: distance = abs(elem - elem2) if distance < threshold: return True return False
    """
    path = create_dataset(
        theme,
        "groq_llama3-70b-8192",
        "hf_mistralai/Mistral-7B-v0.1",
        conditions,
        example_question,
        example_answer,
    )
    print(path)
