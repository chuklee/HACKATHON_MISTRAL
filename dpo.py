import dotenv

dotenv.load_dotenv()

import os
import torch
from datasets import load_dataset
from logging_config import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


from trl import DPOTrainer, AutoModelForCausalLMWithValueHead
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import gc
from utils.save_model import save_model_locally, save_model_locally_and_push_to_hugging_face
import heapq
from generate_dataset import create_similar_dataset
from accelerate import PartialState

# Examples for generating dataset.

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_top_2_exercises_rankings(dpo_trainer):
    data = dpo_trainer.get_train_dataloader()

    exercises_rankings = []

    for batch in data:
        recap = dpo_trainer.get_batch_loss_metrics(dpo_trainer.model, batch)
        exercises_rankings.append((batch['prompt'], recap[1]['rewards/chosen'].item()))
        del recap
        gc.collect()


    top_2_exercises_rankings = heapq.nsmallest(4, exercises_rankings, key=lambda x: x[1])
    return [exercise[0] for exercise in top_2_exercises_rankings]

def instantiate_trainer(model, tokenizer, train_dataset):
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        max_steps=100,
        save_strategy="no",
        logging_steps=1,
        output_dir='dpo_mistral',
        optim="paged_adamw_32bit",
        warmup_steps=50,
        bf16=True,
       # report_to="wandb",
        remove_unused_columns=False,
    )
    dpo_trainer = DPOTrainer(
        model,
        None,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.1,
        max_prompt_length=1024,
        max_length=1536,
    )

    return dpo_trainer


def get_train_dataset(dataset_path: str, tokenizer):
    ds = load_dataset('json', data_files=dataset_path, split="train")

    def transform_to_conversation(prompt, response):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

    def process(row):    
        chosen_conversation = transform_to_conversation(row["prompt"], row["chosen"])
        row["chosen"] = tokenizer.apply_chat_template(chosen_conversation, tokenize=False)
        rejected_conversation = transform_to_conversation(row["prompt"], row["rejected"])
        row["rejected"] = tokenizer.apply_chat_template(rejected_conversation, tokenize=False)
        return row

    train_dataset = ds.map(
        process,
        load_from_cache_file=False,
    )
    return train_dataset


def load_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    return model, tokenizer

def fine_tune(model_name: str, model_path: str, dataset_path: str, conditions, example_question, example_answer):
    logger.info("Fine-tuning model %s with dataset %s", model_name, dataset_path)
    base_model_path = model_path
    NUMBER_OF_EPOCHS = 15
    current_epoch = 1 # Arbitrary number of epochs to run on 6 hours
    model, tokenizer = load_model(model_path)
    train_dataset = get_train_dataset(dataset_path, tokenizer)
    dpo_trainer = instantiate_trainer(model, tokenizer, train_dataset)
    while current_epoch <= NUMBER_OF_EPOCHS:
        logger.info("Starting epoch %s", current_epoch)
        dpo_trainer.train()
        top_2_exercises_rankings = get_top_2_exercises_rankings(dpo_trainer)

        if current_epoch % 5 == 0 or current_epoch == NUMBER_OF_EPOCHS:
            logger.info(f"Saving model at epoch {current_epoch}")
            model_path = "./dpo_mistral"
            save_model_locally(model, tokenizer, model_path)
            
        if current_epoch != NUMBER_OF_EPOCHS:
            logger.info("Starting dataset creating")
            new_dataset_path = create_similar_dataset(
                top_2_exercises_rankings, 
                "groq_llama3-70b-8192" ,
                model_path, 
                conditions,
                example_question,
                example_answer,
                dataset_path,
                )

            train_dataset = get_train_dataset(new_dataset_path, tokenizer)
            with PartialState().local_main_process_first():
                # tokenize the dataset
                logger.info("Changing the train dataset")
                dpo_trainer.train_dataset = train_dataset.map(dpo_trainer.tokenize_row, num_proc=None)

        else:
            logger.info("Last Epoch")

        current_epoch += 1

    logger.info("Fine-tuning completed successfully")
    del dpo_trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(base_model, "dpo_mistral")
    model = model.merge_and_unload()
    save_model_locally_and_push_to_hugging_face(model, tokenizer, "dpo_mistral", "cvmistralparis/smol")
    return True

#if __name__ == '__main__':
#    model = AutoModelForCausalLM.from_pretrained(
#        "dpo_mistral"
#    )
#    tokenizer = AutoTokenizer.from_pretrained("dpo_mistral")
#    save_model_locally_and_push_to_hugging_face(model, tokenizer, "dpo_mistral", "cvmistralparis/smol")