import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


from trl import DPOTrainer, AutoModelForCausalLMWithValueHead
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import gc
from utils.save_model import save_model_locally
import heapq


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


if __name__ == "__main__":
    new_model = "Smol-mistral-7B"
    hf_token = "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"
    ################
    # Model & Tokenizer
    ################
    model_path = 'mistralai/Mistral-7B-v0.1'

    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        )

    model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    ################
    # Dataset
    ################
    dataset_path = 'dataset.json'
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

    ################
    # Training
    ################

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        max_steps=400,
        save_strategy="no",
        logging_steps=1,
        output_dir='dpo_gemma',
        optim="paged_adamw_32bit",
        warmup_steps=100,
        bf16=True,
        report_to="wandb",
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

#    dpo_trainer.train()

    data = dpo_trainer.get_train_dataloader()

    exercises_rankings = []

    for batch in data:
        recap = dpo_trainer.get_batch_loss_metrics(dpo_trainer.model, batch)
        exercises_rankings.append((batch['prompt'], recap[1]['rewards/chosen'].item()))
        del recap
        gc.collect()


    top_2_exercises_rankings = heapq.nsmallest(2, exercises_rankings, key=lambda x: x[1])
    print(top_2_exercises_rankings)


    # Save artifacts
#    dpo_trainer.model.save_pretrained("final_checkpoint")
#    tokenizer.save_pretrained("final_checkpoint")
#
#    # Flush memory
#    del dpo_trainer, model
#    gc.collect()
#    torch.cuda.empty_cache()
#
#    # Reload model in FP16 (instead of NF4)
#    base_model = AutoModelForCausalLM.from_pretrained(
#        model_path,
#        return_dict=True,
#        torch_dtype=torch.float16,
#    )
#    tokenizer = AutoTokenizer.from_pretrained(model_path)
#
#    # Merge base model with the adapter
#    model = PeftModel.from_pretrained(base_model, "final_checkpoint")
#    model = model.merge_and_unload()
#
#    save_model_locally(model, tokenizer, "./dpo_mistral")


    # Save model and tokenizer
    #model.save_pretrained(new_model)
    #tokenizer.save_pretrained(new_model)

    # Push them to the HF Hub
    #model.push_to_hub(new_model, use_temp_dir=False, token=hf_token)
    #tokenizer.push_to_hub(new_model, use_temp_dir=False, token=hf_token)