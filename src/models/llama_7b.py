import os
from typing import Any, Union

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from src.data.hugging_face_data_loader import HuggingFaceDataLoader
from src.models.generic_model import GenericModel


class Llama7B(GenericModel):
    """Interface for generic Llama7B model with full fine-tuning"""

    def __init__(self, dataset_loader: HuggingFaceDataLoader):
        """
        Initialize the base Llama7B model
        """
        super().__init__()
        self.base_model = "NousResearch/Llama-2-7b-chat-hf"
        self.dataset = dataset_loader.output_dataset("train")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = None

    def finetune(self, **kwargs) -> Any:
        """
        Function handling the fine-tuning procedure of the LLM
        """
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            **kwargs,
        )

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Prepare the dataset
        def formatting_prompts_func(example):
            return f"### Human: {example['instruction']}\n### Assistant: {example['response']}"

        formatted_dataset = self.dataset.map(
            lambda x: {"text": formatting_prompts_func(x)},
            remove_columns=self.dataset.column_names,
        )

        # Tokenize the dataset
        tokenized_dataset = formatted_dataset.map(
            lambda x: self.tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
            ),
            batched=True,
        )

        # Define data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # Train the model
        trainer.train()
        trainer.save_model("./finetuned_model")

    def prompt(self, prompt: str, **kwargs) -> Any:
        """
        Function handling the generation of output
        """
        if self.model is None:
            raise ValueError("Model has not been fine-tuned or loaded yet.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs.input_ids, max_new_tokens=100, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save(self, path: Union[str, os.PathLike], **kwargs) -> None:
        """
        Save the model and tokenizer
        """
        if self.model is None:
            raise ValueError("Model has not been fine-tuned or loaded yet.")

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: Union[str, os.PathLike], **kwargs) -> None:
        """
        Load the model and tokenizer from the specified path
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
