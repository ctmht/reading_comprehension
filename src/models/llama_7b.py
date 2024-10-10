import os
from typing import Any, Union

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments, logging,
                          pipeline)
from trl import SFTTrainer

from src.data.hugging_face_data_loader import HuggingFaceDataLoader
from src.models.generic_model import GenericModel


class Llama7B(GenericModel):
    """Interface for generic Llama7B model with full finetuning"""

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
        Function handling the finetuning procedure of the LLM
        on the given training data pairs in (train_in, train_out)
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
            output = f"### Human: {example['instruction']}\n### Assistant: {example['response']}"
            return output

        # Initialize the SFTTrainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            formatting_func=formatting_prompts_func,
            tokenizer=self.tokenizer,
        )

        # Train the model
        trainer.train()

        # Save the model
        trainer.save_model("./finetuned_model")

    def prompt(self, prompt: str, **kwargs) -> Any:
        """
        Function handling the generation of output
        """
        if self.model is None:
            raise ValueError("Model has not been finetuned or loaded yet.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs.input_ids, max_new_tokens=100, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save(self, path: Union[str, os.path], **kwargs) -> None:
        """
        Save pickled model at the specified path
        Or however we decide to save it...
        """
        if self.model is None:
            raise ValueError("Model has not been finetuned or loaded yet.")

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: Union[str, os.path], **kwargs) -> None:
        """
        Save pickled model at the specified path
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
