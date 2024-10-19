import os
from typing import Any, Union

import dask.dataframe as dd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (DataCollatorForLanguageModeling, GPT2LMHeadModel,
                          GPT2Tokenizer, get_linear_schedule_with_warmup)

from src.data.hugging_face_data_loader import HuggingFaceDataLoader
from src.models.generic_model import GenericModel


class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized_input = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokenized_input["input_ids"].squeeze()
        attention_mask = tokenized_input["attention_mask"].squeeze()
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class Llama7B(GenericModel):
    """Interface for the GPT-2 model with manual full fine-tuning"""

    def __init__(
        self, dataset_loader: HuggingFaceDataLoader, chunked_data: dd.DataFrame, checkpoint_name_suffix: str
    ):
        """
        Initialize the base GPT-2 model
        """
        super().__init__()
        self.base_model = "gpt2"
        # Convert the Dask DataFrame to a pandas DataFrame for processing
        self.dataset = chunked_data.compute()  # Convert Dask to pandas DataFrame
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None

    def finetune(self, start_epoch=0, num_epochs=3, **kwargs) -> Any:
        """
        Function handling the fine-tuning procedure of the LLM using a manual training loop
        """
        # Load the model or resume from checkpoint
        if start_epoch > 0:
            checkpoint_path = f"data/finetuned/finetuned_model_{self.checkpoint_name_suffix}_epoch_{start_epoch}"
            self.load(checkpoint_path)
            print(f"Resuming training from epoch {start_epoch}")
        else:
            self.model = GPT2LMHeadModel.from_pretrained(self.base_model)
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Prepare the dataset
        def formatting_prompts_func(row):
            return f"### Code:\n{row['code']}\n\n### Summary:\n{row['summary']}"

        self.dataset["text"] = self.dataset.apply(formatting_prompts_func, axis=1)

        # Create custom dataset
        train_dataset = CustomDataset(
            texts=list(self.dataset["text"]),
            tokenizer=self.tokenizer,
            max_length=512,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Create DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=data_collator,
        )

        # Prepare optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        total_steps = num_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.model.train()
        for epoch in range(start_epoch, start_epoch, num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                if step % 10 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item()}")

            # Save checkpoint every epoch
            self.save(f"./finetuned_model_epoch_{epoch+1}")

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
        self.model = GPT2LMHeadModel.from_pretrained(path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def print_dataset_info(self):
        """
        Print information about the dataset
        """
        print("Dataset columns:", self.dataset.columns)
        print("\nFirst few rows of the dataset:")
        print(self.dataset.head())
        print(f"\nTotal number of samples: {len(self.dataset)}")