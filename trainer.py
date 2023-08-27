import torch
import transformers
import peft
import datasets
import parameters
from typing import Tuple
from attrs import define


def dataloader(file_path: str) -> datasets.Dataset:
    return datasets.load_from_disk(file_path)


@define
class TrainerInit:
    llama_model: str
    hf_access_token: str
    lora_config: peft.LoraConfig
    tranier_args: transformers.TrainingArguments


class Trainer(TrainerInit):
    def __init__(self):
        self.model = self.peft_model()
        self.train_dataset = self.create_finetuning_data()
        self.trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            args=self.tranier_args,
            data_collator=self.data_collator,
        )

        