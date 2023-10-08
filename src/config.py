from typing import Dict

import torch
import yaml
from peft import LoraConfig
from transformers import BitsAndBytesConfig, TrainingArguments


class PeftConfig:
    def __init__(self, config_file: str) -> None:
        self.cofig_file = config_file
        with open(f"configs/{self.cofig_file}", "r") as f:
            config: Dict = yaml.safe_load(f)

        self.r = config.get("r", 8)
        self.target_modules = config.get("targeet_modules", ["q_proj", "v_proj"])
        self.lora_alpha = config.get("lora_alpha", 8)
        self.lora_dropout = config.get("dropout", 0)
        self.fan_in_fan_out = config.get("fan_in_fan_out", False)
        self.bias = config.get("bias", "none")
        self.task_type = config.get("task_type", "CAUSAL_LM")

    def __call__(self) -> LoraConfig:
        return LoraConfig(
            r=self.r,
            target_modules=self.target_modules,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            fan_in_fan_out=self.fan_in_fan_out,
            bias=self.bias,
            task_type=self.task_type,
        )


class BnbConfig:
    def __init__(self, config_file: str) -> None:
        with open(f"configs/{config_file}.yaml", "r") as f:
            config: Dict = yaml.safe_load(f)

        dtypes = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }

        self.load_in_4bit = config.get("load_in_4bit", True)
        self.load_in_8bit = config.get("load_in_8bit", False)
        self.bnb_4bit_use_double_quant = config.get("double_quant", True)
        self.bnb_4bit_quant_type = config.get("quant_type", "nf4")
        self.bnb_4bit_compute_dtype = dtypes[config.get("compute_dtype", "fp16")]

    def __call__(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
        )


class TrainingArgs:
    def __init__(self, config_file: str):
        with open(f"configs/{config_file}.yaml", "r") as f:
            config: Dict = yaml.safe_load(f)

        self.per_device_train_batch_size = config.get("per_device_train_batch_size", 2)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
        self.warmup_steps = config.get("warmup_steps", 10)
        self.max_steps = config.get("max_steps", 100)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.fp16 = config.get("fp16", True)
        self.lr_scheduler_type = config.get("lr_scheduler_type", "cosine")
        self.logging_steps = config.get("logging_steps", 5)
        self.output_dir = f'models/{config.get("output_dir", "default")}'
        self.save_steps = config.get("save_steps", 5)
        self.save_total_limit = config.get("save_total_limit", 3)
        self.optim = config.get("optim", "adamw")

    def __call__(self) -> TrainingArguments:
        return TrainingArguments(
            per_device_eval_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            fp16=self.fp16,
            lr_scheduler_type=self.lr_scheduler_type,
            logging_steps=self.logging_steps,
            output_dir=self.output_dir,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            optim=self.optim,
        )
