from peft import LoraConfig
from transformers import BitsAndBytesConfig, TrainingArguments
import yaml
import torch

class PeftConfig:
    def __init__(self) -> None:
        with open("configs/lora_config.yaml", "r") as f:
            config = yaml.safe_load(f)

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
    def __init__(self) -> None:
        with open("configs/bits_and_bytes_config.yaml", "r") as f:
            config = yaml.safe_load(f)

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
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype
        )


class TrainingArgs:
    def __init__(self):
        ...
    def __call__(self) -> TrainingArguments:
        ...