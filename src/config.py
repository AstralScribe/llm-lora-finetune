from peft import LoraConfig
from transformers import BitsAndBytesConfig, TrainingArguments
import yaml

class PeftConfig:
    def __init__(self) -> None:
        with open("configs/lora_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        self.r = config.get("r", 8)
        self.target_modules = config.get("targeet_modules", ["q_proj", "v_proj"])
        self.lora_alpha = config.get("lora_alpha", 8)
        self.lora_dropout = config.get("dropout", 0)
        self.fan_in_fan_out =  config.get("fan_in_fan_out", False)
        self.bias = config.get("bias", "none")
        self.task_type = config.get("task_type", "CAUSAL_LM")
    
    