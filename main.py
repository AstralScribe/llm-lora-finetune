from src.trainer import LLMTrainer
import src.config as config


peft_config = config.PeftConfig("llama2")()
bnb_config = config.BnbConfig("bnb_config")()
training_args = config.TrainingArgs("trainer")()


llm = LLMTrainer(
    llm_model = "llama2",
    hf_access_token = "",
    bits_and_bytes_config = bnb_config,
    lora_config = peft_config,
    training_args = training_args,
    data_path = "",
    device = "",
)

llm.trainer.train()