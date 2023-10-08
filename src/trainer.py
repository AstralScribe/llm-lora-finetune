import datasets
import peft
import utils
from attrs import define
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@define
class LLMTrainerInit:
    llm_model: str
    hf_access_token: str
    bits_and_bytes_config: BitsAndBytesConfig
    lora_config: peft.LoraConfig
    training_args: TrainingArguments
    data_path: str
    device: str
    trust_remote_code: bool = False
    mlm: bool = False


class LLMTrainer(LLMTrainerInit):
    def __init__(self):
        self.model = self._peft_model()
        self.train_dataset = self.create_finetuning_data()
        self.tokenizer = AutoTokenizer(self.llm_model, token=self.hf_access_token)
        self.data_collator = DataCollatorForLanguageModeling(
            self.tokenizer, mlm=self.mlm
        )
        self.trainer = Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            args=self.training_args,
            data_collator=self.data_collator,
        )

    def _tokens_gen(self, texts):
        return self.tokenizer(texts)

    def _data_loader(self) -> datasets.Dataset:
        return utils.data_loader(self.data_path)

    def _peft_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model,
            quantization_config=self.bits_and_bytes_config,
            device=self.device,
            token=self.hf_access_token,
            trust_remote_code=self.trust_remote_code
        )
        model.gradient_checkpointing_enable()
        model = peft.prepare_model_for_kbit_training(model)
        model = peft.get_peft_model(model, self.lora_config)

        return model

    def create_finetuning_data(self) -> datasets.Dataset:
        data = self._data_loader()
        data = data.map(self._tokens_gen, batched=True)

        return data


if __name__ == "__main__":
    ...
