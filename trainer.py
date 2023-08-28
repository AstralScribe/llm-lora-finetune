import datasets
import peft
import transformers
from attrs import define
import utils


@define
class TrainerInit:
    llama_model: str
    hf_access_token: str
    bits_and_bytes_config: transformers.BitsAndBytesConfig
    lora_config: peft.LoraConfig
    tranier_args: transformers.TrainingArguments
    data_path: str
    device: str


class Trainer(TrainerInit):
    def __init__(self):
        self.model = self._peft_model()
        self.train_dataset = self.create_finetuning_data()
        self.trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            args=self.tranier_args,
            data_collator=self.data_collator,
        )
        self.tokenizer = transformers.AutoTokenizer(
            self.llama_model, token=self.hf_access_token
        )

    def _tokens_gen(self, texts):
        return self.tokenizer(texts)

    def _data_loader(self) -> datasets.Dataset:
        return utils.data_loader(self.data_path)

    def _peft_model(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.llama_model,
            quantization_config=self.bits_and_bytes_config,
            device=self.device,
            token=self.hf_access_token,
        )
        model.gradient_checkpointing_enable()
        model = peft.prepare_model_for_kbit_training(model)
        model = peft.get_peft_model(model, self.lora_config)

        return model

    def create_finetuning_data(self, file_path) -> datasets.Dataset:
        data = self._data_loader()
        data = data.map(self._tokens_gen, batched=True)

        return data
