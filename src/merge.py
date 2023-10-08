import torch
from attrs import define
from peft import AutoPeftModelForCausalLM


@define
class ModelMerge:
    peft_model: str
    dtype: str
    token: str
    save_path: str
    device: str = "auto"
    dtypes: dict = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }

    def merge(self) -> None:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.peft_model,
                device_map=self.device,
                torch_dtype=self.dtypes[self.dtype],
                token=self.token,
            )
            model = model.merge_and_unload()
            print("Model merging successful.")
            model.save_pretrained(self.save_path)
            print("Model saved.")
            return None
        
        except Exception as err:
            print(err)
            print("Model merger failed.")
            return None

        
