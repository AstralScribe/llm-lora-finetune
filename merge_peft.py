import torch
from peft import AutoPeftModelForCausalLM

dtypes = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


def merge_model(
    peft_model: str, dtype: str, token: str, save_path: str, device: str = "auto"
) -> None:
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model, device_map=device, torch_dtype=dtypes[dtype], token=token
    )

    model = model.merge_and_unload()
    model.save_pretrained(save_path)

    return None
