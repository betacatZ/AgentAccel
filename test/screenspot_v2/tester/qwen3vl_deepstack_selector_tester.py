from .qwen3vl_visionselector_tester import Qwen3VLVisionSelectorTester
from models.token_compression.deepstack_selector_model import (
    Qwen3VLForConditionalGeneration_DeepstackSelector,
)
from transformers import GenerationConfig
import torch


class Qwen3VLDeepstackSelectorTester(Qwen3VLVisionSelectorTester):
    def __init__(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        super().__init__(model_path, device, **kwargs)
        self.model = Qwen3VLForConditionalGeneration_DeepstackSelector.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        budgets = kwargs.get("budgets", None)
        if budgets is not None:
            self.model.visual.budgets = budgets
        generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True).to_dict()
        generation_config.update(do_sample=False, temperature=0.0)
        self.model.generation_config = GenerationConfig(**generation_config)
