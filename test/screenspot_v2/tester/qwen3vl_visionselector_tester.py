from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer
from transformers import GenerationConfig

from models.token_compression.selector_model import (
    Qwen3VLForConditionalGeneration_Selector,
)

from .base_tester import BaseTester
from .util import convert_pil_image_to_base64
import json


class Qwen3VLVisionSelectorTester(BaseTester):
    def __init__(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        super().__init__(model_path, device, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration_Selector.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()

        budgets = kwargs.get("budgets", None)
        if budgets is not None:
            self.model.visual.budgets = budgets
        self.processor = AutoProcessor.from_pretrained(model_path)

        generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True).to_dict()
        generation_config.update(do_sample=False, temperature=0.0)
        self.model.generation_config = GenerationConfig(**generation_config)
        self.system_prompt = self.system_prompt = """
        # Tools

        You may call one or more functions to assist with the user query.

        You are provided with function signatures within <tools></tools> XML tags:
        <tools>
        {
        "type": "function",
        "function": {
            "name": "mobile_use",
            "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
        * This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
        * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
        * The screen's resolution is 999x999.
        * Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
            "parameters": {
            "properties": {
                "action": {
                "description": "The action to perform. The available actions are:
        * `click`: Click the point on the screen with coordinate (x, y).",
                "enum": [
                    "click",
                ],
                "type": "string"
                },
                "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`.",
                "type": "array"
                },
            },
            "required": ["action"],
            "type": "object"
            }
        }
        }
        </tools>

        For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
        <tool_call>
        {"name": <function-name>, "arguments": <args-json-object>}
        </tool_call>

        # Response format

        Response format for every step:
        1) Thought: one concise sentence explaining the next move (no multi-step reasoning).
        2) Action: a short imperative describing what to do in the UI.
        3) A single <tool_call>...</tool_call> block containing only the JSON.

        Rules:
        - Output exactly in the order: Thought, Action, <tool_call>.
        - Be brief: one sentence for Thought, one for Action.
        - Do not output anything else outside those three parts.
        """

    def generate_click_coordinate(self, instruction: str, image: Image.Image):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + convert_pil_image_to_base64(image)},
                    },
                ],
            },
        ]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # text_input = text_input + self.guide_text

        inputs = self.processor(text=[text_input], images=[image], padding=True, return_tensors="pt").to(
            self.model.device
        )

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        # response = self.guide_text + response
        # cut_index = response.rfind("}")
        # if cut_index != -1:
        #     response = response[: cut_index + 1]

        coordinates = self._parse_output(response)

        return coordinates, response

    def _parse_output(self, response: str) -> tuple[float, float] | None:
        try:
            action = json.loads(response.split("<tool_call>\n")[1].split("\n</tool_call>")[0])
            if action["arguments"]["action"] == "click":
                coordinates = action["arguments"]["coordinate"]
                coordinates = [coordinates[0] / 999, coordinates[1] / 999]
                if isinstance(coordinates, list) and len(coordinates) == 2:
                    return tuple(coordinates)
        except Exception:
            return None
        return None
