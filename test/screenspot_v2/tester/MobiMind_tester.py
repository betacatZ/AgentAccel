import json
import logging
import re
from PIL import Image
from typing import Optional, Tuple
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
from .base_tester import BaseTester
from util import convert_pil_image_to_base64, robust_json_loads


def parse_json_response(response_str: str) -> dict:
    """
    解析 JSON 响应

    Args:
        response_str: 模型返回的响应字符串

    Returns:
        解析后的 JSON 对象

    """
    if not response_str or not isinstance(response_str, str):
        logging.error(f"Invalid response: {response_str}")
        raise ValueError(f"无效的响应格式: {response_str}")

    response_str = response_str.strip()

    # 首先尝试直接解析 JSON（guided decoding 输出纯 JSON）
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        pass

    # 如果直接解析失败，尝试提取 JSON 部分（兼容非 guided decoding 的情况）
    try:
        # 方法1: 提取 ```json ... ``` 代码块
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", response_str, re.MULTILINE)
        if json_match:
            json_str = json_match.group(1).strip()
            return json.loads(json_str)

        # 方法2: 提取 ``` ... ``` 代码块
        json_match = re.search(r"```\s*(\{[\s\S]*?\})\s*```", response_str, re.MULTILINE)
        if json_match:
            json_str = json_match.group(1).strip()
            return json.loads(json_str)

        # 方法3: 查找最外层的花括号包围的 JSON
        # 这种方法需要更仔细的处理，避免误匹配嵌套结构
        start_idx = response_str.find("{")
        if start_idx != -1:
            # 从第一个 { 开始，找到匹配的 }
            brace_count = 0
            for i in range(start_idx, len(response_str)):
                if response_str[i] == "{":
                    brace_count += 1
                elif response_str[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response_str[start_idx : i + 1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            continue

        # 如果都失败了
        logging.error("无法在响应中找到有效的 JSON")
        logging.error(f"原始响应: {response_str[:200]}...")
        raise ValueError("无法解析 JSON 响应，响应格式不正确")

    except json.JSONDecodeError as e:
        logging.error(f"JSON 解析失败: {e}")
        logging.error(f"原始响应: {response_str[:200]}...")
        raise ValueError(f"无法解析 JSON 响应: {e}")


class MobiMindTester(BaseTester):
    def __init__(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        super().__init__(model_path, device, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.decider = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        self.grounder = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.decider_prompt = """
        <image>
        You are a phone-use AI agent. Now your task is "{task}". 
        Please provide the next action based on the screenshot. You should do careful reasoning before providing the action.
        Your action space includes:
        - Name: click, Parameters: target_element (a high-level description of the UI element to click).
        Your output should be a JSON object with the following format:
        {{"reasoning": "Your reasoning here", "action": "The next action (click)", "parameters": {{"param1": "value1", ...}}}}"""
        self.grounder_prompt = """
        <image>
        Based on user's intent and the description of the target UI element, locate the element in the screenshot.
        User's intent: {reasoning}
        Target element's description: {description}
        Report the bbox coordinates in JSON format.
        """

    def generate_click_coordinate(self, instruction: str, image: Image.Image):
        # Implement MobiMind specific logic to generate click coordinates
        decider_msg = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + convert_pil_image_to_base64(image)},
                    },
                    {"type": "text", "text": self.decider_prompt.format(task=instruction)},
                ],
            }
        ]
        decider_input = self.processor.apply_chat_template(decider_msg, tokenize=False, add_generation_prompt=True)
        decider_inputs = self.processor(text=[decider_input], images=[image], padding=True, return_tensors="pt").to(
            self.decider.device
        )
        generated_ids = self.decider.generate(**decider_inputs, max_new_tokens=256, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(decider_inputs.input_ids, generated_ids)
        ]
        decider_response_str = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        try:
            decider_response = robust_json_loads(decider_response_str)
            reasoning = decider_response["reasoning"]
            target_element = decider_response["parameters"]["target_element"]
        except Exception:
            print("decider_response:", decider_response_str)

        try:
            grounder_msg = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + convert_pil_image_to_base64(image)},
                        },
                        {
                            "type": "text",
                            "text": self.grounder_prompt.format(reasoning=reasoning, description=target_element),
                        },
                    ],
                }
            ]
            grounder_input = self.processor.apply_chat_template(
                grounder_msg, tokenize=False, add_generation_prompt=True
            )
            grounder_inputs = self.processor(
                text=[grounder_input], images=[image], padding=True, return_tensors="pt"
            ).to(self.grounder.device)
            generated_ids = self.grounder.generate(**grounder_inputs, max_new_tokens=128, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(grounder_inputs.input_ids, generated_ids)
            ]
            grounder_response_str = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]
            coordinates = self._parse_output(grounder_response_str)
        except Exception:
            coordinates = None
            grounder_response_str = None
        return coordinates, {"decider_response": decider_response_str, "grounder_response": grounder_response_str}

    def _parse_output(self, response: str) -> Optional[Tuple[float, float]]:
        try:
            grounder_response = parse_json_response(response)
            bbox = None
            for key in grounder_response:
                if key.lower() in ["bbox", "bbox_2d", "bbox-2d", "bbox_2D", "bbox2d", "bbox_2009"]:
                    bbox = grounder_response[key]
                    break
            coordinates = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]  # type: ignore
            coordinates = [coordinates[0] / 999, coordinates[1] / 999]
            return tuple(coordinates)
        except Exception:
            return None
