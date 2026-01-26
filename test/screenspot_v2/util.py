import base64
from io import BytesIO
import json
import logging
import re
from PIL import Image
from typing import Optional, Tuple


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# 预处理 decider_response_str，增强健壮性
def robust_json_loads(s):
    """
    健壮的 JSON 加载函数
    支持 guided decoding 和普通模式的混合输出
    """
    if not isinstance(s, str):
        s = str(s)

    s = s.strip()

    # 首先尝试直接解析 JSON（guided decoding 纯输出）
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 提取 ```json ... ``` 代码块
    codeblock = re.search(r"```json\s*([\s\S]*?)\s*```", s, re.MULTILINE)
    if codeblock:
        s = codeblock.group(1).strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

    # 替换中文省略号为英文 ...
    s = s.replace("…", "...")

    # 尝试再次解析
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 对象（从第一个 { 到最后一个 }）
    start_idx = s.find("{")
    if start_idx != -1:
        brace_count = 0
        for i in range(start_idx, len(s)):
            if s[i] == "{":
                brace_count += 1
            elif s[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_str = s[start_idx : i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

    # 解析失败，记录错误
    logging.error("解析 decider_response 失败")
    logging.error(f"原始内容: {s[:300]}...")
    raise ValueError("无法解析 JSON 响应: 响应格式不正确")


def align_size_to_patch(image: Image.Image, patch_size: int = 16) -> Tuple[int, int]:
    """
    将图像调整到patch_size的倍数大小，获得新的图像大小
    """
    width, height = image.size
    new_width = round(width / patch_size) * patch_size
    new_height = round(height / patch_size) * patch_size
    if new_width == width and new_height == height:
        return width, height
    return new_width, new_height


def find_range(input_ids, tokenizer):
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    input_ids_list = input_ids[0].tolist()
    start_indices = [i for i, x in enumerate(input_ids_list) if x == im_start_id]
    text_range, vision_range = None, None
    for start_idx in start_indices:
        chk_end = min(start_idx + 10, len(input_ids_list))
        role_tokens = input_ids_list[start_idx + 1 : chk_end]
        role_str = tokenizer.decode(role_tokens).strip().lower()
        if role_str.startswith("user"):
            try:
                start_vision_idx = input_ids_list.index(vision_start_id, start_idx)
                vision_end_idx = input_ids_list.index(vision_end_id, start_vision_idx)
                vision_range = (start_vision_idx + 1, vision_end_idx)
                end_idx = input_ids_list.index(im_end_id, vision_end_idx)
                text_range = (vision_end_idx + 1, end_idx)
                return text_range, vision_range
            except ValueError:
                continue
    return text_range, vision_range
