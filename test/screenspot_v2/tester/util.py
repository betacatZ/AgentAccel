import base64
from io import BytesIO
import json
import logging
import re


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


