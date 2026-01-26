import argparse
import json
import os
import torch
import matplotlib.pyplot as plt
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import align_size_to_patch, find_range


def plot_attention_map(
    tokenizer, instruction_idx, attention_matrix, image, patch_size, save_path, title="Attention Map"
):
    # 遍历每个文本token
    for i, token_id in enumerate(instruction_idx):
        token_str = tokenizer.decode([token_id]).strip()
        if not token_str or token_str.isspace():
            continue

        # 获取当前token的attention权重
        attn = attention_matrix[i]  # Shape: (num_patches,)
        # 计算patch网格大小
        num_patches_w = image.width // patch_size
        num_patches_h = image.height // patch_size
        # 将attention reshape为图像patch大小
        attn_reshaped = attn.reshape(num_patches_h, num_patches_w)

        # 创建画布
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制原图
        ax.imshow(image)

        # 绘制热力图，使用bilinear插值和透明度叠加
        heatmap = ax.imshow(
            attn_reshaped,
            cmap="jet",
            alpha=0.5,
            interpolation="bilinear",
            extent=(0.0, float(image.width), float(image.height), 0.0),  # 匹配图像坐标
        )

        # 添加colorbar
        cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention Weight")

        # 设置标题
        ax.set_title(f"{title}\nToken: '{token_str}'")

        # 关闭坐标轴
        ax.axis("off")

        # 调整布局并保存
        plt.tight_layout()

        # 生成保存路径
        os.makedirs(save_path, exist_ok=True)
        token_save_path = os.path.join(save_path, f"{i}_{token_str}.png")
        plt.savefig(token_save_path, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"  保存注意力图: {token_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize attention for Qwen3-VL")
    parser.add_argument("--json_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument(
        "--model_path", type=str, default="/data8/zhangdeming/models/Qwen/Qwen3-VL-8B-Instruct", help="Path to model"
    )
    parser.add_argument("--output_dir", type=str, default="./output/attn_output", help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="cuda",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    patch_size = model.visual.patch_size * model.visual.spatial_merge_size
    print(f"Loading data from {args.json_path}")
    with open(args.json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
        if type(data_list) is not list:
            data_list = [data_list]

    for idx, item in enumerate(data_list):
        img_path = item.get("img_path")
        instruction = item.get("text") or item.get("instruction")  # Handle potentially different key

        if not img_path or not os.path.exists(img_path):
            print(f"Skipping index {idx}: Image not found or missing path: {img_path}")
            continue

        if not instruction:
            print(f"Skipping index {idx}: Missing instruction")
            continue

        print(f"Processing index {idx}: {img_path}")
        system_prompt = """
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
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": instruction},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=16,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # output.attentions is a tuple of tuples.
        # Outer tuple: generated tokens
        # Inner tuple: layers
        # Shape of each attention tensor: (batch_size, num_heads, seq_len, seq_len)

        # We extract attention from all layers of the first generated token step
        if getattr(output, "attentions"):
            token_attentions = output.attentions[0]  # type: ignore
            total_layers = len(token_attentions)

            # 遍历所有层
            for layer_idx in range(total_layers):
                # 获取当前层的注意力
                layer_attention = token_attentions[layer_idx]  # [batch, heads, q_len, k_len]

                # Average over heads
                avg_attention = layer_attention.mean(dim=1).squeeze(0).float().cpu().numpy()  # [q_len, k_len]
                text_range, vision_range = find_range(inputs["input_ids"], processor.tokenizer)

                if text_range is None or vision_range is None:
                    print(f"Skipping index {idx}, layer {layer_idx}: Could not locate text or vision range.")
                    continue

                avg_attention = avg_attention[text_range[0] : text_range[1], vision_range[0] : vision_range[1]]

                image_name = os.path.splitext(os.path.basename(img_path))[0]
                # 在保存路径中包含层信息
                save_path = os.path.join(args.output_dir, f"{image_name}", f"layer_{layer_idx}")

                image = Image.open(img_path).convert("RGB")
                width, height = align_size_to_patch(image, patch_size)
                image = image.resize((width, height))
                instruction_idx = inputs["input_ids"][0].tolist()[text_range[0] : text_range[1]]
                temp_text = instruction if len(instruction) <= 30 else instruction[:30] + "..."
                plot_attention_map(
                    processor.tokenizer,
                    instruction_idx,
                    avg_attention,
                    image,
                    patch_size,
                    save_path,
                    title=f"Avg Attention Layer {layer_idx}/{total_layers - 1}\n{temp_text}",
                )
                print(f"Saved attention map for layer {layer_idx} to {save_path}")


if __name__ == "__main__":
    main()
