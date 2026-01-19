import argparse
import json
import os
import torch
import matplotlib.pyplot as plt
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def plot_attention_map(attention_matrix, save_path, title="Attention Map"):
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_matrix, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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
                end_idx = input_ids_list.index(vision_start_id, start_idx)
                text_range = (start_idx, end_idx + 1)
                start_vision_idx = end_idx
                end_vision_idx = input_ids_list.index(vision_end_id, start_vision_idx)
                vision_range = (start_vision_idx, end_vision_idx + 1)
                return text_range, vision_range
            except ValueError:
                continue
    return text_range, vision_range


def main():
    parser = argparse.ArgumentParser(description="Visualize attention for Qwen3-VL")
    parser.add_argument("--json_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument(
        "--model_path", type=str, default="/data8/zhangdeming/models/Qwen/Qwen3-VL-8B-Instruct", help="Path to model"
    )
    parser.add_argument("--output_dir", type=str, default="visualization_output", help="Output directory")
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

    print(f"Loading data from {args.json_path}")
    with open(args.json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

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
                    {"type": "text", "text": instruction},
                    {
                        "type": "image",
                        "image": img_path,
                    },
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

        # We extract attention from the last layer of the first generated token step
        if output.attentions:
            first_token_attentions = output.attentions[0]
            last_layer_attention = first_token_attentions[-1]  # [batch, heads, q_len, k_len]

            # Average over heads
            avg_attention = last_layer_attention.mean(dim=1).squeeze(0).cpu().numpy()  # [q_len, k_len]
            text_range, vision_range = find_range(inputs["input_ids"], processor.tokenizer)

            if text_range is None or vision_range is None:
                print(f"Skipping index {idx}: Could not locate text or vision range.")
                continue

            avg_attention = avg_attention[text_range[0] : text_range[1], vision_range[0] : vision_range[1]]

            image_name = os.path.splitext(os.path.basename(img_path))[0]
            save_name = f"{idx:03d}_{image_name}_attn.png"
            save_path = os.path.join(args.output_dir, save_name)

            plot_attention_map(avg_attention, save_path, title=f"Avg Attention Layer -1\n{instruction[:30]}...")
            print(f"Saved attention map to {save_path}")


if __name__ == "__main__":
    main()
