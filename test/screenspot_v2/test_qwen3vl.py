import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.generation.configuration_utils import GenerationConfig
import json
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
from util import convert_pil_image_to_base64

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
args = parser.parse_args()


# Load model and processor
model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).eval()
processor = AutoProcessor.from_pretrained(model_path)

generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True).to_dict()
generation_config.update(max_length=2048, do_sample=False, temperature=0.0)
model.generation_config = GenerationConfig(**generation_config)
print("Load Success")

# Determine tasks
tasks = ["mobile", "desktop", "web"] if args.task == "all" else [args.task]

tasks_result = []
result = []

for task in tasks:
    dataset = f"screenspot_{task}_v2.json"
    screenspot_data = json.load(open(os.path.join(args.dataset_path, dataset), "r"))
    print("Num of sample: " + str(len(screenspot_data)))

    system_prompt = (
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
        '{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen\'s resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}\n</tools>\n\n'
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n'
    )

    num_action = 0
    corr_action = 0
    text_correct = []
    icon_correct = []
    num_wrong_format = 0

    for j, item in tqdm(enumerate(screenspot_data)):
        num_action += 1
        filename = item["img_filename"]
        img_path = os.path.join(args.dataset_path, "screenspotv2_image", filename)
        if not os.path.exists(img_path):
            print("img not found")
            input()
        image = Image.open(img_path)
        instruction = item["instruction"]
        bbox = item["bbox"]
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        img_size = image.size
        bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        guide_text = '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": ['
        text_input = text_input + guide_text

        inputs = processor(text=[text_input], images=[image], padding=True, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        response = guide_text + response
        cut_index = response.rfind("}")
        if cut_index != -1:
            response = response[: cut_index + 1]

        try:
            action = json.loads(response.split("<tool_call>\n")[1])
            coordinates = action["arguments"]["coordinate"]
            if len(coordinates) == 2:
                point_x, point_y = coordinates
                click_point = [point_x / 1000, point_y / 1000]

            if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                corr_action += 1
                if item["data_type"] == "text":
                    text_correct.append(1)
                else:
                    icon_correct.append(1)
                logging.info("match " + str(corr_action / num_action))
            else:
                if item["data_type"] == "text":
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                logging.info("unmatch " + str(corr_action / num_action))

            result.append(
                {
                    "img_path": img_path,
                    "text": instruction,
                    "bbox": bbox,
                    "pred": click_point,
                    "type": item["data_type"],
                    "source": item["data_source"],
                }
            )
        except Exception:
            num_wrong_format += 1
            if item["data_type"] == "text":
                text_correct.append(0)
            else:
                icon_correct.append(0)
            logging.info("Step: " + str(j) + " wrong format")

    logging.info("Action Acc: " + str(corr_action / num_action))
    logging.info("Total num: " + str(num_action))
    logging.info("Wrong format num: " + str(num_wrong_format))
    logging.info("Text Acc: " + str(sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0))
    logging.info("Icon Acc: " + str(sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0))

    text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
    icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
    tasks_result.append([text_acc, icon_acc])

logging.info(tasks_result)
