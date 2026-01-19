from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from tester.util import convert_pil_image_to_base64
from PIL import Image

model_path = "Qwen/Qwen3-VL"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).eval()
img_path = "test/screenspot_v2/visualization/sample_image.png"
processor = AutoProcessor.from_pretrained(model_path)
image = Image.open(img_path).convert("RGB")
instruction = "Describe the image."
messages = [
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
# text_input = text_input + self.guide_text

inputs = processor(text=[text_input], images=[image], padding=True, return_tensors="pt").to(model.device)
output = model(**inputs, output_attentions=True)
print(len(output.attentions))
