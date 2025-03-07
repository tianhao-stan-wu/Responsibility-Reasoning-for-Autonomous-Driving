from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True)
model.to("cuda:0")

image = Image.open("../out/continuous_turn/00034879.png")

prompt = """
You are given a picture depicting the front view of an autonomous vehicle. Analyze the given image and list only the traffic rules that are explicitly relevant to what is visible in the scene. Do not infer or assume the presence of elements that are not clearly depicted. Only mention rules that the autonomous vehicle must follow based on the visible road signs, signals, and markings.

        """

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=1000)

print(processor.decode(output[0], skip_special_tokens=True))



