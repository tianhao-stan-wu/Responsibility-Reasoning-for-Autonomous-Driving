from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

image = Image.open("../out/crafted_examples/street.png")

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True)
model.to("cuda:0")



prompt = 'Given the picture, identify what the vehicle should do next. Do not output guesses or hypothetical scenarios. Your answers should be solid, concise, and only based on information available from the picture. The answer should be a single sentence that describes important aspects for safe driving.'

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



