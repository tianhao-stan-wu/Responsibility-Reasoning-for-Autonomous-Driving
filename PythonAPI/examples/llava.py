from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True)
model.to("cuda:0")

image = Image.open("out/004150.png")

prompt = """
When given instructions to finish some tasks, humans tend to reason in a hierarchical manner. Please decompose the natural language task description into a hierarchical structure based on logical relationships. 
The task description is "How to ensure driving safety in this scenario?"

Output Format:
root task
(1.1) summary of subtasks [task a, task c]
    1.1.1. [task a]
    1.1.2. [task c]
(1.2) summary of subtasks [task b]
    1.2.1. [task b]


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