from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

image = Image.open("./out/continuous_turn/00034879.png")

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True)
model.to("cuda:0")



prompt = """
You are given an autonomous driving scene from the front view of the ego vehicle. You will be given a sequence of tasks. Answer each task based on the prompt. 


Task1:  Based on the scene, what are legal traffic rules to follow in this situation? You should only give answers that you are confident of. Do not output guesses or indeterminate instructions. Your answers should be solid, concrete, and concise such that people without expert knowledge in traffic rules can easily understand. If a rule pertains to other traffic participants, only identify such rules if traffic participants appear in the scene. Your answer should follow the format below without any extra words.



1. description of rule 1

2. description of rule 2

3. …



Task2:

We define three classes of safety specification in autonomous driving. Class 1 describes the safety between ego vehicle and other traffic participants, denoted S1(ego, participant_name). Class 2 describes the safety of the ego vehicle with respect to the lanes, denoted S2(ego, lane). Class 3 descibes the safety of the ego vehicle itself, denoted S3(ego). Classify your previous answers into these three classes. Each class can contain multiple specifications or none. Your answer should follow the format below without any extra words.



S1(ego, participant_name): do/do not (a1) until (b1)

…

S2(ego, lane): do/do not (a2) until (b2)

…

S3(ego): do/do not (a3) until (b3)

…



Task3:

Next, we need to formulate the safety specifications (a1, a2, …) in natural language into mathematical expression describing the state of the vehicle. The state is formulated as [x, y, theta, v], where x and y are locations, theta the heading, and v the speed of the vehicle. We decide to use Constraints, with control barrier functions, and Objectives, with control lyapunov functions to formulate.



Constraints:

C1(x_o, y_o): ensures the ego vehicle does not collide with object at location (x_o, y_o) 

C2(theta_min, theta_max): ensures the ego vehicle’s heading is restricted to the range (theta_min, theta_max)

C3(v_min, v_max): ensures the ego vehicle’s speed is restricted to the range (v_min, v_max)



Objectives:

O1(x_g, y_g): ensures the ego vehicle stabilizes to location (x_o, y_o) 

O2(theta_d): ensures the ego vehicle’s heading converges to theta_d

O3(speed_d): ensures the ego vehicle’s speed converges to speed_d



For each safety specification (a1, a2, …) above, find the most proper translation in C1,C2,C3,O1,O2,O3 and fill in the parameter based on the given information. Your answer should follow the format below without any extra words.



S1(ego, participant_name): C1(x_o, y_o) until (b1)

…

S2(ego, lane): C2(theta_min, theta_max) until (b2)

…

S3(ego): O3(speed_v) until (b3)

…

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



