from openai import OpenAI
import os
import base64


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "out/drive_test/00101817.png"

# Getting the Base64 string
base64_image = encode_image(image_path)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Analyze the given image and list only the traffic rules that are explicitly relevant to what is visible in the scene. Do not infer or assume the presence of elements that are not clearly depicted. Only mention rules that the autonomous vehicle must follow based on the visible road signs, signals, and markings."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)


