import openai

client = openai.Client(
    api_key="cannot be empty",
    base_url=f"http://10.171.248.164:9997/v1"
)
import os
import base64

#  读取本地文件，并编码为 BASE64 格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("/Users/daihongtao/Downloads/test.jpg")


response = client.chat.completions.create(
    model="qwen-vl-chat", # qwen-vl-chat 或者 yi-vl-chat
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这个图片是什么？"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    # "image_url": {
                    #     "url": "https://p3.dcarimg.com/img/motor-mis-img/2465f0c78280efddcc305991bd1f2ea2~2508x0.jpg",
                    # },
                },
            ],
        }
    ],
)
print(response.choices[0])
