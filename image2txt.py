import requests
import json
import base64   

def vlm():
    with open("/Users/hong.tao.dai/Downloads/project/ragflow/tmp/095f8df466a311f0839b8315a63d5a7e.png","rb") as f: 
    # b64encode是编码，b64decode是解码 
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")  # 转为字符串

    url = "http://127.0.0.1:11434/api/generate"
    data = {
        "model": "qwen2.5vl:latest",
        "prompt": '理解图片',
        "stream": True,  # 关键：明确要求服务端流式输出（如果API支持）
        "images": [image_base64]
    }

    response = requests.post(url, json=data, stream=True)

    for line in response.iter_lines():
        if line:
            # 解析 JSON 数据
            json_data = json.loads(line.decode('utf-8'))
                    
            # 提取 response 字段内容
            current_response = json_data.get("response", "")
                    
            # 实时输出新增内容（非覆盖模式）
            if current_response:
                print(current_response, end='', flush=True)  # 逐词输出
                        

vlm()