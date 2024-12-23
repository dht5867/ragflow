import requests

url = "http://10.171.248.164:7861/knowledge_base/upload_temp_docs"

payload={}
files=[
  ('files',('dead_lock.txt',open('/Users/daihongtao/Downloads/吉林银行/dead_lock.txt','rb'),'text/plain'))
]
headers = {
  'Content-Type': 'multipart/form-data'
}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)
