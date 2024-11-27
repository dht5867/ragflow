from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# 配置转发的目标 URL
TARGET_URL = "http://127.0.0.1:7861/chat/upload_temp_docs"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    uploaded_file = request.files['files']
    
    if uploaded_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 转发文件到目标服务器
    files = {
        'files': (uploaded_file.filename, uploaded_file.stream, uploaded_file.content_type)
    }
    
    try:
        response = requests.post(TARGET_URL,files=files)
        return jsonify({"status": "success", "response": response.text}), response.status_code
    except requests.RequestException as e:
        return jsonify({"error": "Failed to forward the file", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
