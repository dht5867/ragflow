
from api.apiutils import ApiRequest

api = ApiRequest()

def cmdb_chat_stream(input_txt: str):
    try:
        return api.cmdb_chat_stream(input_txt)
    except Exception as e:
        print(f"Error during cmdb_chat_stream: {e}")
        return None

def cmdb_chat_chinese(input_txt: str):
    try:
        return api.cmdb_chat_chinese(input_txt)
    except Exception as e:
        print(f"Error during cmdb_chat_chinese: {e}")
        return None
