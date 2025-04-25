
import logging
from api.apiutils import ApiRequest

api = ApiRequest()

def cmdb_chat_stream(input_txt: str):
    try:
        logging.info("----cmdb_chat_stream")
        return api.cmdb_chat_stream(input_txt)
    except Exception as e:
        logging.error(f"Error during cmdb_chat_stream: {e}")
        return None

def cmdb_chat_chinese(input_txt: str):
    try:
        return api.cmdb_chat_chinese(input_txt)
    except Exception as e:
        print(f"Error during cmdb_chat_chinese: {e}")
        return None


def swtich_chat_stream(input_txt: str):
    try:
        logging.info("----swtich_chat_stream")
        return api.switch_chat_stream(input_txt)
    except Exception as e:
        logging.error(f"Error during swtich_chat_stream: {e}")
        return None
