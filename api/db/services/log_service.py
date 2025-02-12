import logging
from api.apiutils import ApiRequest
from typing import *

from pathlib import Path


api = ApiRequest()

def upload_temp_docs(files: Any):
    try:
        logging.info("----upload_temp_docs")
        return api.upload_temp_docs(files)
    except Exception as e:
        logging.error(f"Error during upload_temp_docs: {e}")
        return None
 
def log_file_chat(
        query: str,
        knowledge_id: str,
        history: List[Dict],
        stream: bool,
        model: str,
        max_tokens: int = None,
        prompt_name: str = "default",
        lang: str="zh"
    ):
    try:
        logging.info("----query")
        return api.file_chat(query,knowledge_id,history,stream,model,max_tokens,prompt_name,lang)
    except Exception as e:
        logging.error(f"file_chat: {e}")
        return None

