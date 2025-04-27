#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
from datetime import date
from enum import IntEnum, Enum
import json
import rag.utils.es_conn
import rag.utils.infinity_conn

import rag.utils
from rag.nlp import search
from graphrag import search as kg_search
from api.utils import get_base_config, decrypt_database_config
from api.constants import RAG_FLOW_SERVICE_NAME
from api.utils.file_utils import get_project_base_directory

LIGHTEN = int(os.environ.get('LIGHTEN', "0"))

LLM = None
LLM_FACTORY = None
LLM_BASE_URL = None
CHAT_MDL = ""
EMBEDDING_MDL = ""
RERANK_MDL = ""
ASR_MDL = ""
IMAGE2TEXT_MDL = ""
API_KEY = None
PARSERS = None
HOST_IP = None
HOST_PORT = None
SECRET_KEY = None
FACTORY_LLM_INFOS = None

DATABASE_TYPE = os.getenv("DB_TYPE", 'mysql')
DATABASE = decrypt_database_config(name=DATABASE_TYPE)

# authentication
AUTHENTICATION_CONF = None

# client
CLIENT_AUTHENTICATION = None
HTTP_APP_KEY = None
GITHUB_OAUTH = None
FEISHU_OAUTH = None

DOC_ENGINE = None
docStoreConn = None

retrievaler = None
kg_retrievaler = None

# user registration switch
REGISTER_ENABLED = 1


def init_settings():
    global LLM, LLM_FACTORY, LLM_BASE_URL, LIGHTEN, DATABASE_TYPE, DATABASE, FACTORY_LLM_INFOS, REGISTER_ENABLED
    LIGHTEN = int(os.environ.get('LIGHTEN', "0"))
    DATABASE_TYPE = os.getenv("DB_TYPE", 'mysql')
    DATABASE = decrypt_database_config(name=DATABASE_TYPE)
    LLM = get_base_config("user_default_llm", {})
    LLM_DEFAULT_MODELS = LLM.get("default_models", {})
    LLM_FACTORY = LLM.get("factory", "Tongyi-Qianwen")
    LLM_BASE_URL = LLM.get("base_url")
    try:
        REGISTER_ENABLED = int(os.environ.get("REGISTER_ENABLED", "1"))
    except Exception:
        pass  
    
    try:
        with open(os.path.join(get_project_base_directory(), "conf", "llm_factories.json"), "r") as f:
            FACTORY_LLM_INFOS = json.load(f)["factory_llm_infos"]
    except Exception:
        FACTORY_LLM_INFOS = []

    global CHAT_MDL, EMBEDDING_MDL, RERANK_MDL, ASR_MDL, IMAGE2TEXT_MDL
    if not LIGHTEN:
        EMBEDDING_MDL = "BAAI/bge-large-zh-v1.5@BAAI"

    if LLM_DEFAULT_MODELS:
        CHAT_MDL = LLM_DEFAULT_MODELS.get("chat_model", CHAT_MDL)
        EMBEDDING_MDL = LLM_DEFAULT_MODELS.get("embedding_model", EMBEDDING_MDL)
        RERANK_MDL = LLM_DEFAULT_MODELS.get("rerank_model", RERANK_MDL)
        ASR_MDL = LLM_DEFAULT_MODELS.get("asr_model", ASR_MDL)
        IMAGE2TEXT_MDL = LLM_DEFAULT_MODELS.get("image2text_model", IMAGE2TEXT_MDL)

        # factory can be specified in the config name with "@". LLM_FACTORY will be used if not specified
        CHAT_MDL = CHAT_MDL + (f"@{LLM_FACTORY}" if "@" not in CHAT_MDL and CHAT_MDL != "" else "")
        EMBEDDING_MDL = EMBEDDING_MDL + (f"@{LLM_FACTORY}" if "@" not in EMBEDDING_MDL and EMBEDDING_MDL != "" else "")
        RERANK_MDL = RERANK_MDL + (f"@{LLM_FACTORY}" if "@" not in RERANK_MDL and RERANK_MDL != "" else "")
        ASR_MDL = ASR_MDL + (f"@{LLM_FACTORY}" if "@" not in ASR_MDL and ASR_MDL != "" else "")
        IMAGE2TEXT_MDL = IMAGE2TEXT_MDL + (
            f"@{LLM_FACTORY}" if "@" not in IMAGE2TEXT_MDL and IMAGE2TEXT_MDL != "" else "")

    global API_KEY, PARSERS, HOST_IP, HOST_PORT, SECRET_KEY
    API_KEY = LLM.get("api_key", "")
    PARSERS = LLM.get(
        "parsers",
        "naive:General,qa:Q&A,resume:Resume,manual:Manual,table:Table,paper:Paper,book:Book,laws:Laws,presentation:Presentation,picture:Picture,one:One,audio:Audio,email:Email,tag:Tag")

    HOST_IP = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("host", "127.0.0.1")
    HOST_PORT = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("http_port")

    SECRET_KEY = get_base_config(
        RAG_FLOW_SERVICE_NAME,
        {}).get("secret_key", str(date.today()))

    global AUTHENTICATION_CONF, CLIENT_AUTHENTICATION, HTTP_APP_KEY, GITHUB_OAUTH, FEISHU_OAUTH
    # authentication
    AUTHENTICATION_CONF = get_base_config("authentication", {})

    # client
    CLIENT_AUTHENTICATION = AUTHENTICATION_CONF.get(
        "client", {}).get(
        "switch", False)
    HTTP_APP_KEY = AUTHENTICATION_CONF.get("client", {}).get("http_app_key")
    GITHUB_OAUTH = get_base_config("oauth", {}).get("github")
    FEISHU_OAUTH = get_base_config("oauth", {}).get("feishu")

    global DOC_ENGINE, docStoreConn, retrievaler, kg_retrievaler
    DOC_ENGINE = os.environ.get('DOC_ENGINE', "elasticsearch")
    lower_case_doc_engine = DOC_ENGINE.lower()
    if lower_case_doc_engine == "elasticsearch":
        docStoreConn = rag.utils.es_conn.ESConnection()
    elif lower_case_doc_engine == "infinity":
        docStoreConn = rag.utils.infinity_conn.InfinityConnection()
    else:
        raise Exception(f"Not supported doc engine: {DOC_ENGINE}")

    retrievaler = search.Dealer(docStoreConn)
    kg_retrievaler = kg_search.KGSearch(docStoreConn)


class CustomEnum(Enum):
    @classmethod
    def valid(cls, value):
        try:
            cls(value)
            return True
        except BaseException:
            return False

    @classmethod
    def values(cls):
        return [member.value for member in cls.__members__.values()]

    @classmethod
    def names(cls):
        return [member.name for member in cls.__members__.values()]


class RetCode(IntEnum, CustomEnum):
    SUCCESS = 0
    NOT_EFFECTIVE = 10
    EXCEPTION_ERROR = 100
    ARGUMENT_ERROR = 101
    DATA_ERROR = 102
    OPERATING_ERROR = 103
    CONNECTION_ERROR = 105
    RUNNING = 106
    PERMISSION_ERROR = 108
    AUTHENTICATION_ERROR = 109
    UNAUTHORIZED = 401
    SERVER_ERROR = 500
    FORBIDDEN = 403
    NOT_FOUND = 404

import sys
DEFAULT_BIND_HOST = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"
# httpx 请求默认超时时间（秒）。如果加载模型或对话较慢，出现超时错误，可以适当加大该值。
HTTPX_DEFAULT_TIMEOUT = 30000.0

# 是否显示详细日志
log_verbose = False

# 缓存临时向量库数量（针对FAISS），用于文件对话
CACHED_MEMO_VS_NUM = 10

# 知识库中单段文本长度(不适用MarkdownHeaderTextSplitter)
CHUNK_SIZE = 250

# 知识库中相邻文本重合长度(不适用MarkdownHeaderTextSplitter)
OVERLAP_SIZE = 50

# 是否开启中文标题加强，以及标题增强的相关配置
# 通过增加标题判断，判断哪些文本为标题，并在metadata中进行标记；
# 然后将文本与往上一级的标题进行拼合，实现文本信息的增强。
ZH_TITLE_ENHANCE = False


API_SERVER = {
    "host": "10.171.248.164",
    "port": 7861,
}

# cmdb server
CMDB_SERVER = {
    "host": "10.171.248.164",
    "port": 8101,
}
# cmdb chat server

CMDB_CHAT_SERVER = {
    "host": "10.171.248.164",
    "port": 8000,
}

SWITCH_CHAT_SERVER = {
    "host": "10.171.248.164",
    "port": 8000,
}

def api_address() -> str:

    host = API_SERVER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = API_SERVER["port"]
    return f"http://{host}:{port}"

def cmdb_api_address() -> str:
    host = CMDB_SERVER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = CMDB_SERVER["port"]
    return f"http://{host}:{port}"


def cmdb_chat_address() -> str:
    host = CMDB_CHAT_SERVER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = CMDB_CHAT_SERVER["port"]
    return f"http://{host}:{port}"


def switch_chat_address() -> str:
    host = SWITCH_CHAT_SERVER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = SWITCH_CHAT_SERVER["port"]
    return f"http://{host}:{port}"
