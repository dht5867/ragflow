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
import base64
from io import BytesIO
import os
import binascii
from datetime import datetime
import logging
import binascii
import json
import json_repair
import re
import time
from copy import deepcopy
from functools import partial
from timeit import default_timer as timer
from collections import defaultdict

from langfuse import Langfuse

from agentic_reasoning import DeepResearcher
from api import settings
from typing import Optional, Union
from api.db import LLMType, ParserType, StatusEnum
from api.db.db_models import Conversation, DB, Dialog
from api.db.services.common_service import CommonService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.langfuse_service import TenantLangfuseService
from api.db.services.llm_service import LLMBundle, TenantLLMService
from api.utils import current_timestamp, datetime_format
from api.db.services.llm_service import LLMService, TenantLLMService, LLMBundle
from api import settings
from api.db.services.txt2image_service import txt_image
from api.db.services.user_service import UserTenantService
from api.settings import  retrievaler, kg_retrievaler
from graphrag.utils import get_tags_from_cache, set_tags_to_cache
from rag.app.resume import forbidden_select_fields4resume
from rag.app.tag import label_question
from rag.nlp.search import index_name
from rag.prompts import chunks_format, citation_prompt, full_question, kb_prompt, keyword_extraction, llm_id2llm_type, message_fit_in
from rag.utils import num_tokens_from_string, rmSpace, encoder
from rag.utils.tavily_conn import Tavily
from api.db.services.log_service import log_file_chat
from api.db.services.file2document_service import File2DocumentService
from rag.utils.storage_factory import STORAGE_IMPL
from api.utils.file_utils import get_project_base_directory
from api.db.services.document_service import DocumentService

PROMPT_TEMPLATES = {
    "llm_chat": {
        "default":
            '当用户询问你是谁，或者在自我介绍时，请回答“我是小吉，你的IT运维助手。”\n'
            '以下是问题 【 {prompt} 】\n',

        "with_history":
            'The following is a friendly conversation between a human and an AI. '
            'The AI is talkative and provides lots of specific details from its context. '
            'If the AI does not know the answer to a question, it truthfully says it does not know.\n\n'
            'Current conversation:\n'
            '{history}\n'
            'Human: {prompt}\n'
            'AI:',

        "py":
            '你是一个聪明的代码助手，请你给我写出简单的py代码。 \n'
            '{{ input }}',
    },

    "file_base_chat": {
        "default":
            '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，'
            '不允许在答案中添加编造成分，答案请使用中文。 当用户询问你是谁，或者在自我介绍时，请回答“我是小吉，你的IT运维助手。”</指令>\n'
            '<已知信息>{{ knowledge }}</已知信息>\n'
            '<问题> 作为IT系统日志分析专家，你的任务是分析以下日志并撰写一份详细的报告，报告格式应包括以下内容：'
            '1. 概述：简要介绍分析的目的和方法。'
            '2. 日志分析：根据给定的日志内容，分析出潜在的问题、异常或趋势。'
            '3. 根本原因分析：对于发现的问题或异常，提出可能的根本原因，并给出相关的证据支持。特别是对于ORACLE日志，当出现ORA开头的错误码时，尽量找出相关的sql语句和session信息。'
            '4. 解决方案建议：针对每个问题或异常，提出相应的解决方案，并说明实施步骤和预期效果。'
            '5. 预防措施建议：为了避免类似问题再次发生，提出一些预防措施或最佳实践建议。'
            '并请根据以上提示，撰写一份完整的报告，确保包含上述提到的所有要素。{{ prompt }}</问题>\n',


        "log":
            '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，'
            '不允许在答案中添加编造成分，答案请使用中文。 当用户询问你是谁，或者在自我介绍时，请回答“我是小吉，你的IT运维助手。”</指令>\n'
            '<已知信息>{knowledge}</已知信息>\n'
            '<问题> 作为IT系统日志分析专家，你的任务是分析以下日志并撰写一份详细的报告，报告格式应包括以下内容：'
            '1. 概述：简要介绍分析的目的和方法。'
            '2. 日志分析：根据给定的日志内容，分析出潜在的问题、异常或趋势。'
            '3. 根本原因分析：对于发现的问题或异常，提出可能的根本原因，并给出相关的证据支持。特别是对于ORACLE日志，当出现ORA开头的错误码时，尽量找出相关的sql语句和session信息。'
            '4. 解决方案建议：针对每个问题或异常，提出相应的解决方案，并说明实施步骤和预期效果。'
            '5. 预防措施建议：为了避免类似问题再次发生，提出一些预防措施或最佳实践建议。'
            '并请根据以上提示，撰写一份完整的报告，确保包含上述提到的所有要素。{prompt}</问题>\n',


        "text":
            '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，答案请使用中文。 当用户询问你是谁，或者在自我介绍时，请回答“我是小吉，你的IT运维助手。”</指令>\n'
            '<已知信息>{{ context }}</已知信息>\n'
            '<问题>{{ question }}</问题>\n',

        "empty":  # 搜不到知识库的时候使用
            '请你回答我的问题:\n'
            '{{ question }}\n\n',
    },



    "knowledge_base_chat": {
        "default":
            '<指令>请根据已知信息，简洁和专业地回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，'
            '不允许在答案中添加编造成分，答案请使用中文。 当用户询问你是谁，或者在自我介绍时，请回答“我是小吉，你的IT运维助手。”</指令>\n'
            '<已知信息>{knowledge}</已知信息>\n'
            '<问题>{prompt}</问题>\n',


        "log":
            '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，'
            '不允许在答案中添加编造成分，答案请使用中文。 当用户询问你是谁，或者在自我介绍时，请回答“我是小吉，你的IT运维助手。”</指令>\n'
            '<已知信息>{knowledge}</已知信息>\n'
            '<问题> 作为IT系统日志分析专家，你的任务是分析以下日志并撰写一份详细的报告，报告格式应包括以下内容：'
            '1. 概述：简要介绍分析的目的和方法。'
            '2. 日志分析：根据给定的日志内容，分析出潜在的问题、异常或趋势。'
            '3. 根本原因分析：对于发现的问题或异常，提出可能的根本原因，并给出相关的证据支持。特别是对于ORACLE日志，当出现ORA开头的错误码时，尽量找出相关的sql语句和session信息。'
            '4. 解决方案建议：针对每个问题或异常，提出相应的解决方案，并说明实施步骤和预期效果。'
            '5. 预防措施建议：为了避免类似问题再次发生，提出一些预防措施或最佳实践建议。'
            '并请根据以上提示，撰写一份完整的报告，确保包含上述提到的所有要素。{prompt}</问题>\n',


        "text":
            '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，答案请使用中文。 当用户询问你是谁，或者在自我介绍时，请回答“我是小吉，你的IT运维助手。”</指令>\n'
            '<已知信息>{knowledge}</已知信息>\n'
            '<问题>{prompt}</问题>\n',

        "empty":  # 搜不到知识库的时候使用
            '请你回答我的问题:\n'
            '{prompt}\n\n',
    },


    "search_engine_chat": {
        "default":
            '<指令>这是我搜索到的互联网信息，请你根据这些信息进行提取并有调理，简洁的回答问题。'
            '如果无法从中得到答案，请说 “无法搜索到能回答问题的内容”。 </指令>\n'
            '<已知信息>{knowledge}</已知信息>\n'
            '<问题>{prompt}</问题>\n',

        "search":
            '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，答案请使用中文。 </指令>\n'
            '<已知信息>{knowledge}</已知信息>\n'
            '<问题>{prompt}</问题>\n',
    },


    "agent_chat": {
        "default":
            'Answer the following questions as best you can. If it is in order, you can use some tools appropriately. '
            'You have access to the following tools:\n\n'
            '{tools}\n\n'
            'Use the following format:\n'
            'Question: the input question you must answer1\n'
            'Thought: you should always think about what to do and what tools to use.\n'
            'Action: the action to take, should be one of [{tool_names}]\n'
            'Action Input: the input to the action\n'
            'Observation: the result of the action\n'
            '... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\n'
            'Thought: I now know the final answer\n'
            'Final Answer: the final answer to the original input question\n'
            'Begin!\n\n'
            'history: {history}\n\n'
            'Question: {input}\n\n'
            'Thought: {agent_scratchpad}\n',

        "ChatGLM3":
            'You can answer using the tools, or answer directly using your knowledge without using the tools. '
            'Respond to the human as helpfully and accurately as possible.\n'
            'You have access to the following tools:\n'
            '{tools}\n'
            'Use a json blob to specify a tool by providing an action key (tool name) '
            'and an action_input key (tool input).\n'
            'Valid "action" values: "Final Answer" or  [{tool_names}]'
            'Provide only ONE action per $JSON_BLOB, as shown:\n\n'
            '```\n'
            '{{{{\n'
            '  "action": $TOOL_NAME,\n'
            '  "action_input": $INPUT\n'
            '}}}}\n'
            '```\n\n'
            'Follow this format:\n\n'
            'Question: input question to answer\n'
            'Thought: consider previous and subsequent steps\n'
            'Action:\n'
            '```\n'
            '$JSON_BLOB\n'
            '```\n'
            'Observation: action result\n'
            '... (repeat Thought/Action/Observation N times)\n'
            'Thought: I know what to respond\n'
            'Action:\n'
            '```\n'
            '{{{{\n'
            '  "action": "Final Answer",\n'
            '  "action_input": "Final response to human"\n'
            '}}}}\n'
            'Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. '
            'Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.\n'
            'history: {history}\n\n'
            'Question: {input}\n\n'
            'Thought: {agent_scratchpad}',
    }
}
def get_prompt_template(type: str, name: str) -> Optional[str]:
    '''
    从prompt_config中加载模板内容
    type: "llm_chat","agent_chat","knowledge_base_chat","search_engine_chat"的其中一种，如果有新功能，应该进行加入。
    '''

    return PROMPT_TEMPLATES[type].get(name)


class DialogService(CommonService):
    model = Dialog

    @classmethod
    def save(cls, **kwargs):
        """Save a new record to database.

        This method creates a new record in the database with the provided field values,
        forcing an insert operation rather than an update.

        Args:
            **kwargs: Record field values as keyword arguments.

        Returns:
            Model instance: The created record object.
        """
        sample_obj = cls.model(**kwargs).save(force_insert=True)
        return sample_obj

    @classmethod
    def update_many_by_id(cls, data_list):
        """Update multiple records by their IDs.

        This method updates multiple records in the database, identified by their IDs.
        It automatically updates the update_time and update_date fields for each record.

        Args:
            data_list (list): List of dictionaries containing record data to update.
                             Each dictionary must include an 'id' field.
        """
        with DB.atomic():
            for data in data_list:
                data["update_time"] = current_timestamp()
                data["update_date"] = datetime_format(datetime.now())
                cls.model.update(data).where(cls.model.id == data["id"]).execute()

    @classmethod
    @DB.connection_context()
    def get_list(cls, tenant_id, page_number, items_per_page, orderby, desc, id, name):
        chats = cls.model.select()
        if id:
            chats = chats.where(cls.model.id == id)
        if name:
            chats = chats.where(cls.model.name == name)
        chats = chats.where((cls.model.tenant_id == tenant_id) & (cls.model.status == StatusEnum.VALID.value))
        if desc:
            chats = chats.order_by(cls.model.getter_by(orderby).desc())
        else:
            chats = chats.order_by(cls.model.getter_by(orderby).asc())

        chats = chats.paginate(page_number, items_per_page)

        return list(chats.dicts())


class ConversationService(CommonService):
    model = Conversation

    @classmethod
    @DB.connection_context()
    def get_list(cls,dialog_id,page_number, items_per_page, orderby, desc, id , name):
        sessions = cls.model.select().where(cls.model.dialog_id ==dialog_id)
        if id:
            sessions = sessions.where(cls.model.id == id)
        if name:
            sessions = sessions.where(cls.model.name == name)
        if desc:
            sessions = sessions.order_by(cls.model.getter_by(orderby).desc())
        else:
            sessions = sessions.order_by(cls.model.getter_by(orderby).asc())

        sessions = sessions.paginate(page_number, items_per_page)

        return list(sessions.dicts())
    
    @classmethod
    @DB.connection_context()
    def get_list_by_dialog_id(cls,dialog_id):
        fields = [
            cls.model.id,
            cls.model.name,
            cls.model.dialog_id
        ]
        sessions = cls.model.select(*fields).where(cls.model.dialog_id ==dialog_id)
        return sessions
    


def message_fit_in(msg, max_length=4000):
    def count():
        nonlocal msg
        tks_cnts = []
        for m in msg:
            tks_cnts.append(
                {"role": m["role"], "count": num_tokens_from_string(m["content"])})
        total = 0
        for m in tks_cnts:
            total += m["count"]
        return total

    c = count()
    if c < max_length:
        return c, msg

    msg_ = [m for m in msg[:-1] if m["role"] == "system"]
    if len(msg) > 1:
        msg_.append(msg[-1])
    msg = msg_
    c = count()
    if c < max_length:
        return c, msg

    ll = num_tokens_from_string(msg_[0]["content"])
    ll2 = num_tokens_from_string(msg_[-1]["content"])
    if ll / (ll + ll2) > 0.8:
        m = msg_[0]["content"]
        m = encoder.decode(encoder.encode(m)[:max_length - ll2])
        msg[0]["content"] = m
        return max_length, msg

    m = msg_[1]["content"]
    m = encoder.decode(encoder.encode(m)[:max_length - ll2])
    msg[1]["content"] = m
    return max_length, msg


def llm_id2llm_type(llm_id):
    #glm-4v@ZHIPU-AI
    llm_id, _ = TenantLLMService.split_model_name_and_factory(llm_id)
    fnm = os.path.join(get_project_base_directory(), "conf")
    llm_factories = json.load(open(os.path.join(fnm, "llm_factories.json"), "r"))
    for llm_factory in llm_factories["factory_llm_infos"]:
        for llm in llm_factory["llm"]:
            if llm_id == llm["llm_name"]:
                return llm["model_type"].strip(",")


def kb_prompt(kbinfos, max_tokens):
    knowledges = [ck["content_with_weight"] for ck in kbinfos["chunks"]]
    used_token_count = 0
    chunks_num = 0
    for i, c in enumerate(knowledges):
        used_token_count += num_tokens_from_string(c)
        chunks_num += 1
        if max_tokens * 0.97 < used_token_count:
            knowledges = knowledges[:i]
            break

    docs = DocumentService.get_by_ids([ck["doc_id"] for ck in kbinfos["chunks"][:chunks_num]])
    docs = {d.id: d.meta_fields for d in docs}

    doc2chunks = defaultdict(lambda: {"chunks": [], "meta": []})
    for ck in kbinfos["chunks"][:chunks_num]:
        doc2chunks[ck["docnm_kwd"]]["chunks"].append(ck["content_with_weight"])
        doc2chunks[ck["docnm_kwd"]]["meta"] = docs.get(ck["doc_id"], {})

    knowledges = []
    for nm, cks_meta in doc2chunks.items():
        txt = f"Document: {nm} \n"
        for k,v in cks_meta["meta"].items():
            txt += f"{k}: {v}\n"
        txt += "Relevant fragments as following:\n"
        for i, chunk in enumerate(cks_meta["chunks"], 1):
            txt += f"{i}. {chunk}\n"
        knowledges.append(txt)
    return knowledges

def log_chat(dialog, messages, stream=True, **kwargs):
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."
    #qwen2.5:14b@Ollama
    refs = []
    tmp = dialog.llm_id.split("@")
    lang=kwargs['language']
    
    fid = None
    llm_id = tmp[0]
    if len(tmp)>1: fid = tmp[1]
    #如果fid为False，则只根据llm_name查询LLMService；
    #如果fid不为False，则在查询时还会根据fid进行过滤。
    llm = LLMService.query(llm_name=llm_id) if not fid else LLMService.query(llm_name=llm_id, fid=fid)
    if not llm:
        llm = TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id) if not fid else \
            TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id, llm_factory=fid)
        if not llm:
            #
            raise LookupError("LLM(%s) not found" % dialog.llm_id)
        max_tokens = 8192
    else:
        max_tokens = llm[0].max_tokens
    questions = [m["content"] for m in messages if m["role"] == "user"][-3:]
    logging.info('questions------')
    logging.info(questions[-1])
    attachments = kwargs["doc_ids"].split(",") if "doc_ids" in kwargs else None
    if "doc_ids" in messages[-1]:
        attachments = messages[-1]["doc_ids"]
        for m in messages[:-1]:
            if "doc_ids" in m:
                attachments.extend(m["doc_ids"])
    logging.info('doc_id------')
    #取最后一个上传的日志
    doc_id = attachments[-1]
    logging.info(doc_id)
    # prompt_name=kwargs["prompt_name"]
     
    # 生成答案的装饰器
    def decorate_answer(answer):
        return {"answer": answer, "reference": refs}
    
    def check_error_msg(data: Union[str, dict, list], key: str = "errorMsg") -> str:
        '''
        return error message if error occured when requests API
        '''
        if isinstance(data, dict):
            if key in data:
                return data[key]
            if "code" in data and data["code"] != 200:
                return data["msg"]
        return ""
    # 根据stream选项处理模型的对话响应
    if stream:
        answer = ""
        logging.info('--- stream rest log_chat--')
        for d in log_file_chat(questions[-1], doc_id ,[],stream,llm_id,max_tokens, "default",lang):
            if error_msg := check_error_msg(d):  # check whether error occured
                        logging.error(error_msg)
                        answer=" 后台服务错误，请重试对话"
                        break
            elif chunk := d.get("answer"):
                       answer += chunk
            yield {"answer": answer, "reference": {}}
        yield decorate_answer(answer)
    else:
        answer = ""
        logging.info('---rest log_chat--')
        for d in log_file_chat(questions[-1], doc_id ,[],stream,llm_id,max_tokens, "default"):
            if error_msg := check_error_msg(d):  # check whether error occured
                        logging.error(error_msg)
                        answer=" 后台服务错误，请重试对话"
                        break
            elif chunk := d.get("answer"):
                       answer= chunk
            yield {"answer": answer, "reference": {}}
        yield decorate_answer(answer)
        
def label_question(question, kbs):
    tags = None
    tag_kb_ids = []
    for kb in kbs:
        if kb.parser_config.get("tag_kb_ids"):
            tag_kb_ids.extend(kb.parser_config["tag_kb_ids"])
    if tag_kb_ids:
        all_tags = get_tags_from_cache(tag_kb_ids)
        if not all_tags:
            all_tags = settings.retrievaler.all_tags_in_portion(kb.tenant_id, tag_kb_ids)
            set_tags_to_cache(all_tags, tag_kb_ids)
        else:
            all_tags = json.loads(all_tags)
        tag_kbs = KnowledgebaseService.get_by_ids(tag_kb_ids)
        tags = settings.retrievaler.tag_query(question,
                                              list(set([kb.tenant_id for kb in tag_kbs])),
                                              tag_kb_ids,
                                              all_tags,
                                              kb.parser_config.get("topn_tags", 3)
                                              )
    return tags
def get_storage_binary(bucket, name):
    return STORAGE_IMPL.get(bucket, name)

def image2base64(image):
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        if isinstance(image, BytesIO):
            return base64.b64encode(image.getvalue()).decode("utf-8")
        buffered = BytesIO()
        try:
            image.save(buffered, format="JPEG")
        except Exception:
            image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def txt_image_chat(dialog, messages, stream=True, **kwargs):
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."
    #qwen2.5:14b@Ollama
    refs = []
    tmp = dialog.llm_id.split("@")
    lang=kwargs['language']
    fid = None
    llm_id = tmp[0]
    if len(tmp)>1: fid = tmp[1]
    #如果fid为False，则只根据llm_name查询LLMService；
    #如果fid不为False，则在查询时还会根据fid进行过滤。
    llm = LLMService.query(llm_name=llm_id) if not fid else LLMService.query(llm_name=llm_id, fid=fid)
    if not llm:
        llm = TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id) if not fid else \
            TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id, llm_factory=fid)
        if not llm:
            #
            raise LookupError("LLM(%s) not found" % dialog.llm_id)
    prompt=kwargs['prompt']
   
    # 生成答案的装饰器
    def decorate_answer(answer):
        return {"answer": answer, "reference": refs}
    
    # 根据stream选项处理模型的对话响应
    if stream:
        base64image=""
        base64image = txt_image(prompt,llm_id,lang)
        # with open(os.path.join(get_project_base_directory(), "web/src/assets/yay.jpg"), "rb") as f:
        #     base64image=image2base64(f.read())
        #alt_text = "Generated Image"  # 可以根据需要动态生成
        answer = f'data:image/jpeg;base64,{base64image}'
        yield {"answer": answer, "reference": {}}
        yield decorate_answer(answer)

def chat_solo(dialog, messages, stream=True):
    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    prompt_config = dialog.prompt_config
    tts_mdl = None
    if prompt_config.get("tts"):
        tts_mdl = LLMBundle(dialog.tenant_id, LLMType.TTS)
    msg = [{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])} for m in messages if m["role"] != "system"]
    if stream:
        last_ans = ""
        for ans in chat_mdl.chat_streamly(prompt_config.get("system", ""), msg, dialog.llm_setting):
            answer = ans
            delta_ans = ans[len(last_ans) :]
            if num_tokens_from_string(delta_ans) < 16:
                continue
            last_ans = answer
            yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans), "prompt": "", "created_at": time.time()}
        if delta_ans:
            yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans), "prompt": "", "created_at": time.time()}
    else:
        answer = chat_mdl.chat(prompt_config.get("system", ""), msg, dialog.llm_setting)
        user_content = msg[-1].get("content", "[content not available]")
        logging.info("User: {}|Assistant: {}".format(user_content, answer))
        yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, answer), "prompt": "", "created_at": time.time()}
def image_chat(select_skill,dialog, messages, stream=True, **kwargs):
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."
    refs = []
    # 查询LLM模型服务

    #qwen2.5:14b@Ollama
    tmp = dialog.llm_id.split("@")
    logging.info('dialog-------')
    logging.info(dialog)
    fid = None
    llm_id = tmp[0]
    if len(tmp)>1: fid = tmp[1]
    #如果fid为False，则只根据llm_name查询LLMService；
    #如果fid不为False，则在查询时还会根据fid进行过滤。
    llm = LLMService.query(llm_name=llm_id) if not fid else LLMService.query(llm_name=llm_id, fid=fid)
    logging.info('llm-------')
    logging.info(llm)
    logging.info(dialog.llm_id)
    if not llm:
        llm = TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id) if not fid else \
            TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id, llm_factory=fid)
        if not llm:
            #
            raise LookupError("LLM(%s) not found" % dialog.llm_id)
        max_tokens = 8192
    else:
        max_tokens = llm[0].max_tokens
    questions = [m["content"] for m in messages if m["role"] == "user"][-3:]
    logging.info(questions)
    attachments = kwargs["doc_ids"].split(",") if "doc_ids" in kwargs else None
    if "doc_ids" in messages[-1]:
        attachments = messages[-1]["doc_ids"]
        for m in messages[:-1]:
            if "doc_ids" in m:
                attachments.extend(m["doc_ids"])
    image =""
    logging.info(questions)

    if len(attachments)>0:
        bucket, name = File2DocumentService.get_storage_address(doc_id=attachments[0])
        imageBytes = get_storage_binary(bucket, name)
        image=image2base64(imageBytes)
    # 确定使用的模型类型
    if image ==""  or len(image)<10:
      raise LookupError("please upload image ")
    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    prompt_config = dialog.prompt_config
    logging.info(prompt_config)

    # 配置提示词的参数
    if select_skill=='知识库':
         prompt_config["system"]  = get_prompt_template("file_base_chat", 'default')
    elif select_skill=='日志分析':
         prompt_config["system"]  = get_prompt_template("file_base_chat", 'log')
    else:
        prompt_config["system"]  = get_prompt_template("llm_chat", 'default')
   
    # 准备消息内容
    gen_conf = dialog.llm_setting
    msg = [{"role": "system", "content": prompt_config["system"].format(**kwargs)}]
    msg.extend([{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])}
                for m in messages if m["role"] != "system"])
    
    logging.info(msg)
    logging.info(gen_conf)
    # 计算token使用量
    used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.97))
    assert len(msg) >= 2, f"message_fit_in has bug: {msg}"
    #prompt = msg[0]["content"]
    prompt=kwargs['prompt']

    #prompt = "\n\n### Query:\n%s" % " ".join(questions)
    logging.info(prompt)
    prompt_config = dialog.prompt_config
    tts_mdl = None
    if prompt_config.get("tts"):
        tts_mdl = LLMBundle(dialog.tenant_id, LLMType.TTS)
    # try to use sql if field mapping is good to go
  
    # 调整生成的最大tokens数
    if "max_tokens" in gen_conf:
        gen_conf["max_tokens"] = min(
            gen_conf["max_tokens"],
            max_tokens - used_token_count)

    # 生成答案的装饰器
    def decorate_answer(answer):
        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        return {"answer": answer, "reference": refs}

   
    if stream:
            last_ans = ""
            answer = ""
            for ans in chat_mdl.chat_streamly_image(None, msg[1:], gen_conf,image):
                answer = ans
                delta_ans = ans[len(last_ans):]
                if num_tokens_from_string(delta_ans) < 16:
                    continue
                last_ans = answer
                yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
            delta_ans = answer[len(last_ans):]
            if delta_ans:
                yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
            yield decorate_answer(answer)
    else:
            answer = chat_mdl.chat_image(None, msg[1:], gen_conf,image)
            logging.info("User: {}|Assistant: {}".format(
                msg[-1]["content"], answer))
            res = decorate_answer(answer)
            res["audio_binary"] = tts(tts_mdl, answer)
            yield res
     
#文字对话实现        
def only_chat(select_skill,dialog, messages, stream=True, **kwargs):
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."
    refs = []
    # 查询LLM模型服务

    #qwen2.5:14b@Ollama
    tmp = dialog.llm_id.split("@")
    logging.info('dialog-------')
    logging.info(dialog)
    lang=kwargs["language"]
    fid = None
    llm_id = tmp[0]
    if len(tmp)>1: fid = tmp[1]
    #如果fid为False，则只根据llm_name查询LLMService；
    #如果fid不为False，则在查询时还会根据fid进行过滤。
    llm = LLMService.query(llm_name=llm_id) if not fid else LLMService.query(llm_name=llm_id, fid=fid)
    logging.info('llm-------')
    logging.info(llm)
    if not llm:
        llm = TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id) if not fid else \
            TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id, llm_factory=fid)
        if not llm:
            #
            raise LookupError("LLM(%s) not found" % dialog.llm_id)
        max_tokens = 8192
    else:
        max_tokens = llm[0].max_tokens
    
    questions = [m["content"] for m in messages if m["role"] == "user"][-3:]
    attachments = kwargs["doc_ids"].split(",") if "doc_ids" in kwargs else None
    if "doc_ids" in messages[-1]:
        attachments = messages[-1]["doc_ids"]
        for m in messages[:-1]:
            if "doc_ids" in m:
                attachments.extend(m["doc_ids"])
   
    # 确定使用的模型类型
    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
        raise LookupError("please change LLM(%s)  model to Chat Model" % dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    prompt_config = dialog.prompt_config

    logging.info('------------only_chat2---------')

    logging.info(prompt_config)

    logging.info(chat_mdl.llm_name)

    # 配置提示词的参数
    if select_skill=='知识库':
         prompt_config["system"]  = get_prompt_template("file_base_chat", 'default')
    elif select_skill=='日志分析':
         prompt_config["system"]  = get_prompt_template("file_base_chat", 'log')
    else:
        prompt_config["system"]  = get_prompt_template("llm_chat", 'default')
   
    if lang=="en":
        prompt_config["system"]='When a user asks who you are, or during self-introduction, please respond with I am Xiao Ji, your IT operations and maintenance assistant. \n Please respond the question 【 {prompt} 】 in English .  '
  
    # 准备消息内容
    gen_conf = dialog.llm_setting
    msg = [{"role": "system", "content": prompt_config["system"].format(**kwargs)}]
    msg.extend([{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])}
                for m in messages if m["role"] != "system"])
    logging.info(gen_conf)
    # 计算token使用量
    used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.97))
    assert len(msg) >= 2, f"message_fit_in has bug: {msg}"
    prompt = msg[0]["content"]
    prompt += "\n\n### Query:\n%s" % " ".join(questions)

    prompt_config = dialog.prompt_config
    field_map = KnowledgebaseService.get_field_map(dialog.kb_ids)
    tts_mdl = None
    if prompt_config.get("tts"):
        tts_mdl = LLMBundle(dialog.tenant_id, LLMType.TTS)
    # try to use sql if field mapping is good to go
    if field_map:
        logging.info("Use SQL to retrieval:{}".format(questions[-1]))
        ans = use_sql(questions[-1], field_map, dialog.tenant_id, chat_mdl, prompt_config.get("quote", True))
        if ans:
            yield ans
            return

    for p in prompt_config["parameters"]:
        if p["key"] == "knowledge":
            continue
        if p["key"] not in kwargs and not p["optional"]:
            raise KeyError("Miss parameter: " + p["key"])
        if p["key"] not in kwargs:
            prompt_config["system"] = prompt_config["system"].replace(
                "{%s}" % p["key"], " ")

    if len(questions) > 1 and prompt_config.get("refine_multiturn"):
        questions = [full_question(dialog.tenant_id, dialog.llm_id, messages)]
    else:
        questions = questions[-1:]

    # 调整生成的最大tokens数
    if "max_tokens" in gen_conf:
        gen_conf["max_tokens"] = min(
            gen_conf["max_tokens"],
            max_tokens - used_token_count)

    # 生成答案的装饰器
    def decorate_answer(answer):
        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        return {"answer": answer, "reference": refs}
    logging.info(msg)
    logging.info('--prompt---')
    logging.info(prompt)    
    # 根据stream选项处理模型的对话响应
    if stream:
            last_ans = ""
            answer = ""
            for ans in chat_mdl.chat_streamly(prompt, msg[1:], gen_conf):
                answer = ans
                delta_ans = ans[len(last_ans):]
                if num_tokens_from_string(delta_ans) < 16:
                    continue
                last_ans = answer
                yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
            delta_ans = answer[len(last_ans):]
            if delta_ans:
                yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
            yield decorate_answer(answer)
    else:
            answer = chat_mdl.chat(prompt, msg[1:], gen_conf)
            logging.info("User: {}|Assistant: {}".format(
                msg[-1]["content"], answer))
            res = decorate_answer(answer)
            res["audio_binary"] = tts(tts_mdl, answer)
            yield res

def chat_solo(dialog, messages, stream=True):
    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    prompt_config = dialog.prompt_config
    tts_mdl = None
    if prompt_config.get("tts"):
        tts_mdl = LLMBundle(dialog.tenant_id, LLMType.TTS)
    msg = [{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])} for m in messages if m["role"] != "system"]
    if stream:
        last_ans = ""
        for ans in chat_mdl.chat_streamly(prompt_config.get("system", ""), msg, dialog.llm_setting):
            answer = ans
            delta_ans = ans[len(last_ans) :]
            if num_tokens_from_string(delta_ans) < 16:
                continue
            last_ans = answer
            yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans), "prompt": "", "created_at": time.time()}
        if delta_ans:
            yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans), "prompt": "", "created_at": time.time()}
    else:
        answer = chat_mdl.chat(prompt_config.get("system", ""), msg, dialog.llm_setting)
        user_content = msg[-1].get("content", "[content not available]")
        logging.info("User: {}|Assistant: {}".format(user_content, answer))
        yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, answer), "prompt": "", "created_at": time.time()}
def chat(dialog, messages, stream=True, **kwargs):
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."
    if not dialog.kb_ids:
        for ans in chat_solo(dialog, messages, stream):
            yield ans
        return
    lang=kwargs["language"]

    chat_start_ts = timer()

    if llm_id2llm_type(dialog.llm_id) == "image2text":
        llm_model_config = TenantLLMService.get_model_config(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        llm_model_config = TenantLLMService.get_model_config(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    max_tokens = llm_model_config.get("max_tokens", 8192)

    check_llm_ts = timer()

    langfuse_tracer = None
    langfuse_keys = TenantLangfuseService.filter_by_tenant(tenant_id=dialog.tenant_id)
    if langfuse_keys:
        langfuse = Langfuse(public_key=langfuse_keys.public_key, secret_key=langfuse_keys.secret_key, host=langfuse_keys.host)
        if langfuse.auth_check():
            langfuse_tracer = langfuse
            langfuse.trace = langfuse_tracer.trace(name=f"{dialog.name}-{llm_model_config['llm_name']}")

    check_langfuse_tracer_ts = timer()

    kbs = KnowledgebaseService.get_by_ids(dialog.kb_ids)
    embedding_list = list(set([kb.embd_id for kb in kbs]))
    if len(embedding_list) != 1:
        yield {"answer": "**ERROR**: Knowledge bases use different embedding models.", "reference": []}
        return {"answer": "**ERROR**: Knowledge bases use different embedding models.", "reference": []}

    embedding_model_name = embedding_list[0]

    retriever = settings.retrievaler

    questions = [m["content"] for m in messages if m["role"] == "user"][-3:]
    attachments = kwargs["doc_ids"].split(",") if "doc_ids" in kwargs else None
    if "doc_ids" in messages[-1]:
        attachments = messages[-1]["doc_ids"]

    create_retriever_ts = timer()

    embd_mdl = LLMBundle(dialog.tenant_id, LLMType.EMBEDDING, embedding_model_name)
    if not embd_mdl:
        raise LookupError("Embedding model(%s) not found" % embedding_model_name)

    bind_embedding_ts = timer()

    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)
        toolcall_session, tools = kwargs.get("toolcall_session"), kwargs.get("tools")
        if toolcall_session and tools:
            chat_mdl.bind_tools(toolcall_session, tools)

    bind_llm_ts = timer()

    prompt_config = dialog.prompt_config
    field_map = KnowledgebaseService.get_field_map(dialog.kb_ids)
    tts_mdl = None
    if prompt_config.get("tts"):
        tts_mdl = LLMBundle(dialog.tenant_id, LLMType.TTS)
    # try to use sql if field mapping is good to go
    if field_map:
        logging.info("Use SQL to retrieval:{}".format(questions[-1]))
        ans = use_sql(questions[-1], field_map, dialog.tenant_id, chat_mdl, prompt_config.get("quote", True))
        if ans:
            yield ans
            return

    for p in prompt_config["parameters"]:
        if p["key"] == "knowledge":
            continue
        if p["key"] not in kwargs and not p["optional"]:
            raise KeyError("Miss parameter: " + p["key"])
        if p["key"] not in kwargs:
            prompt_config["system"] = prompt_config["system"].replace("{%s}" % p["key"], " ")

    if len(questions) > 1 and prompt_config.get("refine_multiturn"):
        questions = [full_question(dialog.tenant_id, dialog.llm_id, messages)]
    else:
        questions = questions[-1:]

    refine_question_ts = timer()

    rerank_mdl = None
    if dialog.rerank_id:
        rerank_mdl = LLMBundle(dialog.tenant_id, LLMType.RERANK, dialog.rerank_id)

    bind_reranker_ts = timer()
    generate_keyword_ts = bind_reranker_ts
    thought = ""
    kbinfos = {"total": 0, "chunks": [], "doc_aggs": []}

    if "knowledge" not in [p["key"] for p in prompt_config["parameters"]]:
        knowledges = []
    else:
        if prompt_config.get("keyword", False):
            questions[-1] += keyword_extraction(chat_mdl, questions[-1])
            generate_keyword_ts = timer()

        tenant_ids = list(set([kb.tenant_id for kb in kbs]))

        knowledges = []
        if prompt_config.get("reasoning", False):
            reasoner = DeepResearcher(
                chat_mdl,
                prompt_config,
                partial(retriever.retrieval, embd_mdl=embd_mdl, tenant_ids=tenant_ids, kb_ids=dialog.kb_ids, page=1, page_size=dialog.top_n, similarity_threshold=0.2, vector_similarity_weight=0.3),
            )

            for think in reasoner.thinking(kbinfos, " ".join(questions)):
                if isinstance(think, str):
                    thought = think
                    knowledges = [t for t in think.split("\n") if t]
                elif stream:
                    yield think
        else:
            kbinfos = retriever.retrieval(
                " ".join(questions),
                embd_mdl,
                tenant_ids,
                dialog.kb_ids,
                1,
                dialog.top_n,
                dialog.similarity_threshold,
                dialog.vector_similarity_weight,
                doc_ids=attachments,
                top=dialog.top_k,
                aggs=False,
                rerank_mdl=rerank_mdl,
                rank_feature=label_question(" ".join(questions), kbs),
            )
            if prompt_config.get("tavily_api_key"):
                tav = Tavily(prompt_config["tavily_api_key"])
                tav_res = tav.retrieve_chunks(" ".join(questions))
                kbinfos["chunks"].extend(tav_res["chunks"])
                kbinfos["doc_aggs"].extend(tav_res["doc_aggs"])
            if prompt_config.get("use_kg"):
                ck = settings.kg_retrievaler.retrieval(" ".join(questions), tenant_ids, dialog.kb_ids, embd_mdl, LLMBundle(dialog.tenant_id, LLMType.CHAT))
                if ck["content_with_weight"]:
                    kbinfos["chunks"].insert(0, ck)
            knowledges = kb_prompt(kbinfos, max_tokens)

    logging.info("{}->{}".format(" ".join(questions), "\n->".join(knowledges)))
    retrieval_ts = timer()
    if not knowledges and prompt_config.get("empty_response"):
        empty_res = prompt_config["empty_response"]
        yield {"answer": empty_res, "reference": kbinfos, "prompt": "\n\n### Query:\n%s" % " ".join(questions), "audio_binary": tts(tts_mdl, empty_res)}
        return {"answer": prompt_config["empty_response"], "reference": kbinfos}

    kwargs["knowledge"] = "\n------\n" + "\n\n------\n\n".join(knowledges)
    gen_conf = dialog.llm_setting
    if lang=="en":
        prompt_config["system"]='As an intelligent assistant, I will summarize the contents of the knowledge base to answer your questions. I will list the details from the knowledge base in my response. If none of the contents in the knowledge base are relevant to your question, my response must include the sentence "No answer found in the knowledge base!" The response should take into account the chat history.Here is the knowledge base:{knowledge} This is the knowledge base. Please respond in English.'
    msg = [{"role": "system", "content": prompt_config["system"].format(**kwargs)}]
    prompt4citation = ""
    if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
        prompt4citation = citation_prompt()
    msg.extend([{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])} for m in messages if m["role"] != "system"])
    
    used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.95))
    assert len(msg) >= 2, f"message_fit_in has bug: {msg}"
    prompt = msg[0]["content"]

    if "max_tokens" in gen_conf:
        gen_conf["max_tokens"] = min(gen_conf["max_tokens"], max_tokens - used_token_count)
    logging.info(msg)
    logging.info('--prompt---')
    logging.info(prompt)
    def decorate_answer(answer):
        nonlocal prompt_config, knowledges, kwargs, kbinfos, prompt, retrieval_ts, questions, langfuse_tracer

        refs = []
        ans = answer.split("</think>")
        think = ""
        if len(ans) == 2:
            think = ans[0] + "</think>"
            answer = ans[1]

        if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
            answer = re.sub(r"##[ij]\$\$", "", answer, flags=re.DOTALL)
            idx = set([])
            if not re.search(r"##[0-9]+\$\$", answer):
                answer, idx = retriever.insert_citations(
                    answer,
                    [ck["content_ltks"] for ck in kbinfos["chunks"]],
                    [ck["vector"] for ck in kbinfos["chunks"]],
                    embd_mdl,
                    tkweight=1 - dialog.vector_similarity_weight,
                    vtweight=dialog.vector_similarity_weight,
                )
            else:
                for match in re.finditer(r"##([0-9]+)\$\$", answer):
                    i = int(match.group(1))
                    if i < len(kbinfos["chunks"]):
                        idx.add(i)

            # handle (ID: 1), ID: 2 etc.
            for match in re.finditer(r"\(\s*ID:\s*(\d+)\s*\)|ID[: ]+\s*(\d+)", answer):
                full_match = match.group(0)
                id = match.group(1) or match.group(2)
                if id:
                    i = int(id)
                    if i < len(kbinfos["chunks"]):
                        idx.add(i)
                    answer = answer.replace(full_match, f"##{i}$$")

            idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
            recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
            if not recall_docs:
                recall_docs = kbinfos["doc_aggs"]
            kbinfos["doc_aggs"] = recall_docs
            logging.info("recall_docs: {}".format(recall_docs))

            refs = deepcopy(kbinfos)
            for c in refs["chunks"]:
                if c.get("vector"):
                    del c["vector"]
        logging.info(refs)
        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model providers -> API-Key'"
        finish_chat_ts = timer()

        total_time_cost = (finish_chat_ts - chat_start_ts) * 1000
        check_llm_time_cost = (check_llm_ts - chat_start_ts) * 1000
        check_langfuse_tracer_cost = (check_langfuse_tracer_ts - check_llm_ts) * 1000
        create_retriever_time_cost = (create_retriever_ts - check_langfuse_tracer_ts) * 1000
        bind_embedding_time_cost = (bind_embedding_ts - create_retriever_ts) * 1000
        bind_llm_time_cost = (bind_llm_ts - bind_embedding_ts) * 1000
        refine_question_time_cost = (refine_question_ts - bind_llm_ts) * 1000
        bind_reranker_time_cost = (bind_reranker_ts - refine_question_ts) * 1000
        generate_keyword_time_cost = (generate_keyword_ts - bind_reranker_ts) * 1000
        retrieval_time_cost = (retrieval_ts - generate_keyword_ts) * 1000
        generate_result_time_cost = (finish_chat_ts - retrieval_ts) * 1000
        tk_num = num_tokens_from_string(think + answer)
        prompt += "\n\n### Query:\n%s" % " ".join(questions)
        prompt = (
            f"{prompt}\n\n"
            "## Time elapsed:\n"
            f"  - Total: {total_time_cost:.1f}ms\n"
            f"  - Check LLM: {check_llm_time_cost:.1f}ms\n"
            f"  - Check Langfuse tracer: {check_langfuse_tracer_cost:.1f}ms\n"
            f"  - Create retriever: {create_retriever_time_cost:.1f}ms\n"
            f"  - Bind embedding: {bind_embedding_time_cost:.1f}ms\n"
            f"  - Bind LLM: {bind_llm_time_cost:.1f}ms\n"
            f"  - Multi-turn optimization: {refine_question_time_cost:.1f}ms\n"
            f"  - Bind reranker: {bind_reranker_time_cost:.1f}ms\n"
            f"  - Generate keyword: {generate_keyword_time_cost:.1f}ms\n"
            f"  - Retrieval: {retrieval_time_cost:.1f}ms\n"
            f"  - Generate answer: {generate_result_time_cost:.1f}ms\n\n"
            "## Token usage:\n"
            f"  - Generated tokens(approximately): {tk_num}\n"
            f"  - Token speed: {int(tk_num / (generate_result_time_cost / 1000.0))}/s"
        )

        langfuse_output = "\n" + re.sub(r"^.*?(### Query:.*)", r"\1", prompt, flags=re.DOTALL)
        langfuse_output = {"time_elapsed:": re.sub(r"\n", "  \n", langfuse_output), "created_at": time.time()}

        # Add a condition check to call the end method only if langfuse_tracer exists
        if langfuse_tracer and "langfuse_generation" in locals():
            langfuse_generation.end(output=langfuse_output)

        return {"answer": think + answer, "reference": refs, "prompt": re.sub(r"\n", "  \n", prompt), "created_at": time.time()}

    if langfuse_tracer:
        langfuse_generation = langfuse_tracer.trace.generation(name="chat", model=llm_model_config["llm_name"], input={"prompt": prompt, "prompt4citation": prompt4citation, "messages": msg})

    if stream:
        last_ans = ""
        answer = ""
        for ans in chat_mdl.chat_streamly(prompt + prompt4citation, msg[1:], gen_conf):
            if thought:
                ans = re.sub(r"<think>.*</think>", "", ans, flags=re.DOTALL)
            answer = ans
            delta_ans = ans[len(last_ans) :]
            if num_tokens_from_string(delta_ans) < 16:
                continue
            last_ans = answer
            yield {"answer": thought + answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
        delta_ans = answer[len(last_ans) :]
        if delta_ans:
            yield {"answer": thought + answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
        yield decorate_answer(thought + answer)
    else:
        answer = chat_mdl.chat(prompt + prompt4citation, msg[1:], gen_conf)
        user_content = msg[-1].get("content", "[content not available]")
        logging.info("User: {}|Assistant: {}".format(user_content, answer))
        res = decorate_answer(answer)
        res["audio_binary"] = tts(tts_mdl, answer)
        yield res


def use_sql(question, field_map, tenant_id, chat_mdl, quota=True):
    sys_prompt = "You are a Database Administrator. You need to check the fields of the following tables based on the user's list of questions and write the SQL corresponding to the last question."
    user_prompt = """
Table name: {};
Table of database fields are as follows:
{}

Question are as follows:
{}
Please write the SQL, only SQL, without any other explanations or text.
""".format(index_name(tenant_id), "\n".join([f"{k}: {v}" for k, v in field_map.items()]), question)
    tried_times = 0

    def get_table():
        nonlocal sys_prompt, user_prompt, question, tried_times
        sql = chat_mdl.chat(sys_prompt, [{"role": "user", "content": user_prompt}], {"temperature": 0.06})
        sql = re.sub(r"<think>.*</think>", "", sql, flags=re.DOTALL)
        logging.info(f"{question} ==> {user_prompt} get SQL: {sql}")
        sql = re.sub(r"[\r\n]+", " ", sql.lower())
        sql = re.sub(r".*select ", "select ", sql.lower())
        sql = re.sub(r" +", " ", sql)
        sql = re.sub(r"([;；]|```).*", "", sql)
        if sql[: len("select ")] != "select ":
            return None, None
        if not re.search(r"((sum|avg|max|min)\(|group by )", sql.lower()):
            if sql[: len("select *")] != "select *":
                sql = "select doc_id,docnm_kwd," + sql[6:]
            else:
                flds = []
                for k in field_map.keys():
                    if k in forbidden_select_fields4resume:
                        continue
                    if len(flds) > 11:
                        break
                    flds.append(k)
                sql = "select doc_id,docnm_kwd," + ",".join(flds) + sql[8:]

        logging.info(f"{question} get SQL(refined): {sql}")
        tried_times += 1
        return settings.retrievaler.sql_retrieval(sql, format="json"), sql

    tbl, sql = get_table()
    if tbl is None:
        return None
    if tbl.get("error") and tried_times <= 2:
        user_prompt = """
        Table name: {};
        Table of database fields are as follows:
        {}

        Question are as follows:
        {}
        Please write the SQL, only SQL, without any other explanations or text.


        The SQL error you provided last time is as follows:
        {}

        Error issued by database as follows:
        {}

        Please correct the error and write SQL again, only SQL, without any other explanations or text.
        """.format(index_name(tenant_id), "\n".join([f"{k}: {v}" for k, v in field_map.items()]), question, sql, tbl["error"])
        tbl, sql = get_table()
        logging.info("TRY it again: {}".format(sql))

    logging.info("GET table: {}".format(tbl))
    if tbl.get("error") or len(tbl["rows"]) == 0:
        return None

    docid_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"] == "doc_id"])
    doc_name_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"] == "docnm_kwd"])
    column_idx = [ii for ii in range(len(tbl["columns"])) if ii not in (docid_idx | doc_name_idx)]

    # compose Markdown table
    columns = (
            "|" + "|".join([re.sub(r"(/.*|（[^（）]+）)", "", field_map.get(tbl["columns"][i]["name"], tbl["columns"][i]["name"])) for i in column_idx]) + ("|Source|" if docid_idx and docid_idx else "|")
    )

    line = "|" + "|".join(["------" for _ in range(len(column_idx))]) + ("|------|" if docid_idx and docid_idx else "")

    rows = ["|" + "|".join([rmSpace(str(r[i])) for i in column_idx]).replace("None", " ") + "|" for r in tbl["rows"]]
    rows = [r for r in rows if re.sub(r"[ |]+", "", r)]
    if quota:
        rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
    else:
        rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
    rows = re.sub(r"T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+Z)?\|", "|", rows)

    if not docid_idx or not doc_name_idx:
        logging.warning("SQL missing field: " + sql)
        return {"answer": "\n".join([columns, line, rows]), "reference": {"chunks": [], "doc_aggs": []}, "prompt": sys_prompt}

    docid_idx = list(docid_idx)[0]
    doc_name_idx = list(doc_name_idx)[0]
    doc_aggs = {}
    for r in tbl["rows"]:
        if r[docid_idx] not in doc_aggs:
            doc_aggs[r[docid_idx]] = {"doc_name": r[doc_name_idx], "count": 0}
        doc_aggs[r[docid_idx]]["count"] += 1
    return {
        "answer": "\n".join([columns, line, rows]),
        "reference": {
            "chunks": [{"doc_id": r[docid_idx], "docnm_kwd": r[doc_name_idx]} for r in tbl["rows"]],
            "doc_aggs": [{"doc_id": did, "doc_name": d["doc_name"], "count": d["count"]} for did, d in doc_aggs.items()],
        },
        "prompt": sys_prompt,
    }


def tts(tts_mdl, text):
    if not tts_mdl or not text:
        return
    bin = b""
    for chunk in tts_mdl.tts(text):
        bin += chunk
    return binascii.hexlify(bin).decode("utf-8")


def ask(question, kb_ids, tenant_id):
    kbs = KnowledgebaseService.get_by_ids(kb_ids)
    embedding_list = list(set([kb.embd_id for kb in kbs]))

    is_knowledge_graph = all([kb.parser_id == ParserType.KG for kb in kbs])
    retriever = settings.retrievaler if not is_knowledge_graph else settings.kg_retrievaler

    embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING, embedding_list[0])
    chat_mdl = LLMBundle(tenant_id, LLMType.CHAT)
    max_tokens = chat_mdl.max_length
    tenant_ids = list(set([kb.tenant_id for kb in kbs]))
    kbinfos = retriever.retrieval(question, embd_mdl, tenant_ids, kb_ids, 1, 12, 0.1, 0.3, aggs=False, rank_feature=label_question(question, kbs))
    knowledges = kb_prompt(kbinfos, max_tokens)
    prompt = """
    Role: You're a smart assistant. Your name is Miss R.
    Task: Summarize the information from knowledge bases and answer user's question.
    Requirements and restriction:
      - DO NOT make things up, especially for numbers.
      - If the information from knowledge is irrelevant with user's question, JUST SAY: Sorry, no relevant information provided.
      - Answer with markdown format text.
      - Answer in language of user's question.
      - DO NOT make things up, especially for numbers.

    ### Information from knowledge bases
    %s

    The above is information from knowledge bases.

    """ % "\n".join(knowledges)
    msg = [{"role": "user", "content": question}]

    def decorate_answer(answer):
        nonlocal knowledges, kbinfos, prompt
        answer, idx = retriever.insert_citations(answer, [ck["content_ltks"] for ck in kbinfos["chunks"]], [ck["vector"] for ck in kbinfos["chunks"]], embd_mdl, tkweight=0.7, vtweight=0.3)
        idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
        recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
        if not recall_docs:
            recall_docs = kbinfos["doc_aggs"]
        kbinfos["doc_aggs"] = recall_docs
        refs = deepcopy(kbinfos)
        for c in refs["chunks"]:
            if c.get("vector"):
                del c["vector"]

        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        refs["chunks"] = chunks_format(refs)
        return {"answer": answer, "reference": refs}

    answer = ""
    for ans in chat_mdl.chat_streamly(prompt, msg, {"temperature": 0.1}):
        answer = ans
        yield {"answer": answer, "reference": {}}
    yield decorate_answer(answer)
