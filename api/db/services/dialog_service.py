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
import logging
import binascii
import os
import json
import json_repair
import re
from collections import defaultdict
from copy import deepcopy
from timeit import default_timer as timer
import datetime
from datetime import timedelta
from typing import Optional, Union
from api.db import LLMType, ParserType, StatusEnum
from api.db.db_models import Conversation, Dialog, DB
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMService, TenantLLMService, LLMBundle
from api import settings
from api.db.services.user_service import UserTenantService
from api.settings import  retrievaler, kg_retrievaler
from graphrag.utils import get_tags_from_cache, set_tags_to_cache
from rag.app.resume import forbidden_select_fields4resume
from rag.nlp.search import index_name
from rag.settings import TAG_FLD
from rag.utils import rmSpace, num_tokens_from_string, encoder
from api.utils.file_utils import get_project_base_directory
from api.db.services.log_service import log_file_chat
from api.db.services.file2document_service import File2DocumentService
from rag.utils.storage_factory import STORAGE_IMPL

PROMPT_TEMPLATES = {
    "llm_chat": {
        "default":
            '当用户询问你是谁，或者在自我介绍时，请回答“我是小吉，你的IT运维助手。”\n'
            '{{ input }}',

        "with_history":
            'The following is a friendly conversation between a human and an AI. '
            'The AI is talkative and provides lots of specific details from its context. '
            'If the AI does not know the answer to a question, it truthfully says it does not know.\n\n'
            'Current conversation:\n'
            '{history}\n'
            'Human: {input}\n'
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
    @DB.connection_context()
    def get_list(cls, tenant_id,
                 page_number, items_per_page, orderby, desc, id, name):
        chats = cls.model.select()
        if id:
            chats = chats.where(cls.model.id == id)
        if name:
            chats = chats.where(cls.model.name == name)
        chats = chats.where(
            (cls.model.tenant_id == tenant_id)
            & (cls.model.status == StatusEnum.VALID.value)
        )
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
        for d in log_file_chat(questions[-1], doc_id ,[],stream,llm_id,max_tokens, "default"):
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


def chat(dialog, messages, stream=True, **kwargs):
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."

    chat_start_ts = timer()

    # Get llm model name and model provider name
    llm_id, model_provider = TenantLLMService.split_model_name_and_factory(dialog.llm_id)

    # Get llm model instance by model and provide name
    llm = LLMService.query(llm_name=llm_id) if not model_provider else LLMService.query(llm_name=llm_id, fid=model_provider)

    if not llm:
        # Model name is provided by tenant, but not system built-in
        llm = TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id) if not model_provider else \
            TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id, llm_factory=model_provider)
        if not llm:
            raise LookupError("LLM(%s) not found" % dialog.llm_id)
        max_tokens = 8192
    else:
        max_tokens = llm[0].max_tokens

    check_llm_ts = timer()

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
        for m in messages[:-1]:
            if "doc_ids" in m:
                attachments.extend(m["doc_ids"])

    create_retriever_ts = timer()

    embd_mdl = LLMBundle(dialog.tenant_id, LLMType.EMBEDDING, embedding_model_name)
    if not embd_mdl:
        raise LookupError("Embedding model(%s) not found" % embedding_model_name)

    bind_embedding_ts = timer()

    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    bind_llm_ts = timer()

    prompt_config = dialog.prompt_config
    field_map = KnowledgebaseService.get_field_map(dialog.kb_ids)
    tts_mdl = None
    if prompt_config.get("tts"):
        tts_mdl = LLMBundle(dialog.tenant_id, LLMType.TTS)
    # try to use sql if field mapping is good to go
    if field_map:
        logging.debug("Use SQL to retrieval:{}".format(questions[-1]))
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

    refine_question_ts = timer()

    rerank_mdl = None
    if dialog.rerank_id:
        rerank_mdl = LLMBundle(dialog.tenant_id, LLMType.RERANK, dialog.rerank_id)

    bind_reranker_ts = timer()
    generate_keyword_ts = bind_reranker_ts

    if "knowledge" not in [p["key"] for p in prompt_config["parameters"]]:
        kbinfos = {"total": 0, "chunks": [], "doc_aggs": []}
    else:
        if prompt_config.get("keyword", False):
            questions[-1] += keyword_extraction(chat_mdl, questions[-1])
            generate_keyword_ts = timer()

        tenant_ids = list(set([kb.tenant_id for kb in kbs]))

        kbinfos = retriever.retrieval(" ".join(questions), embd_mdl, tenant_ids, dialog.kb_ids, 1, dialog.top_n,
                                      dialog.similarity_threshold,
                                      dialog.vector_similarity_weight,
                                      doc_ids=attachments,
                                      top=dialog.top_k, aggs=False, rerank_mdl=rerank_mdl,
                                      rank_feature=label_question(" ".join(questions), kbs)
                                      )
        if prompt_config.get("use_kg"):
            ck = settings.kg_retrievaler.retrieval(" ".join(questions),
                                              tenant_ids,
                                              dialog.kb_ids,
                                              embd_mdl,
                                              LLMBundle(dialog.tenant_id, LLMType.CHAT))
            if ck["content_with_weight"]:
                kbinfos["chunks"].insert(0, ck)

    retrieval_ts = timer()

    knowledges = kb_prompt(kbinfos, max_tokens)
    logging.debug(
        "{}->{}".format(" ".join(questions), "\n->".join(knowledges)))

    if not knowledges and prompt_config.get("empty_response"):
        empty_res = prompt_config["empty_response"]
        yield {"answer": empty_res, "reference": kbinfos, "audio_binary": tts(tts_mdl, empty_res)}
        return {"answer": prompt_config["empty_response"], "reference": kbinfos}

    kwargs["knowledge"] = "\n------\n" + "\n\n------\n\n".join(knowledges)
    gen_conf = dialog.llm_setting

    msg = [{"role": "system", "content": prompt_config["system"].format(**kwargs)}]
    msg.extend([{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])}
                for m in messages if m["role"] != "system"])
    used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.97))
    assert len(msg) >= 2, f"message_fit_in has bug: {msg}"
    prompt = msg[0]["content"]
    prompt += "\n\n### Query:\n%s" % " ".join(questions)

    if "max_tokens" in gen_conf:
        gen_conf["max_tokens"] = min(
            gen_conf["max_tokens"],
            max_tokens - used_token_count)

    def decorate_answer(answer):
        nonlocal prompt_config, knowledges, kwargs, kbinfos, prompt, retrieval_ts

        finish_chat_ts = timer()

        refs = []
        if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
            answer, idx = retriever.insert_citations(answer,
                                                     [ck["content_ltks"]
                                                      for ck in kbinfos["chunks"]],
                                                     [ck["vector"]
                                                      for ck in kbinfos["chunks"]],
                                                     embd_mdl,
                                                     tkweight=1 - dialog.vector_similarity_weight,
                                                     vtweight=dialog.vector_similarity_weight)
            idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
            recall_docs = [
                d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
            if not recall_docs:
                recall_docs = kbinfos["doc_aggs"]
            kbinfos["doc_aggs"] = recall_docs

            refs = deepcopy(kbinfos)
            for c in refs["chunks"]:
                if c.get("vector"):
                    del c["vector"]

        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model providers -> API-Key'"
        finish_chat_ts = timer()

        total_time_cost = (finish_chat_ts - chat_start_ts) * 1000
        check_llm_time_cost = (check_llm_ts - chat_start_ts) * 1000
        create_retriever_time_cost = (create_retriever_ts - check_llm_ts) * 1000
        bind_embedding_time_cost = (bind_embedding_ts - create_retriever_ts) * 1000
        bind_llm_time_cost = (bind_llm_ts - bind_embedding_ts) * 1000
        refine_question_time_cost = (refine_question_ts - bind_llm_ts) * 1000
        bind_reranker_time_cost = (bind_reranker_ts - refine_question_ts) * 1000
        generate_keyword_time_cost = (generate_keyword_ts - bind_reranker_ts) * 1000
        retrieval_time_cost = (retrieval_ts - generate_keyword_ts) * 1000
        generate_result_time_cost = (finish_chat_ts - retrieval_ts) * 1000

        prompt = f"{prompt}\n\n - Total: {total_time_cost:.1f}ms\n  - Check LLM: {check_llm_time_cost:.1f}ms\n  - Create retriever: {create_retriever_time_cost:.1f}ms\n  - Bind embedding: {bind_embedding_time_cost:.1f}ms\n  - Bind LLM: {bind_llm_time_cost:.1f}ms\n  - Tune question: {refine_question_time_cost:.1f}ms\n  - Bind reranker: {bind_reranker_time_cost:.1f}ms\n  - Generate keyword: {generate_keyword_time_cost:.1f}ms\n  - Retrieval: {retrieval_time_cost:.1f}ms\n  - Generate answer: {generate_result_time_cost:.1f}ms"
        return {"answer": answer, "reference": refs, "prompt": re.sub(r"\n", "  \n", prompt)}

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
        logging.debug("User: {}|Assistant: {}".format(
            msg[-1]["content"], answer))
        res = decorate_answer(answer)
        res["audio_binary"] = tts(tts_mdl, answer)
        yield res

def file_chat(dialog, messages, stream=True, **kwargs):
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."
    st = timer()
    tmp = dialog.llm_id.split("@")
    fid = None
    llm_id = tmp[0]
    if len(tmp)>1: fid = tmp[1]

    llm = LLMService.query(llm_name=llm_id) if not fid else LLMService.query(llm_name=llm_id, fid=fid)
    if not llm:
        llm = TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id) if not fid else \
            TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id, llm_factory=fid)
        if not llm:
            raise LookupError("LLM(%s) not found" % dialog.llm_id)
        max_tokens = 8192
    else:
        max_tokens = llm[0].max_tokens
     # TODO
    #选择的知识库，如果多个日志文件都放在一个知识库里面 如何区分处理
    kbs = KnowledgebaseService.get_by_ids(dialog.kb_ids)
    if not kbs:
           raise LookupError("Can't find this knowledgebase!")
    embd_nms = list(set([kb.embd_id for kb in kbs]))
    # if not kbs:
    #         raise LookupError("Can't find this knowledgebase!")
    #     #找出名字中包含日志分析的知识库
    # kb=None
    # for log_kb in  kbs:
    #         if '日志分析' in log_kb.name:
    #             kb=log_kb
    #             continue
    # if(kb is None):
    #      raise LookupError("please create log knowledgebase!")
    # #直接选择日志分析的知识库
    # dialog.kb_ids= [kb.id]
    # embd_nms = list(set([kb.embd_id]))
    if len(embd_nms) != 1:
        yield {"answer": "**ERROR**: Knowledge bases use different embedding models.", "reference": []}
        return {"answer": "**ERROR**: Knowledge bases use different embedding models.", "reference": []}

    is_kg = all([kb.parser_id == ParserType.KG for kb in kbs])
    retr = retrievaler if not is_kg else kg_retrievaler
    logging.info('---------kwargs----')
    logging.info(kwargs)
    questions = [m["content"] for m in messages if m["role"] == "user"][-3:]
    logging.info('---------questions')
    logging.info(questions)

    attachments = kwargs["doc_ids"].split(",") if "doc_ids" in kwargs else None
    if "doc_ids" in messages[-1]:
        attachments = messages[-1]["doc_ids"]
        for m in messages[:-1]:
            if "doc_ids" in m:
                attachments.extend(m["doc_ids"])

    logging.info('---------attachments')
    logging.info(attachments)
    embd_mdl = LLMBundle(dialog.tenant_id, LLMType.EMBEDDING, embd_nms[0])
    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    prompt_config = dialog.prompt_config
    logging.info('---prompt_config-----')

    logging.info(prompt_config)

    field_map = KnowledgebaseService.get_field_map(dialog.kb_ids)
    tts_mdl = None
    if prompt_config.get("tts"):
        tts_mdl = LLMBundle(dialog.tenant_id, LLMType.TTS)
    # try to use sql if field mapping is good to go
    logging.info('---field_map-----')
    logging.info(field_map)
    if field_map:
        logging.info("Use SQL to retrieval:{}".format(questions[-1]))
        ans = use_sql(questions[-1], field_map, dialog.tenant_id, chat_mdl, prompt_config.get("quote", True))
        if ans:
            yield ans
            return

    # for p in prompt_config["parameters"]:
    #     if p["key"] == "knowledge":
    #         continue
    #     if p["key"] not in kwargs and not p["optional"]:
    #         raise KeyError("Miss parameter: " + p["key"])
    #     if p["key"] not in kwargs:
    #         prompt_config["system"] = prompt_config["system"].replace(
    #             "{%s}" % p["key"], " ")
    prompt_config["system"]  = get_prompt_template("file_base_chat", 'log')   
    logging.info('--2222-prompt_config-----')

    logging.info(prompt_config)
    
    if len(questions) > 1 and prompt_config.get("refine_multiturn"):
        questions = [full_question(dialog.tenant_id, dialog.llm_id, messages)]
    else:
        questions = questions[-1:]

    rerank_mdl = None
    if dialog.rerank_id:
        rerank_mdl = LLMBundle(dialog.tenant_id, LLMType.RERANK, dialog.rerank_id)
    #team_id= get_index_id(dialog.tenant_id)
    for _ in range(len(questions) // 2):
        questions.append(questions[-1])
    if "knowledge" not in [p["key"] for p in prompt_config["parameters"]]:
        kbinfos = {"total": 0, "chunks": [], "doc_aggs": []}
    else:
        logging.info('---------retrieval---')
        if prompt_config.get("keyword", False):
            questions[-1] += keyword_extraction(chat_mdl, questions[-1])
        tenant_ids = list(set([kb.tenant_id for kb in kbs]))
        kbinfos = retr.retrieval(" ".join(questions), embd_mdl, tenant_ids, dialog.kb_ids, 1, dialog.top_n,
                                        dialog.similarity_threshold,
                                        dialog.vector_similarity_weight,
                                        doc_ids=attachments,
                                        top=dialog.top_k, aggs=False, rerank_mdl=rerank_mdl)
    knowledges = [ck["content_with_weight"] for ck in kbinfos["chunks"]]
    logging.info('-----knowledges')
    logging.info(knowledges)
    logging.info("{}->{}".format(" ".join(questions), "\n->".join(knowledges)))
    retrieval_tm = timer()

    if not knowledges and prompt_config.get("empty_response"):
        empty_res = prompt_config["empty_response"]
        yield {"answer": empty_res, "reference": kbinfos, "audio_binary": tts(tts_mdl, empty_res)}
        return {"answer": prompt_config["empty_response"], "reference": kbinfos}

    kwargs["knowledge"] = "\n\n------\n\n".join(knowledges)
   
    logging.info('-----kwargs')
    logging.info(kwargs)

    gen_conf = dialog.llm_setting

    msg = [{"role": "system", "content": prompt_config["system"].format(**kwargs)}]
    logging.info('msg---------')
    logging.info(msg)


    msg.extend([{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])}
                for m in messages if m["role"] != "system"])
    used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.97))
    assert len(msg) >= 2, f"message_fit_in has bug: {msg}"
    prompt = msg[0]["content"]
    #prompt += "\n\n### Query:\n%s" % " ".join(questions)


    logging.info('=====prompt=====')

    logging.info(prompt)

    if "max_tokens" in gen_conf:
        gen_conf["max_tokens"] = min(
            gen_conf["max_tokens"],
            max_tokens - used_token_count)

    def decorate_answer(answer):
        nonlocal prompt_config, knowledges, kwargs, kbinfos, prompt, retrieval_tm
        refs = []
        if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
            answer, idx = retr.insert_citations(answer,
                                                       [ck["content_ltks"]
                                                        for ck in kbinfos["chunks"]],
                                                       [ck["vector"]
                                                        for ck in kbinfos["chunks"]],
                                                       embd_mdl,
                                                       tkweight=1 - dialog.vector_similarity_weight,
                                                       vtweight=dialog.vector_similarity_weight)
            idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
            recall_docs = [
                d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
            if not recall_docs: recall_docs = kbinfos["doc_aggs"]
            kbinfos["doc_aggs"] = recall_docs

            refs = deepcopy(kbinfos)
            for c in refs["chunks"]:
                if c.get("vector"):
                    del c["vector"]

        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        done_tm = timer()
        prompt += "\n\n### Elapsed\n  - Retrieval: %.1f ms\n  - LLM: %.1f ms"%((retrieval_tm-st)*1000, (done_tm-st)*1000)
        return {"answer": answer, "reference": refs, "prompt": prompt}

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
    logging.info("--model_type")
    logging.info (llm_id2llm_type(dialog.llm_id))
   
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
    logging.info('------------only_chat3---------')

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
        logging.debug("Use SQL to retrieval:{}".format(questions[-1]))
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

   
    if stream:
            last_ans = ""
            answer = ""
            for ans in chat_mdl.chat_streamly_image(prompt, msg[1:], gen_conf,image):
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
            answer = chat_mdl.chat_image(prompt, msg[1:], gen_conf,image)
            logging.debug("User: {}|Assistant: {}".format(
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
   
  
    # 准备消息内容
    gen_conf = dialog.llm_setting
    msg = [{"role": "system", "content": prompt_config["system"].format(**kwargs)}]
    msg.extend([{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])}
                for m in messages if m["role"] != "system"])
 
    
    logging.info('------------only_chat3---------')

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
        logging.debug("Use SQL to retrieval:{}".format(questions[-1]))
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

    logging.info('---only_chat4--')
    logging.info(msg)
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
            logging.debug("User: {}|Assistant: {}".format(
                msg[-1]["content"], answer))
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
""".format(
        index_name(tenant_id),
        "\n".join([f"{k}: {v}" for k, v in field_map.items()]),
        question
    )
    tried_times = 0

    def get_table():
        nonlocal sys_prompt, user_prompt, question, tried_times
        sql = chat_mdl.chat(sys_prompt, [{"role": "user", "content": user_prompt}], {
            "temperature": 0.06})
        logging.debug(f"{question} ==> {user_prompt} get SQL: {sql}")
        sql = re.sub(r"[\r\n]+", " ", sql.lower())
        sql = re.sub(r".*select ", "select ", sql.lower())
        sql = re.sub(r" +", " ", sql)
        sql = re.sub(r"([;；]|```).*", "", sql)
        if sql[:len("select ")] != "select ":
            return None, None
        if not re.search(r"((sum|avg|max|min)\(|group by )", sql.lower()):
            if sql[:len("select *")] != "select *":
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

        logging.debug(f"{question} get SQL(refined): {sql}")
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
        """.format(
            index_name(tenant_id),
            "\n".join([f"{k}: {v}" for k, v in field_map.items()]),
            question, sql, tbl["error"]
        )
        tbl, sql = get_table()
        logging.debug("TRY it again: {}".format(sql))

    logging.debug("GET table: {}".format(tbl))
    if tbl.get("error") or len(tbl["rows"]) == 0:
        return None

    docid_idx = set([ii for ii, c in enumerate(
        tbl["columns"]) if c["name"] == "doc_id"])
    doc_name_idx = set([ii for ii, c in enumerate(
        tbl["columns"]) if c["name"] == "docnm_kwd"])
    column_idx = [ii for ii in range(
        len(tbl["columns"])) if ii not in (docid_idx | doc_name_idx)]

    # compose Markdown table
    columns = "|" + "|".join([re.sub(r"(/.*|（[^（）]+）)", "", field_map.get(tbl["columns"][i]["name"],
                                                                          tbl["columns"][i]["name"])) for i in
                              column_idx]) + ("|Source|" if docid_idx and docid_idx else "|")

    line = "|" + "|".join(["------" for _ in range(len(column_idx))]) + \
           ("|------|" if docid_idx and docid_idx else "")

    rows = ["|" +
            "|".join([rmSpace(str(r[i])) for i in column_idx]).replace("None", " ") +
            "|" for r in tbl["rows"]]
    rows = [r for r in rows if re.sub(r"[ |]+", "", r)]
    if quota:
        rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
    else:
        rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
    rows = re.sub(r"T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+Z)?\|", "|", rows)

    if not docid_idx or not doc_name_idx:
        logging.warning("SQL missing field: " + sql)
        return {
            "answer": "\n".join([columns, line, rows]),
            "reference": {"chunks": [], "doc_aggs": []},
            "prompt": sys_prompt
        }

    docid_idx = list(docid_idx)[0]
    doc_name_idx = list(doc_name_idx)[0]
    doc_aggs = {}
    for r in tbl["rows"]:
        if r[docid_idx] not in doc_aggs:
            doc_aggs[r[docid_idx]] = {"doc_name": r[doc_name_idx], "count": 0}
        doc_aggs[r[docid_idx]]["count"] += 1
    return {
        "answer": "\n".join([columns, line, rows]),
        "reference": {"chunks": [{"doc_id": r[docid_idx], "docnm_kwd": r[doc_name_idx]} for r in tbl["rows"]],
                      "doc_aggs": [{"doc_id": did, "doc_name": d["doc_name"], "count": d["count"]} for did, d in
                                   doc_aggs.items()]},
        "prompt": sys_prompt
    }


def relevant(tenant_id, llm_id, question, contents: list):
    if llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
    prompt = """
        You are a grader assessing relevance of a retrieved document to a user question. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        No other words needed except 'yes' or 'no'.
    """
    if not contents:
        return False
    contents = "Documents: \n" + "   - ".join(contents)
    contents = f"Question: {question}\n" + contents
    if num_tokens_from_string(contents) >= chat_mdl.max_length - 4:
        contents = encoder.decode(encoder.encode(contents)[:chat_mdl.max_length - 4])
    ans = chat_mdl.chat(prompt, [{"role": "user", "content": contents}], {"temperature": 0.01})
    if ans.lower().find("yes") >= 0:
        return True
    return False


def rewrite(tenant_id, llm_id, question):
    if llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
    prompt = """
        You are an expert at query expansion to generate a paraphrasing of a question.
        I can't retrieval relevant information from the knowledge base by using user's question directly.     
        You need to expand or paraphrase user's question by multiple ways such as using synonyms words/phrase, 
        writing the abbreviation in its entirety, adding some extra descriptions or explanations, 
        changing the way of expression, translating the original question into another language (English/Chinese), etc. 
        And return 5 versions of question and one is from translation.
        Just list the question. No other words are needed.
    """
    ans = chat_mdl.chat(prompt, [{"role": "user", "content": question}], {"temperature": 0.8})
    return ans


def keyword_extraction(chat_mdl, content, topn=3):
    prompt = f"""
Role: You're a text analyzer. 
Task: extract the most important keywords/phrases of a given piece of text content.
Requirements: 
  - Summarize the text content, and give top {topn} important keywords/phrases.
  - The keywords MUST be in language of the given piece of text content.
  - The keywords are delimited by ENGLISH COMMA.
  - Keywords ONLY in output.

### Text Content 
{content}

"""
    msg = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Output: "}
    ]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


def question_proposal(chat_mdl, content, topn=3):
    prompt = f"""
Role: You're a text analyzer. 
Task:  propose {topn} questions about a given piece of text content.
Requirements: 
  - Understand and summarize the text content, and propose top {topn} important questions.
  - The questions SHOULD NOT have overlapping meanings.
  - The questions SHOULD cover the main content of the text as much as possible.
  - The questions MUST be in language of the given piece of text content.
  - One question per line.
  - Question ONLY in output.

### Text Content 
{content}

"""
    msg = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Output: "}
    ]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


def full_question(tenant_id, llm_id, messages):
    if llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
    conv = []
    for m in messages:
        if m["role"] not in ["user", "assistant"]:
            continue
        conv.append("{}: {}".format(m["role"].upper(), m["content"]))
    conv = "\n".join(conv)
    today = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - timedelta(days=1)).isoformat()
    tomorrow = (datetime.date.today() + timedelta(days=1)).isoformat()
    prompt = f"""
Role: A helpful assistant

Task and steps: 
    1. Generate a full user question that would follow the conversation.
    2. If the user's question involves relative date, you need to convert it into absolute date based on the current date, which is {today}. For example: 'yesterday' would be converted to {yesterday}.
    
Requirements & Restrictions:
  - Text generated MUST be in the same language of the original user's question.
  - If the user's latest question is completely, don't do anything, just return the original question.
  - DON'T generate anything except a refined question.

######################
-Examples-
######################

# Example 1
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
###############
Output: What's the name of Donald Trump's mother?

------------
# Example 2
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
ASSISTANT:  Mary Trump.
User: What's her full name?
###############
Output: What's the full name of Donald Trump's mother Mary Trump?

------------
# Example 3
## Conversation
USER: What's the weather today in London?
ASSISTANT:  Cloudy.
USER: What's about tomorrow in Rochester?
###############
Output: What's the weather in Rochester on {tomorrow}?
######################

# Real Data
## Conversation
{conv}
###############
    """
    ans = chat_mdl.chat(prompt, [{"role": "user", "content": "Output: "}], {"temperature": 0.2})
    return ans if ans.find("**ERROR**") < 0 else messages[-1]["content"]


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
    kbinfos = retriever.retrieval(question, embd_mdl, tenant_ids, kb_ids,
                                  1, 12, 0.1, 0.3, aggs=False,
                                  rank_feature=label_question(question, kbs)
                                  )
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
        answer, idx = retriever.insert_citations(answer,
                                                 [ck["content_ltks"]
                                                  for ck in kbinfos["chunks"]],
                                                 [ck["vector"]
                                                  for ck in kbinfos["chunks"]],
                                                 embd_mdl,
                                                 tkweight=0.7,
                                                 vtweight=0.3)
        idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
        recall_docs = [
            d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
        if not recall_docs:
            recall_docs = kbinfos["doc_aggs"]
        kbinfos["doc_aggs"] = recall_docs
        refs = deepcopy(kbinfos)
        for c in refs["chunks"]:
            if c.get("vector"):
                del c["vector"]

        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        return {"answer": answer, "reference": refs}

    answer = ""
    for ans in chat_mdl.chat_streamly(prompt, msg, {"temperature": 0.1}):
        answer = ans
        yield {"answer": answer, "reference": {}}
    yield decorate_answer(answer)


def content_tagging(chat_mdl, content, all_tags, examples, topn=3):
    prompt = f"""
Role: You're a text analyzer. 

Task: Tag (put on some labels) to a given piece of text content based on the examples and the entire tag set.

Steps:: 
  - Comprehend the tag/label set.
  - Comprehend examples which all consist of both text content and assigned tags with relevance score in format of JSON.
  - Summarize the text content, and tag it with top {topn} most relevant tags from the set of tag/label and the corresponding relevance score.

Requirements
  - The tags MUST be from the tag set.
  - The output MUST be in JSON format only, the key is tag and the value is its relevance score.
  - The relevance score must be range from 1 to 10.
  - Keywords ONLY in output.

# TAG SET
{", ".join(all_tags)}

"""
    for i, ex in enumerate(examples):
        prompt += """
# Examples {}
### Text Content
{}

Output:
{}

        """.format(i, ex["content"], json.dumps(ex[TAG_FLD], indent=2, ensure_ascii=False))

    prompt += f"""
# Real Data
### Text Content
{content}

"""
    msg = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Output: "}
    ]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.5})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    if kwd.find("**ERROR**") >= 0:
        raise Exception(kwd)

    try:
        return json_repair.loads(kwd)
    except json_repair.JSONDecodeError:
        try:
            result = kwd.replace(prompt[:-1], '').replace('user', '').replace('model', '').strip()
            result = '{' + result.split('{')[1].split('}')[0] + '}'
            return json_repair.loads(result)
        except Exception as e:
            logging.exception(f"JSON parsing error: {result} -> {e}")
            raise e
