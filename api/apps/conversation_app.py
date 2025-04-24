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
import json
import logging
import re
import time
import traceback
from copy import deepcopy
from typing import Union
from api.db.services.cmdb_service import cmdb_chat_stream

import trio
from flask import Response, request
from flask_login import current_user, login_required

from api import settings
from api.db import LLMType
from api.db.db_models import APIToken
from api.db.services.conversation_service import ConversationService, structure_answer

from api.db.services.dialog_service import DialogService, ConversationService, ask, image_chat, log_chat, log_chat, only_chat, chat, txt_image_chat
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle, TenantService
from api.db.services.user_service import UserTenantService
from api.utils.api_utils import get_data_error_result, get_json_result, server_error_response, validate_request
from graphrag.general.mind_map_extractor import MindMapExtractor
from rag.app.tag import label_question


@manager.route("/set", methods=["POST"])  # noqa: F821
@login_required
def set_conversation():
    req = request.json
    conv_id = req.get("conversation_id")
    is_new = req.get("is_new")
    del req["is_new"]
    if not is_new:
        del req["conversation_id"]
        try:
            if not ConversationService.update_by_id(conv_id, req):
                return get_data_error_result(message="Conversation not found!")
            e, conv = ConversationService.get_by_id(conv_id)
            if not e:
                return get_data_error_result(message="Fail to update a conversation!")
            conv = conv.to_dict()
            return get_json_result(data=conv)
        except Exception as e:
            return server_error_response(e)

    try:
        e, dia = DialogService.get_by_id(req["dialog_id"])
        if not e:
            return get_data_error_result(message="Dialog not found")
        conv = {"id": conv_id, "dialog_id": req["dialog_id"], "name": req.get("name", "New conversation"), "message": [{"role": "assistant", "content": dia.prompt_config["prologue"]}]}
        ConversationService.save(**conv)
        return get_json_result(data=conv)
    except Exception as e:
        return server_error_response(e)


@manager.route("/get", methods=["GET"])  # noqa: F821
@login_required
def get():
    conv_id = request.args["conversation_id"]
    try:
        e, conv = ConversationService.get_by_id(conv_id)
        if not e:
            return get_data_error_result(message="Conversation not found!")
        tenants = UserTenantService.query(user_id=current_user.id)
        avatar = None
        for tenant in tenants:
            dialog = DialogService.query(tenant_id=tenant.tenant_id, id=conv.dialog_id)
            if dialog and len(dialog) > 0:
                avatar = dialog[0].icon
                break
        else:
            return get_json_result(data=False, message="Only owner of conversation authorized for this operation.", code=settings.RetCode.OPERATING_ERROR)

        def get_value(d, k1, k2):
            return d.get(k1, d.get(k2))

        for ref in conv.reference:
            if isinstance(ref, list):
                continue
            ref["chunks"] = [
                {
                    "id": get_value(ck, "chunk_id", "id"),
                    "content": get_value(ck, "content", "content_with_weight"),
                    "document_id": get_value(ck, "doc_id", "document_id"),
                    "document_name": get_value(ck, "docnm_kwd", "document_name"),
                    "dataset_id": get_value(ck, "kb_id", "dataset_id"),
                    "image_id": get_value(ck, "image_id", "img_id"),
                    "positions": get_value(ck, "positions", "position_int"),
                }
                for ck in ref.get("chunks", [])
            ]

        conv = conv.to_dict()
        conv["avatar"] = avatar
        return get_json_result(data=conv)
    except Exception as e:
        return server_error_response(e)


@manager.route("/getsse/<dialog_id>", methods=["GET"])  # type: ignore # noqa: F821
def getsse(dialog_id):
    token = request.headers.get("Authorization").split()
    if len(token) != 2:
        return get_data_error_result(message='Authorization is not valid!"')
    token = token[1]
    objs = APIToken.query(beta=token)
    if not objs:
        return get_data_error_result(message='Authentication error: API key is invalid!"')
    try:
        e, conv = DialogService.get_by_id(dialog_id)
        if not e:
            return get_data_error_result(message="Dialog not found!")
        conv = conv.to_dict()
        conv["avatar"] = conv["icon"]
        del conv["icon"]
        return get_json_result(data=conv)
    except Exception as e:
        return server_error_response(e)


@manager.route("/rm", methods=["POST"])  # noqa: F821
@login_required
def rm():
    conv_ids = request.json["conversation_ids"]
    try:
        for cid in conv_ids:
            exist, conv = ConversationService.get_by_id(cid)
            if not exist:
                return get_data_error_result(message="Conversation not found!")
            tenants = UserTenantService.query(user_id=current_user.id)
            for tenant in tenants:
                if DialogService.query(tenant_id=tenant.tenant_id, id=conv.dialog_id):
                    break
            else:
                return get_json_result(data=False, message="Only owner of conversation authorized for this operation.", code=settings.RetCode.OPERATING_ERROR)
            ConversationService.delete_by_id(cid)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route("/list", methods=["GET"])  # noqa: F821
@login_required
def list_convsersation():
    dialog_id = request.args["dialog_id"]
    try:
        if not DialogService.query(tenant_id=current_user.id, id=dialog_id):
            return get_json_result(data=False, message="Only owner of dialog authorized for this operation.", code=settings.RetCode.OPERATING_ERROR)
        convs = ConversationService.query(dialog_id=dialog_id, order_by=ConversationService.model.create_time, reverse=True)

        convs = [d.to_dict() for d in convs]
        return get_json_result(data=convs)
    except Exception as e:
        return server_error_response(e)


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


 # 生成答案的装饰器
def decorate_answer(answer):
        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        return {"answer": answer, "reference": {}}
 
 
@manager.route("/completion", methods=["POST"])
@login_required
@validate_request("conversation_id", "messages")
def completion():
    req = request.json
    msg = []
    logging.info('-------completion----')
   
    logging.info(req)
    prompt = req["prompt"]
    selectedSkill = req["selectedSkill"]
    for m in req["messages"]:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant" and not msg:
            continue
        msg.append(m)
    message_id = msg[-1].get("id")
    try:
        e, conv = ConversationService.get_by_id(req["conversation_id"])
        if not e:
            return get_data_error_result(message="Conversation not found!")
        conv.message = deepcopy(req["messages"])
        e, dia = DialogService.get_by_id(conv.dialog_id)
        if not e:
            return get_data_error_result(message="Dialog not found!")
        del req["conversation_id"]
        del req["messages"]

        if not conv.reference:
            conv.reference = []
        else:

            def get_value(d, k1, k2):
                return d.get(k1, d.get(k2))

            for ref in conv.reference:
                if isinstance(ref, list):
                    continue
                ref["chunks"] = [
                    {
                        "id": get_value(ck, "chunk_id", "id"),
                        "content": get_value(ck, "content", "content_with_weight"),
                        "document_id": get_value(ck, "doc_id", "document_id"),
                        "document_name": get_value(ck, "docnm_kwd", "document_name"),
                        "dataset_id": get_value(ck, "kb_id", "dataset_id"),
                        "image_id": get_value(ck, "image_id", "img_id"),
                        "positions": get_value(ck, "positions", "position_int"),
                    }
                    for ck in ref.get("chunks", [])
                ]

        if not conv.reference:
            conv.reference = []
        conv.reference.append({"chunks": [], "doc_aggs": []})
       
        def stream():
            nonlocal dia, msg, req, conv,message_id
            try:
                if selectedSkill=='知识库' or selectedSkill=='KNOWLEDGE' or selectedSkill=='知識庫':
                    for ans in chat(dia, msg, True, **req):
                        ans = structure_answer(conv, ans, message_id, conv.id)
                        conv.message[-1] = {"role": "assistant", "content": ans["answer"],"id": message_id, "prompt": ans.get("prompt", ""),"selectedSkill":selectedSkill}
                        yield "data:" + json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
                        #fillin_conv(ans)
                        #yield "data:" + json.dumps({"retcode": 0, "retmsg": "", "data": ans}, ensure_ascii=False) + "\n\n"
                    ConversationService.update_by_id(conv.id, conv.to_dict())
                elif selectedSkill=='日志分析' or selectedSkill=='LOG' or selectedSkill=='日誌分析' :
                    logging.info('-------日志分析----')
                    for ans in log_chat(dia, msg, True, **req):
                        ans = structure_answer(conv, ans, message_id, conv.id)
                        conv.message[-1] = {"role": "assistant", "content": ans["answer"],"id": message_id, "prompt": ans.get("prompt", ""),"selectedSkill":selectedSkill}
                        yield "data:"+json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
                    ConversationService.update_by_id(conv.id, conv.to_dict())
                elif selectedSkill=='CMDB':
                    logging.info('--------prompt----------')
                    logging.info(prompt)
                    answer = ""
                    result = cmdb_chat_stream(prompt)
                    logging.info('---------CMDB start----------')
                    logging.info(result)
                    for line in result:
                            if error_msg := check_error_msg(line):  # check whether error occured
                                logging.error(error_msg)
                                answer=" 后台服务错误，请重试查询"
                                ans = decorate_answer(answer)
                                ans = structure_answer(conv, ans, message_id, conv.id)
                                conv.message[-1] = {"role": "assistant", "content": ans["answer"],"id": message_id, "prompt": ans.get("prompt", ""),"selectedSkill":selectedSkill}
                                yield "data:"+json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
                                ConversationService.update_by_id(conv.id, conv.to_dict())
                                break
                            lines = line.split("\n")
                            if len(lines)>1:
                                time.sleep(0.5)
                                chunk=lines[1]
                                if len(chunk)>2:
                                    logging.info(chunk)
                                    logging.info("chunk-------------------")
                                    if "data:" in chunk:
                                        json_data=json.loads(chunk[6:]) 
                                        logging.info(json_data)
                                    # 检查并打印run_id
                                        if "run_id" in json_data:
                                            message_id= json_data['run_id']
                                            logging.info(f"cmdb message_id: {message_id}")
                                        if "steps" in json_data:
                                            answer +=json_data['steps'][0]['action']['log']
                                            ans = decorate_answer(answer)
                                            ans = structure_answer(conv, ans, message_id, conv.id)
                                            conv.message[-1] = {"role": "assistant", "content": ans["answer"],"id": message_id, "prompt": ans.get("prompt", ""),"selectedSkill":selectedSkill}
                                            yield "data:"+json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
                                            
                                         #将output 翻译为中文
                                        if "output" in json_data:
                                            answer +=json_data['output']
                                            ans = decorate_answer(answer)
                                            ans = structure_answer(conv, ans, message_id, conv.id)
                                            conv.message[-1] = {"role": "assistant", "content": ans["answer"],"id": message_id, "prompt": ans.get("prompt", ""),"selectedSkill":selectedSkill}

                                            yield "data:"+json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
                                            # output= json_data['output']
                                            # logging.info(output)
                                            # rr = cmdb_chat_chinese(
                                            #     f"请翻译 {output}",
                                            # )
                                            # logging.info(rr)
                                            # ch_text=rr["choices"][0]["message"]["content"]
                                            
                    ConversationService.update_by_id(conv.id, conv.to_dict())
                    logging.info('---------CMDB end----------')
                elif selectedSkill=='图片理解'or selectedSkill=='Image2Txt' or selectedSkill=='圖片理解' :
                    logging.info('-------image_chat----')
                    for ans in image_chat(selectedSkill,dia, msg, True, **req):
                        ans = structure_answer(conv, ans, message_id, conv.id)
                        conv.message[-1] = {"role": "assistant", "content": ans["answer"],"id": message_id, "prompt": ans.get("prompt", ""),"selectedSkill":selectedSkill}
                        yield "data:"+json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
                    ConversationService.update_by_id(conv.id, conv.to_dict())    
                
                elif selectedSkill=='图片生成'or selectedSkill=='Txt2Image' or selectedSkill=='圖片生成' :
                    logging.info('-------txt-image_chat----')
                    for ans in txt_image_chat(dia, msg, True, **req):
                        ans = structure_answer(conv, ans, message_id, conv.id)
                        conv.message[-1] = {"role": "assistant", "content": ans["answer"],"id": message_id, "prompt": ans.get("prompt", ""),"selectedSkill":selectedSkill}
                        yield "data:"+json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
                    ConversationService.update_by_id(conv.id, conv.to_dict())
                    pass
                else:
                    logging.info('-------only_chat----')
                    for ans in only_chat(selectedSkill,dia, msg, True, **req):
                        ans = structure_answer(conv, ans, message_id, conv.id)
                        conv.message[-1] = {"role": "assistant", "content": ans["answer"],"id": message_id, "prompt": ans.get("prompt", ""),"selectedSkill":selectedSkill}
                        yield "data:"+json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
                    ConversationService.update_by_id(conv.id, conv.to_dict())            
            except Exception as e:
                traceback.print_exc()
                yield "data:" + json.dumps({"code": 500, "message": str(e), "data": {"answer": "**ERROR**: " + str(e), "reference": []}}, ensure_ascii=False) + "\n\n"
            yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"
        if req.get("stream", True):
            resp = Response(stream(), mimetype="text/event-stream")
            resp.headers.add_header("Cache-control", "no-cache")
            resp.headers.add_header("Connection", "keep-alive")
            resp.headers.add_header("X-Accel-Buffering", "no")
            resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
            return resp

        else:
            answer = None
            for ans in chat(dia, msg, **req):
                answer = structure_answer(conv, ans, message_id, req["conversation_id"])
                conv.message[-1] = {"role": "assistant", "content": ans["answer"],"id": message_id, "prompt": ans.get("prompt", ""),"selectedSkill":selectedSkill}
                ConversationService.update_by_id(conv.id, conv.to_dict())
                break
            return get_json_result(data=answer)
    except Exception as e:
        return server_error_response(e)


@manager.route("/tts", methods=["POST"])  # noqa: F821
@login_required
def tts():
    req = request.json
    text = req["text"]

    tenants = TenantService.get_info_by(current_user.id)
    if not tenants:
        return get_data_error_result(message="Tenant not found!")

    tts_id = tenants[0]["tts_id"]
    if not tts_id:
        return get_data_error_result(message="No default TTS model is set")

    tts_mdl = LLMBundle(tenants[0]["tenant_id"], LLMType.TTS, tts_id)

    def stream_audio():
        try:
            for txt in re.split(r"[，。/《》？；：！\n\r:;]+", text):
                for chunk in tts_mdl.tts(txt):
                    yield chunk
        except Exception as e:
            yield ("data:" + json.dumps({"code": 500, "message": str(e), "data": {"answer": "**ERROR**: " + str(e)}}, ensure_ascii=False)).encode("utf-8")

    resp = Response(stream_audio(), mimetype="audio/mpeg")
    resp.headers.add_header("Cache-Control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")

    return resp


@manager.route("/delete_msg", methods=["POST"])  # noqa: F821
@login_required
@validate_request("conversation_id", "message_id")
def delete_msg():
    req = request.json
    e, conv = ConversationService.get_by_id(req["conversation_id"])
    if not e:
        return get_data_error_result(message="Conversation not found!")

    conv = conv.to_dict()
    for i, msg in enumerate(conv["message"]):
        if req["message_id"] != msg.get("id", ""):
            continue
        assert conv["message"][i + 1]["id"] == req["message_id"]
        conv["message"].pop(i)
        conv["message"].pop(i)
        conv["reference"].pop(max(0, i // 2 - 1))
        break

    ConversationService.update_by_id(conv["id"], conv)
    return get_json_result(data=conv)


@manager.route("/thumbup", methods=["POST"])  # noqa: F821
@login_required
@validate_request("conversation_id", "message_id")
def thumbup():
    req = request.json
    e, conv = ConversationService.get_by_id(req["conversation_id"])
    if not e:
        return get_data_error_result(message="Conversation not found!")
    up_down = req.get("thumbup")
    feedback = req.get("feedback", "")
    conv = conv.to_dict()
    for i, msg in enumerate(conv["message"]):
        if req["message_id"] == msg.get("id", "") and msg.get("role", "") == "assistant":
            if up_down:
                msg["thumbup"] = True
                if "feedback" in msg:
                    del msg["feedback"]
            else:
                msg["thumbup"] = False
                if feedback:
                    msg["feedback"] = feedback
            break

    ConversationService.update_by_id(conv["id"], conv)
    return get_json_result(data=conv)


@manager.route("/ask", methods=["POST"])  # noqa: F821
@login_required
@validate_request("question", "kb_ids")
def ask_about():
    req = request.json
    uid = current_user.id

    def stream():
        nonlocal req, uid
        try:
            for ans in ask(req["question"], req["kb_ids"], uid):
                yield "data:" + json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
        except Exception as e:
            yield "data:" + json.dumps({"code": 500, "message": str(e), "data": {"answer": "**ERROR**: " + str(e), "reference": []}}, ensure_ascii=False) + "\n\n"
        yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

    resp = Response(stream(), mimetype="text/event-stream")
    resp.headers.add_header("Cache-control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")
    resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
    return resp


@manager.route("/mindmap", methods=["POST"])  # noqa: F821
@login_required
@validate_request("question", "kb_ids")
def mindmap():
    req = request.json
    kb_ids = req["kb_ids"]
    e, kb = KnowledgebaseService.get_by_id(kb_ids[0])
    if not e:
        return get_data_error_result(message="Knowledgebase not found!")

    embd_mdl = LLMBundle(kb.tenant_id, LLMType.EMBEDDING, llm_name=kb.embd_id)
    chat_mdl = LLMBundle(current_user.id, LLMType.CHAT)
    question = req["question"]
    ranks = settings.retrievaler.retrieval(question, embd_mdl, kb.tenant_id, kb_ids, 1, 12, 0.3, 0.3, aggs=False, rank_feature=label_question(question, [kb]))
    mindmap = MindMapExtractor(chat_mdl)
    mind_map = trio.run(mindmap, [c["content_with_weight"] for c in ranks["chunks"]])
    mind_map = mind_map.output
    if "error" in mind_map:
        return server_error_response(Exception(mind_map["error"]))
    return get_json_result(data=mind_map)


@manager.route("/related_questions", methods=["POST"])  # noqa: F821
@login_required
@validate_request("question")
def related_questions():
    req = request.json
    question = req["question"]
    chat_mdl = LLMBundle(current_user.id, LLMType.CHAT)
    prompt = """
Role: You are an AI language model assistant tasked with generating 5-10 related questions based on a user’s original query. These questions should help expand the search query scope and improve search relevance.

Instructions:
	Input: You are provided with a user’s question.
	Output: Generate 5-10 alternative questions that are related to the original user question. These alternatives should help retrieve a broader range of relevant documents from a vector database.
	Context: Focus on rephrasing the original question in different ways, making sure the alternative questions are diverse but still connected to the topic of the original query. Do not create overly obscure, irrelevant, or unrelated questions.
	Fallback: If you cannot generate any relevant alternatives, do not return any questions.
	Guidance:
	1. Each alternative should be unique but still relevant to the original query.
	2. Keep the phrasing clear, concise, and easy to understand.
	3. Avoid overly technical jargon or specialized terms unless directly relevant.
	4. Ensure that each question contributes towards improving search results by broadening the search angle, not narrowing it.

Example:
Original Question: What are the benefits of electric vehicles?

Alternative Questions:
	1. How do electric vehicles impact the environment?
	2. What are the advantages of owning an electric car?
	3. What is the cost-effectiveness of electric vehicles?
	4. How do electric vehicles compare to traditional cars in terms of fuel efficiency?
	5. What are the environmental benefits of switching to electric cars?
	6. How do electric vehicles help reduce carbon emissions?
	7. Why are electric vehicles becoming more popular?
	8. What are the long-term savings of using electric vehicles?
	9. How do electric vehicles contribute to sustainability?
	10. What are the key benefits of electric vehicles for consumers?

Reason:
	Rephrasing the original query into multiple alternative questions helps the user explore different aspects of their search topic, improving the quality of search results.
	These questions guide the search engine to provide a more comprehensive set of relevant documents.
"""
    ans = chat_mdl.chat(
        prompt,
        [
            {
                "role": "user",
                "content": f"""
Keywords: {question}
Related search terms:
    """,
            }
        ],
        {"temperature": 0.9},
    )
    return get_json_result(data=[re.sub(r"^[0-9]\. ", "", a) for a in ans.split("\n") if re.match(r"^[0-9]\. ", a)])
