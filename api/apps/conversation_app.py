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
import re
import time
import traceback
from copy import deepcopy
from typing import Union
from api.db.services.cmdb_service import cmdb_chat_stream
from api.db.services.user_service import UserTenantService
from flask import request, Response
from flask_login import login_required, current_user

from api.db import LLMType
from api.db.services.dialog_service import DialogService, ConversationService, chat, ask, file_chat, only_chat
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle, TenantService, TenantLLMService
from api import settings
from api.utils.api_utils import get_json_result
from api.utils.api_utils import server_error_response, get_data_error_result, validate_request
from graphrag.mind_map_extractor import MindMapExtractor
from api.settings import chat_logger


@manager.route('/set', methods=['POST'])  # noqa: F821
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
                return get_data_error_result(
                    message="Fail to update a conversation!")
            conv = conv.to_dict()
            return get_json_result(data=conv)
        except Exception as e:
            return server_error_response(e)

    try:
        e, dia = DialogService.get_by_id(req["dialog_id"])
        if not e:
            return get_data_error_result(message="Dialog not found")
        conv = {
            "id": conv_id,
            "dialog_id": req["dialog_id"],
            "name": req.get("name", "New conversation"),
            "message": [{"role": "assistant", "content": dia.prompt_config["prologue"]}]
        }
        ConversationService.save(**conv)
        e, conv = ConversationService.get_by_id(conv["id"])
        if not e:
            return get_data_error_result(message="Fail to new a conversation!")
        conv = conv.to_dict()
        return get_json_result(data=conv)
    except Exception as e:
        return server_error_response(e)


@manager.route('/get', methods=['GET'])  # noqa: F821
@login_required
def get():
    conv_id = request.args["conversation_id"]
    try:
        e, conv = ConversationService.get_by_id(conv_id)
        if not e:
            return get_data_error_result(message="Conversation not found!")
        tenants = UserTenantService.query(user_id=current_user.id)
        for tenant in tenants:
            if DialogService.query(tenant_id=tenant.tenant_id, id=conv.dialog_id):
                break
        else:
            return get_json_result(
                data=False, message='Only owner of conversation authorized for this operation.',
                code=settings.RetCode.OPERATING_ERROR)
        conv = conv.to_dict()
        return get_json_result(data=conv)
    except Exception as e:
        return server_error_response(e)


@manager.route('/rm', methods=['POST'])  # noqa: F821
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
                return get_json_result(
                    data=False, message='Only owner of conversation authorized for this operation.',
                    code=settings.RetCode.OPERATING_ERROR)
            ConversationService.delete_by_id(cid)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/list', methods=['GET'])  # noqa: F821
@login_required
def list_convsersation():
    dialog_id = request.args["dialog_id"]
    try:
        if not DialogService.query(tenant_id=current_user.id, id=dialog_id):
            return get_json_result(
                data=False, message='Only owner of dialog authorized for this operation.',
                code=settings.RetCode.OPERATING_ERROR)
        convs = ConversationService.query(
            dialog_id=dialog_id,
            order_by=ConversationService.model.create_time,
            reverse=True)
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
 
 
@manager.route('/completion', methods=['POST'])
@login_required
@validate_request("conversation_id", "messages")
def completion():
    req = request.json
    msg = []
    chat_logger.info('-------completion----')
    chat_logger.info(req)
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
        conv.message.append({"role": "assistant", "content": "", "id": message_id})
        conv.reference.append({"chunks": [], "doc_aggs": []})

        def fillin_conv(ans):
            nonlocal conv, message_id
            if not conv.reference:
                conv.reference.append(ans["reference"])
            else:
                conv.reference[-1] = ans["reference"]
            conv.message[-1] = {"role": "assistant", "content": ans["answer"],"id": message_id, "prompt": ans.get("prompt", ""),"selectedSkill":selectedSkill}
            ans["id"] = message_id

        def stream():
            nonlocal dia, msg, req, conv
            try:
                if selectedSkill=='知识库' or selectedSkill=='KNOWLEDGE':
                    for ans in chat(dia, msg, True, **req):
                        fillin_conv(ans)
                        yield "data:" + json.dumps({"retcode": 0, "retmsg": "", "data": ans}, ensure_ascii=False) + "\n\n"
                    ConversationService.update_by_id(conv.id, conv.to_dict())
                elif selectedSkill=='日志分析' or selectedSkill=='LOG' :
                    chat_logger.info('-------日志分析----')
                    for ans in file_chat(dia, msg, True, **req):
                        fillin_conv(ans)
                        yield "data:"+json.dumps({"retcode": 0, "retmsg": "", "data": ans}, ensure_ascii=False) + "\n\n"
                    ConversationService.update_by_id(conv.id, conv.to_dict())
                elif selectedSkill=='CMDB':
                    chat_logger.info('--------prompt----------')
                    chat_logger.info(prompt)
                    answer = ""
                    result = cmdb_chat_stream(prompt)
                    chat_logger.info('---------CMDB start----------')
                    chat_logger.info(result)
                    for line in result:
                            if error_msg := check_error_msg(line):  # check whether error occured
                                chat_logger.error(error_msg)
                                answer=" 后台服务错误，请重试查询"
                                ans = decorate_answer(answer)
                                fillin_conv(ans)
                                yield "data:"+json.dumps({"retcode": 0, "retmsg": "", "data": ans}, ensure_ascii=False) + "\n\n"
                                ConversationService.update_by_id(conv.id, conv.to_dict())
                                break
                            lines = line.split("\n")
                            if len(lines)>1:
                                time.sleep(0.5)
                                chunk=lines[1]
                                if len(chunk)>2:
                                    chat_logger.info(chunk)
                                    chat_logger.info("chunk-------------------")
                                    if "data:" in chunk:
                                        json_data=json.loads(chunk[6:]) 
                                        chat_logger.info(json_data)
                                        
                                    # 检查并打印run_id
                                        if "run_id" in json_data:
                                            message_id= json_data['run_id']
                                            chat_logger.info(f"cmdb message_id: {message_id}")
                                        if "steps" in json_data:
                                            answer +=json_data['steps'][0]['action']['log']
                                            ans = decorate_answer(answer)
                                            fillin_conv(ans)
                                            yield "data:"+json.dumps({"retcode": 0, "retmsg": "", "data": ans}, ensure_ascii=False) + "\n\n"
                                            
                                         #将output 翻译为中文
                                        if "output" in json_data:
                                            answer +=json_data['output']
                                            ans = decorate_answer(answer)
                                            fillin_conv(ans)
                                            yield "data:"+json.dumps({"retcode": 0, "retmsg": "", "data": ans}, ensure_ascii=False) + "\n\n"
                                            # output= json_data['output']
                                            # chat_logger.info(output)
                                            # rr = cmdb_chat_chinese(
                                            #     f"请翻译 {output}",
                                            # )
                                            # chat_logger.info(rr)
                                            # ch_text=rr["choices"][0]["message"]["content"]
                                            
                    ConversationService.update_by_id(conv.id, conv.to_dict())
                    chat_logger.info('---------CMDB end----------')
                else:
                    chat_logger.info('-------chat----')
                    for ans in only_chat(selectedSkill,dia, msg, True, **req):
                        yield "data:"+json.dumps({"retcode": 0, "retmsg": "", "data": ans}, ensure_ascii=False) + "\n\n"
                        fillin_conv(ans)
                    ConversationService.update_by_id(conv.id, conv.to_dict())            
            except Exception as e:
                traceback.print_exc()
                yield "data:" + json.dumps({"code": 500, "message": str(e),
                                            "data": {"answer": "**ERROR**: " + str(e), "reference": []}},
                                           ensure_ascii=False) + "\n\n"
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
                answer = ans
                fillin_conv(ans)
                ConversationService.update_by_id(conv.id, conv.to_dict())
                break
            return get_json_result(data=answer)
    except Exception as e:
        return server_error_response(e)


@manager.route('/tts', methods=['POST'])  # noqa: F821
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
            yield ("data:" + json.dumps({"code": 500, "message": str(e),
                                         "data": {"answer": "**ERROR**: " + str(e)}},
                                        ensure_ascii=False)).encode('utf-8')

    resp = Response(stream_audio(), mimetype="audio/mpeg")
    resp.headers.add_header("Cache-Control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")

    return resp


@manager.route('/delete_msg', methods=['POST'])  # noqa: F821
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


@manager.route('/thumbup', methods=['POST'])  # noqa: F821
@login_required
@validate_request("conversation_id", "message_id")
def thumbup():
    req = request.json
    e, conv = ConversationService.get_by_id(req["conversation_id"])
    if not e:
        return get_data_error_result(message="Conversation not found!")
    up_down = req.get("set")
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


@manager.route('/ask', methods=['POST'])  # noqa: F821
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
            yield "data:" + json.dumps({"code": 500, "message": str(e),
                                        "data": {"answer": "**ERROR**: " + str(e), "reference": []}},
                                       ensure_ascii=False) + "\n\n"
        yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

    resp = Response(stream(), mimetype="text/event-stream")
    resp.headers.add_header("Cache-control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")
    resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
    return resp


@manager.route('/mindmap', methods=['POST'])  # noqa: F821
@login_required
@validate_request("question", "kb_ids")
def mindmap():
    req = request.json
    kb_ids = req["kb_ids"]
    e, kb = KnowledgebaseService.get_by_id(kb_ids[0])
    if not e:
        return get_data_error_result(message="Knowledgebase not found!")

    embd_mdl = TenantLLMService.model_instance(
        kb.tenant_id, LLMType.EMBEDDING.value, llm_name=kb.embd_id)
    chat_mdl = LLMBundle(current_user.id, LLMType.CHAT)
    ranks = settings.retrievaler.retrieval(req["question"], embd_mdl, kb.tenant_id, kb_ids, 1, 12,
                                           0.3, 0.3, aggs=False)
    mindmap = MindMapExtractor(chat_mdl)
    mind_map = mindmap([c["content_with_weight"] for c in ranks["chunks"]]).output
    if "error" in mind_map:
        return server_error_response(Exception(mind_map["error"]))
    return get_json_result(data=mind_map)


@manager.route('/related_questions', methods=['POST'])  # noqa: F821
@login_required
@validate_request("question")
def related_questions():
    req = request.json
    question = req["question"]
    chat_mdl = LLMBundle(current_user.id, LLMType.CHAT)
    prompt = """
Objective: To generate search terms related to the user's search keywords, helping users find more valuable information.
Instructions:
 - Based on the keywords provided by the user, generate 5-10 related search terms.
 - Each search term should be directly or indirectly related to the keyword, guiding the user to find more valuable information.
 - Use common, general terms as much as possible, avoiding obscure words or technical jargon.
 - Keep the term length between 2-4 words, concise and clear.
 - DO NOT translate, use the language of the original keywords.

### Example:
Keywords: Chinese football
Related search terms:
1. Current status of Chinese football
2. Reform of Chinese football
3. Youth training of Chinese football
4. Chinese football in the Asian Cup
5. Chinese football in the World Cup

Reason:
 - When searching, users often only use one or two keywords, making it difficult to fully express their information needs.
 - Generating related search terms can help users dig deeper into relevant information and improve search efficiency. 
 - At the same time, related terms can also help search engines better understand user needs and return more accurate search results.
 
"""
    ans = chat_mdl.chat(prompt, [{"role": "user", "content": f"""
Keywords: {question}
Related search terms:
    """}], {"temperature": 0.9})
    return get_json_result(data=[re.sub(r"^[0-9]\. ", "", a) for a in ans.split("\n") if re.match(r"^[0-9]\. ", a)])
