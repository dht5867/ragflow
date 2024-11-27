# 该文件封装了对api.py的请求，可以被不同的webui使用
# 通过ApiRequest和AsyncApiRequest支持同步/异步调用

from typing import *
from pathlib import Path

from api.settings import (
    CHUNK_SIZE,
    HTTPX_DEFAULT_TIMEOUT,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    api_address,
    cmdb_api_address,
    cmdb_chat_address,
    log_verbose,
    chat_logger,
)

import httpx
import contextlib
import json
import os
from io import BytesIO


httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = HTTPX_DEFAULT_TIMEOUT
httpx._config.DEFAULT_TIMEOUT_CONFIG.read = HTTPX_DEFAULT_TIMEOUT
httpx._config.DEFAULT_TIMEOUT_CONFIG.write = HTTPX_DEFAULT_TIMEOUT

def get_httpx_client(
    use_async: bool = False,
    proxies: Union[str, Dict] = None,
    timeout: float = HTTPX_DEFAULT_TIMEOUT,
    **kwargs,
) -> Union[httpx.Client, httpx.AsyncClient]:
    """
    helper to get httpx client with default proxies that bypass local addesses.
    """
    default_proxies = {
        # do not use proxy for locahost
        "all://127.0.0.1": None,
        "all://localhost": None,
    }

    # get proxies from system envionrent
    # proxy not str empty string, None, False, 0, [] or {}
    default_proxies.update(
        {
            "http://": (
                os.environ.get("http_proxy")
                if os.environ.get("http_proxy")
                and len(os.environ.get("http_proxy").strip())
                else None
            ),
            "https://": (
                os.environ.get("https_proxy")
                if os.environ.get("https_proxy")
                and len(os.environ.get("https_proxy").strip())
                else None
            ),
            "all://": (
                os.environ.get("all_proxy")
                if os.environ.get("all_proxy")
                and len(os.environ.get("all_proxy").strip())
                else None
            ),
        }
    )
    for host in os.environ.get("no_proxy", "").split(","):
        if host := host.strip():
            # default_proxies.update({host: None}) # Origin code
            default_proxies.update(
                {"all://" + host: None}
            )  # PR 1838 fix, if not add 'all://', httpx will raise error

    # merge default proxies with user provided proxies
    if isinstance(proxies, str):
        proxies = {"all://": proxies}

    if isinstance(proxies, dict):
        default_proxies.update(proxies)

    # construct Client
    kwargs.update(timeout=timeout, proxies=default_proxies)

    if use_async:
        return httpx.AsyncClient(**kwargs)
    else:
        return httpx.Client(**kwargs)


class ApiRequest:
    """
    api.py调用的封装（同步模式）,简化api调用方式
    """

    def __init__(
        self,
        base_url: str = api_address(),
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._use_async = False
        self._client = None

    @property
    def client(self):
        if self._client is None or self._client.is_closed:
            self._client = get_httpx_client(
                base_url=self.base_url, use_async=self._use_async, timeout=self.timeout
            )
        return self._client

    def get(
        self,
        url: str,
        params: Union[Dict, List[Tuple], bytes] = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream("GET", url, params=params, **kwargs)
                else:
                    return self.client.get(url, params=params, **kwargs)
            except Exception as e:
                msg = f"error when get {url}: {e}"
                chat_logger.error(
                    f"{e.__class__.__name__}: {msg}",
                    exc_info=e if log_verbose else None,
                )
                retry -= 1

    def post(
        self,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                # print(kwargs)
                if stream:
                    return self.client.stream(
                        "POST", url, data=data, json=json, **kwargs
                    )
                else:
                    return self.client.post(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when post {url}: {e}"
                chat_logger.error(
                    f"{e.__class__.__name__}: {msg}",
                    exc_info=e if log_verbose else None,
                )
                retry -= 1

    def delete(
        self,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream(
                        "DELETE", url, data=data, json=json, **kwargs
                    )
                else:
                    return self.client.delete(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when delete {url}: {e}"
                chat_logger.error(
                    f"{e.__class__.__name__}: {msg}",
                    exc_info=e if log_verbose else None,
                )
                retry -= 1

    def _httpx_stream2generator(
        self,
        response: contextlib._GeneratorContextManager,
        as_json: bool = False,
    ):
        """
        将httpx.stream返回的GeneratorContextManager转化为普通生成器
        """

        async def ret_async(response, as_json):
            try:
                async with response as r:
                    async for chunk in r.aiter_text(None):
                        if not chunk:  # fastchat api yield empty bytes on start and end
                            continue
                        if as_json:
                            try:
                                if chunk.startswith("data: "):
                                    data = json.loads(chunk[6:-2])
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    data = json.loads(chunk)
                                yield data
                            except Exception as e:
                                msg = f"接口返回json错误： ‘{chunk}’。错误信息是：{e}。"
                                chat_logger.error(
                                    f"{e.__class__.__name__}: {msg}",
                                    exc_info=e if log_verbose else None,
                                )
                        else:
                            # print(chunk, end="", flush=True)
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"无法连接API服务器，请确认 ‘api.py’ 已正常启动。({e})"
                chat_logger.error(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API通信超时，请确认已启动FastChat与API服务（详见Wiki '5. 启动 API 服务或 Web UI'）。（{e}）"
                chat_logger.error(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API通信遇到错误：{e}"
                chat_logger.error(
                    f"{e.__class__.__name__}: {msg}",
                    exc_info=e if log_verbose else None,
                )
                yield {"code": 500, "msg": msg}

        def ret_sync(response, as_json):
            try:
                with response as r:
                    for chunk in r.iter_text(None):
                        if not chunk:  # fastchat api yield empty bytes on start and end
                            continue
                        if as_json:
                            try:
                                if chunk.startswith("data: "):
                                    data = json.loads(chunk[6:-2])
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    data = json.loads(chunk)
                                yield data
                            except Exception as e:
                                msg = f"接口返回json错误： ‘{chunk}’。错误信息是：{e}。"
                                chat_logger.error(
                                    f"{e.__class__.__name__}: {msg}",
                                    exc_info=e if log_verbose else None,
                                )
                        else:
                            # print(chunk, end="", flush=True)
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"无法连接API服务器，请确认 ‘api.py’ 已正常启动。({e})"
                chat_logger.error(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API通信超时，请确认已启动FastChat与API服务（详见Wiki '5. 启动 API 服务或 Web UI'）。（{e}）"
                chat_logger.error(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API通信遇到错误：{e}"
                chat_logger.error(
                    f"{e.__class__.__name__}: {msg}",
                    exc_info=e if log_verbose else None,
                )
                yield {"code": 500, "msg": msg}

        if self._use_async:
            return ret_async(response, as_json)
        else:
            return ret_sync(response, as_json)

    def _get_response_value(
        self,
        response: httpx.Response,
        as_json: bool = False,
        value_func: Callable = None,
    ):
        """
        转换同步或异步请求返回的响应
        `as_json`: 返回json
        `value_func`: 用户可以自定义返回值，该函数接受response或json
        """

        def to_json(r):
            try:
                return r.json()
            except Exception as e:
                msg = "API未能返回正确的JSON。" + str(e)
                if log_verbose:
                    chat_logger.error(
                        f"{e.__class__.__name__}: {msg}",
                        exc_info=e if log_verbose else None,
                    )
                return {"code": 500, "msg": msg, "data": None}

        if value_func is None:
            value_func = lambda r: r

        async def ret_async(response):
            if as_json:
                return value_func(to_json(await response))
            else:
                return value_func(await response)

        if self._use_async:
            return ret_async(response)
        else:
            if as_json:
                return value_func(to_json(response))
            else:
                return value_func(response)

    def cmdb_chat(self, input_txt: str):
        """
        对应webui_pages/dialogue/dialogue_cmdb.py接口
        """
        data = {"input": {"input": input_txt}, "config": {}, "kwargs": {}}
        cmdb_address = cmdb_api_address()

        response = self.post(
            cmdb_address + "/invoke",
            json=data,
        )

        return self._get_response_value(response, as_json=True)

    def cmdb_chat_stream(self, input_txt: str):
        """
        对应webui_pages/dialogue/dialogue_cmdb.py接口
        """
        data = {"input": {"input": input_txt}, "config": {}, "kwargs": {}}
        cmdb_address = cmdb_api_address()
        response = self.post(cmdb_address + "/stream", json=data, stream=True)
        return self._httpx_stream2generator(response, as_json=False)

    def cmdb_chat_chinese(self, input_txt: str):
        """
        对应webui_pages/dialogue/dialogue_cmdb.py接口
        """
        data = {
            "model": "glm4",
            "messages": [{"role": "user", "content": input_txt}],
            "temperature": 0.7,
        }
        cmdb_address = cmdb_chat_address()
        response = self.post(
            cmdb_address + "/v1/chat/completions",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def upload_temp_docs(
        self,
        files: Any,
        knowledge_id: str = None,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE,
        zh_title_enhance=ZH_TITLE_ENHANCE,
    ):
        """
        对应webui_pages/dialogue/dialogue_cmdb.py接口
        """
        api_url = api_address()
       
        data = {
            "knowledge_id": knowledge_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
        }

        response = self.post(
            api_url+"/knowledge_base/upload_temp_docs",
            data=data,
            files=files,
        )
        return self._get_response_value(response, as_json=True)

    def file_chat(
        self,
        query: str,
        knowledge_id: str,
        history: List[Dict],
        stream: bool,
        model: str,
        max_tokens: int = None,
        prompt_name: str = "default",
    ):
        """
        对应api.py/chat/file_chat接口
        """
        data = {
            "query": query,
            "knowledge_id": knowledge_id,
            "history": history,
            "stream": stream,
            "model_name": model,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }
        chat_logger.info('----file_chat---')
        chat_logger.info(data)
        api_url = api_address()
        response = self.post(
            api_url + "/chat/file_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)


class AsyncApiRequest(ApiRequest):
    def __init__(
        self, base_url: str = api_address(), timeout: float = HTTPX_DEFAULT_TIMEOUT
    ):
        super().__init__(base_url, timeout)
        self._use_async = True


def check_error_msg(data: Union[str, dict, list], key: str = "errorMsg") -> str:
    """
    return error message if error occured when requests API
    """
    if isinstance(data, dict):
        if key in data:
            return data[key]
        if "code" in data and data["code"] != 200:
            return data["msg"]
    return ""


def check_success_msg(data: Union[str, dict, list], key: str = "msg") -> str:
    """
    return error message if error occured when requests API
    """
    if (
        isinstance(data, dict)
        and key in data
        and "code" in data
        and data["code"] == 200
    ):
        return data[key]
    return ""


if __name__ == "__main__":
    api = ApiRequest()
    aapi = AsyncApiRequest()
