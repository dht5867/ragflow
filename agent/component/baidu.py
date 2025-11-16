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
import logging
from abc import ABC
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
from agent.component.base import ComponentBase, ComponentParamBase


class BaiduParam(ComponentParamBase):
    """
    Define the Baidu component parameters.
    """

    def __init__(self):
        super().__init__()
        self.top_n = 10

    def check(self):
        self.check_positive_integer(self.top_n, "Top N")


class Baidu(ComponentBase, ABC):
    component_name = "Baidu"

    def _run(self, history, **kwargs):
        ans = self.get_input()
        ans = " - ".join(ans["content"]) if "content" in ans else ""
        if not ans:
            return Baidu.be_output("")
        logging.info(f"baidu --- ans: {ans}")
        results = []
        try:
            logging.info(f"ask  : {ans}")
            logging.info(f"top_n: {str(self._param.top_n)}")
            url = 'http://www.baidu.com/s?wd=' + ans + '&rn=' + str(self._param.top_n)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36'}
            response = requests.get(url=url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
           
            # 百度结果标题一般在 <h3 class="t">
            for item in soup.select("div.result, div.result-op"):
                 # --- 标题 ---
                title_tag = item.select_one("h3.t")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                # 链接在 <a> 标签
                a_tag = item.find("a")
                href = a_tag["href"] if a_tag else None
                # --- 内容简介 contentText ---
                # 常见情况：<div class="c-abstract">
                content_tag = item.select_one("div.c-abstract")
                # 新版可能：<div class="c-line-clamp3">
                if content_tag is None:
                    content_tag = item.select_one("div.c-line-clamp3")
                # 取文本
                content = content_tag.get_text(" ", strip=True) if content_tag else ""
                results.append({
                    "title": title,
                    "url": href,
                    "contentText": content
                })
            # url_res = re.findall(r"'url': \\\"(.*?)\\\"}", response.text)
            # title_res = re.findall(r"'title': \\\"(.*?)\\\",\\n", response.text)
            # body_res = re.findall(r"\"contentText\":\"(.*?)\"", response.text)
            # baidu_res = [{"content": re.sub('<em>|</em>', '', '<a href="' + url + '">' + title + '</a>    ' + body)} for
            #              url, title, body in zip(url_res, title_res, body_res)]
            # logging.info(f"baidu --- : {str(baidu_res)}")
            #del body_res, url_res, title_res
        except Exception as e:
            return Baidu.be_output("**ERROR**: " + str(e))

        if not results:
            return Baidu.be_output("baidu --- 没有搜到结果")

        df = pd.DataFrame(results)
        logging.debug(f"df: {str(df)}")
        return df

