
import logging
import requests

headers = {
                "X-QW-Api-Key": f"5ea2b337a6eb46699591dd67e0e94695",
                # 或者使用其他认证方式，根据 API 文档要求
                # "X-API-Key": self._param.web_apikey,
            }
response = requests.get(url="https://m77fc26jmp.re.qweatherapi.com/geo/v2/city/lookup?location=" + '北京',headers=headers).json()
if response["code"] == "200":
    location_id = response["location"][0]["id"]
    print(location_id)
    logging.info(f"[QWeather] Location ID: {location_id}")
else:
    logging.error("Error: %s", response["message"])