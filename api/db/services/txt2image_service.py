import logging
import openai



def txt_image(
        query: str,
        model: str="stable-diffusion-v1.5",
        lang: str="zh"
    ):
    try:
        # Assume that the model is already launched.
        # The api_key can't be empty, any string is OK.
        client = openai.Client(api_key="not empty", base_url="http://10.171.248.164:9997/v1")
        response=client.images.generate(model=model, prompt=query,response_format="b64_json")
        if response:
            return response.data[0].b64_json
    except Exception as e:
        logging.error(f"file_chat: {e}")
        return None

