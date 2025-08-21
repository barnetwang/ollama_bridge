import base64
import requests

class BaseAdapter:
    name = "base"

    def __init__(self, request_json):
        self.request_json = request_json

    def parse(self):
        raise NotImplementedError

    def get_final_stream_endpoint(self):
        return "/api/chat"

class LobeChatAdapter(BaseAdapter):
    name = "lobe_chat"

    def parse(self):
        user_prompt = ""
        image_base64 = ""
        messages = self.request_json.get("messages", [])

        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    user_prompt = content
                    break

        for message in messages:
            if message.get("role") == "user":
                if "images" in message:
                    images_list = message.get("images", [])
                    if images_list and isinstance(images_list, list) and isinstance(images_list[0], str):
                        image_data = images_list[0]
                        if "," in image_data:
                            image_base64 = image_data.split(',', 1)[-1]
                        else:
                            image_base64 = image_data
                        break
        
        return user_prompt, image_base64

class CherryStudioAdapter(BaseAdapter):
    name = "cherry_studio"

    def parse(self):
        user_prompt = ""
        image_base64 = ""
        messages = self.request_json.get("messages", [])

        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    user_prompt = content
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            user_prompt = part.get("text", "")
                        elif part.get("type") == "image_url":
                            image_url_data = part.get("image_url", {}).get("url", "")
                            if "," in image_url_data:
                                image_base64 = image_url_data.split(',', 1)[-1]
              
                if user_prompt:
                    break
        
        return user_prompt, image_base64

    def get_final_stream_endpoint(self):
        #OpenAI-compatible clients expect the /v1/chat/completions endpoint
        return "/v1/chat/completions"

# --- Adapter Registry ---
# This dictionary maps API path keywords to their corresponding adapter classes.
# The proxy will use this to select the correct adapter for an incoming request.
# Default is set to Cherry Studio and LobeHub
#請修改此字典將 API 路徑關鍵字對應到其相應的轉接器類別
#請修改代理將使用此字典為傳入的請求選擇正確的轉接器
#預設只支援Cherry Studio和LobeHub
ADAPTERS = {
    "api/chat": LobeChatAdapter,
    "v1/chat/completions": CherryStudioAdapter
}
