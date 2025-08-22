import base64
import requests
import json

class BaseAdapter:
    name = "base"
    def __init__(self, request_json):
        self.request_json = request_json

    def parse(self):
        raise NotImplementedError

    def get_final_stream_endpoint(self):
        return "/api/chat"

    def _extract_core_question(self, user_prompt: str) -> str:
        core_question = user_prompt
        if core_question.strip().startswith('[') and core_question.strip().endswith(']'):
            try:
                data = json.loads(core_question)
                if isinstance(data, list) and len(data) > 0 and data[0].get('role') == 'user':
                    core_question = data[0].get('mainText', user_prompt)
            except json.JSONDecodeError:
                pass
        question_keywords = ["My question is:", "我的問題是:"]
        for keyword in question_keywords:
            if keyword in core_question:
                core_question = core_question.split(keyword, 1)[-1].strip()
                break
        if not core_question:
            core_question = user_prompt
        return core_question

class LobeChatAdapter(BaseAdapter):
    name = "lobe_chat"
    def parse(self):
        user_prompt, image_base64 = "", ""
        messages = self.request_json.get("messages", [])
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str): user_prompt = content
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text": user_prompt = part.get("text", ""); break
                if user_prompt: break
        for message in messages:
            if message.get("role") == "user" and "images" in message:
                images_list = message.get("images", [])
                if images_list: image_base64 = images_list[0].split(',')[-1]
                break
        core_question = self._extract_core_question(user_prompt)
        return user_prompt, core_question, image_base64
    def get_final_stream_endpoint(self):
        return "/api/chat"

class CherryStudioAdapter(BaseAdapter):
    name = "cherry_studio"

    def parse(self):
        user_prompt, image_base64 = "", ""
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
                        if part.get("type") == "image_url":
                            image_base64 = part.get("image_url", {}).get("url", "").split(',')[-1]
                if user_prompt:
                    break
        
        core_question = self._extract_core_question(user_prompt)
        return user_prompt, core_question, image_base64

    def get_final_stream_endpoint(self):
        return "/v1/chat/completions"

# --- Adapter Registry ---
ADAPTER_REGISTRY = [
    ("api/chat", LobeChatAdapter),
    ("v1/chat/completions", CherryStudioAdapter)
]

def find_adapter(subpath: str):
    for identifier, adapter_class in ADAPTER_REGISTRY:
        if identifier in subpath:
            return adapter_class
    return None