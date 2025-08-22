import os
from flask import Flask, request, Response
import requests
import json
import base64

from adapters import find_adapter

# --- Basic settings ---
app = Flask(__name__)
OLLAMA_BASE_URL = "http://localhost:11434"
THINKING_MODEL = "gpt-oss:20b" #這邊請設定您的思考模型 | Please set up your thinking model here.
VISION_MODEL = "gemma3:4b"     #這邊請設定您的視覺模型 | Please set up your visual model here.

def load_prompts_from_directory(directory: str) -> dict:
    prompts = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(base_dir, directory)

    if not os.path.isdir(prompts_dir):
        print(f"!! 警告: Prompt 目錄不存在: {prompts_dir}")
        return {}

    for filename in os.listdir(prompts_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(prompts_dir, filename)
            prompt_name = os.path.splitext(filename)[0]
            with open(filepath, 'r', encoding='utf-8') as f:
                prompts[prompt_name] = f.read().strip()
    
    print(f"-> 成功從 '{directory}' 目錄載入 {len(prompts)} 個專家角色。")
    return prompts

EXPERT_PROMPTS = load_prompts_from_directory("prompts")
if "通用助手" not in EXPERT_PROMPTS:
    EXPERT_PROMPTS["通用助手"] = "你是一個樂於助人的人工智能助手。"

@app.after_request
def after_request_func(response):
    origin = request.headers.get('Origin')
    if origin:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def stream_forwarder(response):
    # debug用
    print("\n" + "="*20 + " 開始攔截 OLLAMA 的回應流 " + "="*20)
    chunk_count = 0
    is_empty = True
    for chunk in response.iter_content(chunk_size=None):
        is_empty = False
        chunk_count += 1
    #    print(f"--- CHUNK {chunk_count} --> {chunk.decode('utf-8', errors='ignore')}")
        yield chunk
    
    if is_empty:
        print("--- 警告: OLLAMA 的回應流是空的！(沒有任何數據) ---")
    
    print("="*22 + f" 攔截結束 (共 {chunk_count} 個數據) " + "="*22 + "\n")

def handle_vision_request(adapter, user_prompt, image_base64, expert_system_prompt):
    print("\n==> [執行] 進入圖文處理流程...")
    print(f"--> 步驟 2.1: 調用視覺模型 ({VISION_MODEL}) 進行描述...")
    vision_payload = {
        "model": VISION_MODEL,
        "prompt": "Describe this image in detail from an objective perspective.",
        "images": [image_base64],
        "stream": False
    }
    try:
        vision_response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=vision_payload, timeout=300)
        vision_response.raise_for_status()
        image_description = vision_response.json().get("response", "Could not get a description.")
        print(f"--> 步驟 2.1 成功. 描述: {image_description[:100]}...")
    except requests.exceptions.RequestException as e:
        print(f"!! 步驟 2.1 失敗. 調用視覺模型出錯: {e}")
        return Response(json.dumps({"error": str(e)}), status=502)

    new_messages = [
        {
            "role": "system",
            "content": f"{expert_system_prompt}\n\n附註：使用者同時提供了一張圖片，其客觀內容描述如下：'{image_description}'。請結合這些信息回答用戶的問題。"
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    thinking_payload = {
        "model": THINKING_MODEL,
        "messages": new_messages,
        "stream": True
    }

    final_endpoint = adapter.get_final_stream_endpoint()
    print(f"--> 步驟 2.2: 調用思考模型 ({THINKING_MODEL}) 於端點: {final_endpoint}")
    
    try:
        thinking_response = requests.post(f"{OLLAMA_BASE_URL}{final_endpoint}", json=thinking_payload, stream=True)
        thinking_response.raise_for_status()
        print("--> 步驟 2.2 成功. 將回應流式傳輸到客戶端。")
        return Response(stream_forwarder(thinking_response), status=thinking_response.status_code, content_type=thinking_response.headers.get('content-type'))
    except requests.exceptions.RequestException as e:
        print(f"!! 步驟 2.2 失敗. 調用思考模型出錯: {e}")
        return Response(json.dumps({"error": str(e)}), status=502)

@app.route('/<path:subpath>', methods=['POST', 'OPTIONS'])
def intelligent_proxy(subpath):
    if request.method == 'OPTIONS':
        return Response(status=200)

    if not (request.method == 'POST' and ("v1/chat/completions" in subpath or "api/chat" in subpath)):
        print(f"\n==> 進入通用轉發器處理 {request.method} /{subpath}...")
        target_url = f"{OLLAMA_BASE_URL}/{subpath}"
        try:
            ollama_response = requests.request(
                method=request.method, url=target_url,
                headers={k: v for (k, v) in request.headers if k.lower() != 'host'},
                data=request.get_data(), params=request.args, stream=True
            )
            return Response(stream_forwarder(ollama_response), status=ollama_response.status_code, content_type=ollama_response.headers.get('content-type'))
        except requests.exceptions.RequestException as e:
            return Response(f"Error forwarding: {e}", status=502)

    client_request_json = request.get_json()
    
    is_stream_request = client_request_json.get("stream", False)
    if not is_stream_request:
        print(f"\n==> 檢測到非流式請求，進入通用轉發器...")

        target_url = f"{OLLAMA_BASE_URL}/{subpath}"
        try:
            ollama_response = requests.request(
                method=request.method, url=target_url,
                headers={k: v for (k, v) in request.headers if k.lower() != 'host'},
                data=request.get_data(), params=request.args, stream=False # 注意 stream=False
            )
            return Response(ollama_response.content, status=ollama_response.status_code, content_type=ollama_response.headers.get('content-type'))
        except requests.exceptions.RequestException as e:
            return Response(f"Error forwarding: {e}", status=502)

    print(f"\n--- [STEP 1] 接收到【流式】請求: {subpath} ---")
    
    adapter_class = find_adapter(subpath)
    if not adapter_class:
        return Response(json.dumps({"error": f"Unsupported API path: {subpath}"}), status=404)
    
    adapter = adapter_class(client_request_json)
    print(f"--- [STEP 2] 使用適配器: {adapter.name} ---")

    try:
        user_prompt, core_question, image_base64 = adapter.parse()
        # <--- debug 用 ---
        #print("\n--- [DEBUG] ADAPTER PARSE RESULT ---")
        #print(f"  - user_prompt (長度): {len(user_prompt)}")
        #print(f"  - core_question (長度): {len(core_question)}")
        #print(f"  - image_base64 (長度): {len(image_base64) if image_base64 else 0}")
        #print("---------------------------------")
        if not user_prompt:
             return Response(json.dumps({"error": "Adapter parsing returned empty user_prompt."}, status=400))
    except Exception as e:
        return Response(json.dumps({"error": f"Adapter parsing failed: {e}"}), status=400)

    print(f"--- [STEP 3] 解析出的核心問題: '{core_question[:100]}...' ---")

    print("\n==> [STEP 4] 請求思考模型進行角色選擇...")
    expert_list = list(EXPERT_PROMPTS.keys())
    decision_prompt = (
        f"You are a strict and efficient Chief of Staff. Your task is to assign the following user request to the MINIMUM necessary number of experts from the provided list.\n\n"
        f"### Expert List ###\n{expert_list}\n\n"
        f"### Decision Principles ###\n"
        f"1.  **Single Expert Rule (Default)**: For requests that fall clearly into one domain (e.g., asking for a definition like 'What is X?', asking for a creative piece like 'Write a poem', or asking for medical advice), assign ONLY the single most relevant expert.\n"
        f"2.  **Collaboration Rule (Exception)**: Assign multiple experts ONLY IF the request EXPLICITLY and CLEARLY blends skills from different domains. For example, a request to 'write a short story about Da Vinci surgery' clearly requires both medical accuracy ('Doctor') and narrative skill ('Writer').\n"
        f"3.  **Fallback Rule**: If the request is a simple greeting or is too ambiguous to determine a specific domain, assign the 'Assistant'.\n\n"
        f"### Your Task ###\n"
        f"Analyze the following user request based on the principles above.\n\n"
        f"User Request: \"{core_question}\"\n\n"
        f"Your output MUST BE ONLY the comma-separated list of the chosen expert names."
    )
    decision_payload = {"model": THINKING_MODEL, "prompt": decision_prompt, "stream": False}
    
    selected_expert = "Assistant"
    try:
        decision_response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=decision_payload, timeout=60)
        decision_response.raise_for_status()
        
        response_text = decision_response.json().get("response", "").strip()
        potential_experts = [expert.strip() for expert in response_text.split(',')]
        
        valid_experts = [expert for expert in potential_experts if expert in EXPERT_PROMPTS]
        if valid_experts:
            selected_experts = valid_experts

        print(f"--- [STEP 5] 模型決策: 選擇專家團隊 -> {selected_experts} ---")
    except requests.exceptions.RequestException as e:
        print(f"!! 決策失敗: {e}. 將使用預設專家。")
    
    final_system_prompt = create_fused_prompt(selected_experts)

    if image_base64:
        print(f"--> 執行路徑: 圖文處理 (使用團隊: {selected_experts})。")
        return handle_vision_request(adapter, user_prompt, image_base64, final_system_prompt)
    else:
        print(f"--> 執行路徑: 純文字處理 (使用團隊: {selected_experts})。")
        final_endpoint = adapter.get_final_stream_endpoint()
        target_url = f"{OLLAMA_BASE_URL}{final_endpoint}"

        final_messages = [{"role": "system", "content": final_system_prompt}]

        original_messages = client_request_json.get("messages", [])
        for msg in original_messages:
            if msg.get("role") not in ["system", "developer"]:
                final_messages.append(msg)

        forward_payload = client_request_json.copy()
        forward_payload['messages'] = final_messages
        forward_payload['model'] = THINKING_MODEL
        
        print(f"-> 轉發到 {target_url} (強制使用模型: {THINKING_MODEL})")
        try:
            ollama_response = requests.post(target_url, json=forward_payload, stream=True)
            ollama_response.raise_for_status()
            return Response(stream_forwarder(ollama_response), status=ollama_response.status_code, content_type=ollama_response.headers.get('content-type'))
        except requests.exceptions.RequestException as e:
            error_info = {"error": {"name": "ResponseError", "message": f"Error forwarding: {e}", "status_code": 502}}
            return Response(json.dumps(error_info), status=502, content_type='application/json')

def create_fused_prompt(selected_experts: list) -> str:
    if not selected_experts or len(selected_experts) == 0:
        return EXPERT_PROMPTS.get("Assistant", "You are a helpful AI assistant.")
    
    if len(selected_experts) == 1:
        expert_name = selected_experts[0]
        return EXPERT_PROMPTS.get(expert_name, EXPERT_PROMPTS["Assistant"])

    fused_prompt = (
        "You are a top-tier AI advisory team composed of multiple experts. For this response, you must embody the combined capabilities of the following experts:\n\n"
    )
    
    for i, expert_name in enumerate(selected_experts):
        expert_instruction = EXPERT_PROMPTS.get(expert_name, "")
        fused_prompt += f"### Expert {i+1}: {expert_name}\n"
        fused_prompt += f"{expert_instruction}\n\n"
        
    fused_prompt += "Please provide a comprehensive and professional response that integrates the perspectives of all the above experts."
    return fused_prompt

if __name__ == '__main__':
    print("="*60); print("  Universal Adapter Proxy Started"); print("  Listening on: http://localhost:5000"); print("="*60)
    app.run(host='0.0.0.0', port=5000)
