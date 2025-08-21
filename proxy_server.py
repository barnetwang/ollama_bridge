from flask import Flask, request, Response
import requests
import json
import base64

from adapters import ADAPTERS

# --- Basic settings ---
app = Flask(__name__)
OLLAMA_BASE_URL = "http://localhost:11434"
THINKING_MODEL = "gpt-oss:20b" #這邊請設定您的思考模型 | Please set up your thinking model here.
VISION_MODEL = "gemma3:4b"     #這邊請設定您的視覺模型 | Please set up your visual model here.

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
    for chunk in response.iter_content(chunk_size=None):
        yield chunk

def handle_vision_request(adapter, user_prompt, image_base64):
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
            "content": f"You are a helpful assistant. The user has provided an image with the following content: '{image_description}'. Now, answer the user's question based on this information."
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
    print(f"\n--- 接收到請求: {subpath} ---")
    print(json.dumps(client_request_json, indent=2, ensure_ascii=False))

    adapter_class = None
    for path_key, adapter in ADAPTERS.items():
        if path_key in subpath:
            adapter_class = adapter
            break

    if not adapter_class:
        return Response(json.dumps({"error": f"Unsupported API path: {subpath}"}), status=404)
    
    adapter = adapter_class(client_request_json)
    print(f"==> 使用適配器: {adapter.name}")

    try:
        user_prompt, image_base64 = adapter.parse()
        if not user_prompt:
             return Response(json.dumps({"error": "Could not parse user prompt from request."}))
    except Exception as e:
        print(f"!! Adapter parsing failed: {e}")
        return Response(json.dumps({"error": f"Adapter parsing failed: {e}"}), status=400)

    print("\n==> [決策] 正在請求思考模型進行決策...")
    decision = "yes" if image_base64 else "no"
    print(f"--> 基於圖片存在與否的決策是: '{decision}'")

    if "yes" in decision and image_base64:
        return handle_vision_request(adapter, user_prompt, image_base64)
    else:
        print("==> [執行] 進入純文字轉發流程...")
        final_endpoint = adapter.get_final_stream_endpoint()
        target_url = f"{OLLAMA_BASE_URL}{final_endpoint}"
        
        forward_payload = client_request_json.copy()
        forward_payload['model'] = THINKING_MODEL
        
        print(f"-> 轉發到 {target_url} (強制使用模型: {THINKING_MODEL})")
        try:
            ollama_response = requests.post(target_url, json=forward_payload, stream=True)
            ollama_response.raise_for_status()
            return Response(stream_forwarder(ollama_response), status=ollama_response.status_code, content_type=ollama_response.headers.get('content-type'))
        except requests.exceptions.RequestException as e:
            error_info = {"error": {"name": "ResponseError", "message": f"Error forwarding: {e}", "status_code": 502}, "provider": "ollama"}
            return Response(json.dumps(error_info), status=502, content_type='application/json')


if __name__ == '__main__':
    print("="*60); print("  Universal Adapter Proxy Started"); print("  Listening on: http://localhost:5000"); print("="*60)
    app.run(host='0.0.0.0', port=5000)
