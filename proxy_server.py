import os
from flask import Flask, request, Response
import requests
import json
import base64
import threading
from datetime import datetime, date

from adapters import find_adapter
from googleapiclient.discovery import build
from dotenv import load_dotenv
import trafilatura

load_dotenv()
app = Flask(__name__)

USAGE_FILE = "usage.json"
api_usage_lock = threading.RLock()


def load_api_usage():
    with api_usage_lock:
        if not os.path.exists(USAGE_FILE):
            usage_data = {
                "google_search": {
                    "count": 0,
                    "daily_limit": 100,
                    "reset_date": datetime.now().date()
                }
            }
            save_api_usage(usage_data)
            return usage_data
        try:
            with open(USAGE_FILE, 'r', encoding='utf-8') as f:
                usage_data = json.load(f)
                usage_data["google_search"]["reset_date"] = datetime.fromisoformat(
                    usage_data["google_search"]["reset_date"]).date()
                return usage_data
        except (json.JSONDecodeError, KeyError):
            print(f"!! 警告: {USAGE_FILE} 文件格式错误，将重新创建。")
            os.remove(USAGE_FILE)
            return load_api_usage()


def save_api_usage(usage_data):
    with api_usage_lock:
        data_to_save = json.loads(json.dumps(usage_data, default=str))
        reset_date_val = usage_data["google_search"]["reset_date"]
        if isinstance(reset_date_val, date):
            data_to_save["google_search"]["reset_date"] = reset_date_val.isoformat()
        with open(USAGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)


API_USAGE = load_api_usage()

OLLAMA_BASE_URL = "http://localhost:11434"
THINKING_MODEL = "gpt-oss:20b"
VISION_MODEL = "gemma3:4b"

def load_prompts_from_directory(directory: str) -> dict:
    prompts = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(base_dir, directory)
    if not os.path.isdir(prompts_dir):
        print(f"!! 警告: Prompt 目錄不存在: {prompts_dir}")
        return {}
    for filename in os.listdir(prompts_dir):
        if filename.endswith(".txt"):
            prompt_name = os.path.splitext(filename)[0]
            with open(os.path.join(prompts_dir, filename), 'r', encoding='utf-8') as f:
                prompts[prompt_name] = f.read().strip()
    print(f"-> 成功從 '{directory}' 目錄載入 {len(prompts)} 個專家角色。")
    return prompts


EXPERT_PROMPTS = load_prompts_from_directory("prompts")
if "Assistant" not in EXPERT_PROMPTS:
    EXPERT_PROMPTS["Assistant"] = "You are a helpful AI assistant."


def generate_search_context(search_results: list[dict], question: str) -> str:
    print(f"-> [上下文生成] 正在處理 {len(search_results)} 條搜尋結果...")
    if not search_results:
        return ""

    final_context = "--- CONTEXTUAL SOURCES ---\n"

    for i, result in enumerate(search_results):
        final_context += f"[Source {i+1}]\n"
        final_context += f"Title: {result['title']}\n"
        final_context += f"URL: {result['link']}\n"
        final_context += f"Content Snippet: {result['snippet']}\n\n"

    deep_browse_content = ""
    for i, result in enumerate(search_results[:3]):
        url = result.get('link')
        title = result.get('title', 'N/A')
        if not url:
            continue

        print(f"--> [深度瀏覽 {i+1}/3] 正在嘗試連結: {url}")
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                print(f"    - 警告: 無法從 {url} 下載內容。")
                continue

            page_main_text = trafilatura.extract(
                downloaded, include_comments=False, include_tables=True)
            if page_main_text and len(page_main_text) > 100:
                summary_prompt = (
                    f"Please read the main content from the webpage '{title}' and extract ONLY the key points that are most relevant to the user's question: '{question}'.\n\n"
                    f"--- WEBPAGE MAIN CONTENT ---\n{page_main_text[:4000]}\n\n"
                    f"--- RELEVANT KEY POINTS SUMMARY ---"
                )
                summary_messages = [
                    {"role": "user", "content": summary_prompt}]
                summary_response = call_llm(summary_messages, stream=False)
                if summary_response:
                    summary = summary_response.json().get("message", {}).get("content", "")
                    if summary:
                        deep_browse_content += f"[Deep Dive Summary for Source {i+1}: {title}]\n{summary}\n\n"
                        print(f"    - 成功對 {url} 進行深度總結。")
            else:
                print(f"    - 警告: 未能從 {url} 提取到有效的主要內容。")
        except Exception as e:
            print(f"!! [深度瀏覽 {i+1}/3] 失敗: {url}, 原因: {e}")
            continue

    if deep_browse_content:
        final_context += "--- DEEP DIVE SUMMARIES ---\n" + deep_browse_content

    return final_context


def create_fused_prompt_with_weights(selected_experts: list) -> str:
    if not selected_experts:
        return EXPERT_PROMPTS.get("Assistant", "You are a helpful AI assistant.")
    if len(selected_experts) == 1:
        expert_name = selected_experts[0][0]
        return EXPERT_PROMPTS.get(expert_name, EXPERT_PROMPTS["Assistant"])
    weight_map = {"High": 0, "Medium": 1, "Low": 2}
    sorted_experts = sorted(
        selected_experts, key=lambda x: weight_map.get(x[1], 99))
    fused_prompt = "You are a top-tier AI advisory team... Your final answer's primary tone and focus should align with the 'High' influence expert, incorporating knowledge from others as supporting details.\n\n"
    for expert_name, weight in sorted_experts:
        fused_prompt += f"### Expert: {expert_name} (Influence: {weight})\n{EXPERT_PROMPTS.get(expert_name, '')}\n\n"
    return fused_prompt

@app.after_request
def after_request_func(response):
    origin = request.headers.get('Origin')
    if origin:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Allow-Headers',
                             'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods',
                             'GET,PUT,POST,DELETE,OPTIONS')
    return response

def stream_forwarder(response):
    for chunk in response.iter_content(chunk_size=None):
        yield chunk

def call_llm(messages: list, stream: bool = False):
    payload = {"model": THINKING_MODEL, "messages": messages, "stream": stream}
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=stream, timeout=180)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"!! 內部 LLM 調用失敗: {e}")
        return None

def generate_search_query(original_question: str) -> str:
    print(f"--> [查詢優化] 正在將問題轉換為搜尋關鍵字...")
    prompt = (f"You are a search engine optimization expert... convert the following user's question into a concise, keyword-based search query...\n\n"
              f"### USER QUESTION ###\n\"{original_question}\"\n\n### OPTIMIZED SEARCH QUERY ###")
    payload = {"model": THINKING_MODEL, "prompt": prompt,
               "stream": False, "options": {"temperature": 0.0}}
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=45)
        response.raise_for_status()
        optimized_query = response.json().get(
            "response", original_question).strip().replace("\"", "")
        print(f"--> [查詢優化] 原始問題: '{original_question}'")
        print(f"--> [查詢優化] 優化後查詢: '{optimized_query}'")
        return optimized_query
    except requests.exceptions.RequestException as e:
        print(f"!! [查詢優化] 失敗: {e}. 將使用原始問題進行搜尋。")
        return original_question

def perform_google_search(query: str, max_results: int = 3) -> list[dict]:
    print(f"-> [工具調用] 正在執行 Google 網路搜尋: '{query}'")
    with api_usage_lock:
        today = datetime.now().date()
        if API_USAGE["google_search"]["reset_date"] != today:
            API_USAGE["google_search"]["count"] = 0
            API_USAGE["google_search"]["reset_date"] = today
        if API_USAGE["google_search"]["count"] >= API_USAGE["google_search"]["daily_limit"]:
            print("!! 警告: Google Search API 每日免費額度已用盡。")
            return []
        API_USAGE["google_search"]["count"] += 1
        save_api_usage(API_USAGE)
        print(
            f"-> Google Search API 使用次數: {API_USAGE['google_search']['count']}/{API_USAGE['google_search']['daily_limit']}")
    api_key = os.environ.get('GOOGLE_API_KEY')
    search_engine_id = os.environ.get('GOOGLE_CSE_ID')
    if not api_key or not search_engine_id:
        print("!! 錯誤: GOOGLE_API_KEY 或 GOOGLE_CSE_ID 環境變數未設定。")
        return []

    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=search_engine_id, num=max_results).execute()
        items = res.get('items', [])
        if not items:
            print("-> Google 網路搜尋沒有找到相關結果。")
            return []

        structured_results = [{
            "title": item.get('title', 'N/A'),
            "link": item.get('link', 'N/A'),
            "snippet": item.get('snippet', 'N/A')
        } for item in items]

        print(f"-> Google 搜尋成功，找到 {len(structured_results)} 條結果。")
        return structured_results
    except Exception as e:
        print(f"!! Google 網路搜尋失敗: {e}")
        return []

def generate_search_context(search_results: list[dict], question: str) -> str:
    if not search_results:
        return ""
    final_context = "--- CONTEXTUAL SOURCES ---\n"
    for i, result in enumerate(search_results):
        final_context += f"[Source {i+1}]\nTitle: {result['title']}\nURL: {result['link']}\nContent Snippet: {result['snippet']}\n\n"
    deep_browse_content = ""
    for i, result in enumerate(search_results[:3]):
        url = result.get('link')
        if not url:
            continue
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                continue
            page_main_text = trafilatura.extract(
                downloaded, include_comments=False, include_tables=True)
            if page_main_text and len(page_main_text) > 100:
                summary_prompt = (f"Please read the main content from '{result['title']}' and extract ONLY the key points relevant to: '{question}'.\n\n"
                                  f"--- WEBPAGE MAIN CONTENT ---\n{page_main_text[:4000]}\n\n--- RELEVANT KEY POINTS SUMMARY ---")
                summary_response = call_llm(
                    [{"role": "user", "content": summary_prompt}], stream=False)
                if summary_response:
                    summary = summary_response.json().get("message", {}).get("content", "")
                    if summary:
                        deep_browse_content += f"[Deep Dive Summary for Source {i+1}: {result['title']}]\n{summary}\n\n"
        except Exception:
            continue
    if deep_browse_content:
        final_context += "--- DEEP DIVE SUMMARIES ---\n" + deep_browse_content
    return final_context

def is_context_relevant(context: str, original_question: str) -> bool:
    if not context:
        return False
    check_prompt = (f"You are a fact-checker... determine if the provided CONTEXT contains enough information to directly answer the USER'S QUESTION... Answer ONLY with 'Yes' or 'No'.\n\n"
                    f"--- USER'S QUESTION ---\n{original_question}\n\n--- CONTEXT ---\n{context}\n\n--- VERDICT (Yes/No) ---")
    payload = {"model": THINKING_MODEL, "prompt": check_prompt,
               "stream": False, "options": {"temperature": 0.0, "top_p": 0.1}}
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=45)
        response.raise_for_status()
        verdict = response.json().get("response", "No").strip().lower()
        return "yes" in verdict
    except requests.exceptions.RequestException:
        return False

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
        vision_response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=vision_payload, timeout=300)
        vision_response.raise_for_status()
        image_description = vision_response.json().get(
            "response", "Could not get a description.")
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
    thinking_payload = {"model": THINKING_MODEL,
                        "messages": new_messages, "stream": True}
    final_endpoint = adapter.get_final_stream_endpoint()
    try:
        thinking_response = requests.post(
            f"{OLLAMA_BASE_URL}{final_endpoint}", json=thinking_payload, stream=True)
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
        target_url = f"{OLLAMA_BASE_URL}/{subpath}"
        try:
            resp = requests.request(method=request.method, url=target_url, headers={k: v for (
                k, v) in request.headers if k.lower() != 'host'}, data=request.get_data(), params=request.args, stream=True)
            return Response(stream_forwarder(resp), status=resp.status_code, content_type=resp.headers.get('content-type'))
        except requests.exceptions.RequestException as e:
            return Response(f"Error forwarding: {e}", status=502)

    client_request_json = request.get_json()
    if not client_request_json.get("stream", False):
        target_url = f"{OLLAMA_BASE_URL}/{subpath}"
        try:
            resp = requests.post(target_url, headers={k: v for (
                k, v) in request.headers if k.lower() != 'host'}, json=client_request_json)
            return Response(resp.content, status=resp.status_code, content_type=resp.headers.get('content-type'))
        except requests.exceptions.RequestException as e:
            return Response(f"Error forwarding: {e}", status=502)

    print(f"\n--- [STEP 1] 接收到【流式】請求: {subpath} ---")
    adapter_class = find_adapter(subpath)
    if not adapter_class:
        return Response(json.dumps({"error": f"Unsupported API path: {subpath}"}), status=404)

    adapter = find_adapter(subpath)(client_request_json)
    print(f"==> [STEP 2] 使用適配器: {adapter.name}")
    try:
        user_prompt, core_question, image_base64 = adapter.parse()
    except Exception as e:
        return Response(json.dumps({"error": f"Adapter parsing failed: {e}"}), status=400)

    print(f"--- [STEP 3] 初始解析出的核心問題: '{core_question[:100]}...' ---")
    SEARCH_PREFIX = "@網路搜尋"
    search_context = None
    original_question = core_question
    search_results = []

    if core_question.strip().startswith(SEARCH_PREFIX):
        original_question = core_question.strip().split(
            '\n')[0].replace(SEARCH_PREFIX, "", 1).strip()
        if original_question:
            search_query = generate_search_query(original_question)
            search_results = perform_google_search(search_query, max_results=5)
            if search_results:
                search_context = generate_search_context(
                    search_results, original_question)
            image_base64 = None

    print("\n==> [STEP 4] 請求思考模型進行角色選擇與權重分配...")
    expert_list = list(EXPERT_PROMPTS.keys())
    context_preview = ""
    if search_results:
        context_preview += "### AVAILABLE INFORMATION PREVIEW ###\n"
        for i, result in enumerate(search_results[:3]):
            context_preview += f"- Title {i+1}: {result.get('title', 'N/A')}\n  Snippet: {result.get('snippet', 'N/A')}\n"

    decision_prompt = (f"You are a Chief of Staff... assign a team of experts... based on the user's request and the available information.\n\n"
                       f"### Expert List ###\n{expert_list}\n\n{context_preview}\n"
                       f"### Decision Principles ###\n1. Assign MINIMUM experts...\n2. Collaboration Rule: Assign multiple experts ONLY IF the request CLEARLY blends skills...\n\n"
                       f"User Request: \"{original_question}\"\n\nYour output MUST BE a comma-separated list of 'Expert (Influence)' pairs.")

    selected_experts_with_weights = [("Assistant", "High")]
    try:
        decision_payload = {"model": THINKING_MODEL,
                            "prompt": decision_prompt, "stream": False}
        decision_response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=decision_payload, timeout=60)
        decision_response.raise_for_status()
        response_text = decision_response.json().get("response", "").strip()
        parts = [p.strip() for p in response_text.split(',')]
        parsed_experts = []
        for part in parts:
            if '(' in part and ')' in part:
                name = part.split('(')[0].strip()
                influence = part.split('(')[1].replace(')', '').strip()
                if name in EXPERT_PROMPTS:
                    parsed_experts.append((name, influence))
        if parsed_experts:
            selected_experts_with_weights = parsed_experts
        print(
            f"--- [STEP 5] 模型決策: 選擇專家團隊 -> {selected_experts_with_weights} ---")
    except requests.exceptions.RequestException:
        pass
    print(f"--- [STEP 5] 模型決策: 選擇專家團隊 -> {selected_experts_with_weights} ---")

    if image_base64 and not search_context:
        return handle_vision_request(adapter, user_prompt, image_base64, create_fused_prompt_with_weights(selected_experts_with_weights))

    weight_map = {"High": 0, "Medium": 1, "Low": 2}
    sorted_experts = sorted(selected_experts_with_weights,
                            key=lambda x: weight_map.get(x[1], 99))
    lead_expert_name, _ = sorted_experts[0]
    lead_expert_prompt = EXPERT_PROMPTS.get(lead_expert_name, "")

    consultant_prompts = ""
    if len(sorted_experts) > 1:
        consultant_prompts += "\n### CONSULTING EXPERTS' PERSPECTIVES ###\nYou must incorporate the perspectives of:\n"
        for name, influence in sorted_experts[1:]:
            consultant_prompts += f"- **{name} (Influence: {influence})**: {EXPERT_PROMPTS.get(name, '')}\n"

    base_system_prompt = lead_expert_prompt + consultant_prompts
    final_messages = []

    if search_context and original_question:
        if is_context_relevant(search_context, original_question):
            citation_instruction = (
                "**CRITICAL INSTRUCTIONS (You must follow BOTH):**\n"
                "1.  **In-line Citations:** You MUST cite the source of your information at the end of each relevant sentence using the format `[Source X]`. For multiple sources, use `[Source 1, 3]`.\n"
                "2.  **Final Reference List:** At the VERY END of your entire response, you MUST include a section titled `References` or `資料來源`. Under this title, list every source you cited, mapping the source number to its **full title and its corresponding URL**. The format MUST be exactly as follows:\n"
                "    *   [Source 1] - Title of the first article (URL: the_full_url_here)\n"
                "    *   [Source 2] - Title of the second article (URL: the_full_url_here)"
            )
            final_system_prompt = f"{base_system_prompt}\n\n{citation_instruction}\n\n{search_context}"
            final_messages.append(
                {"role": "system", "content": final_system_prompt})
            final_messages.append(
                {"role": "user", "content": original_question})
        else:
            apology_text = "我進行了網路搜尋，但找到的資料似乎與您提出的問題關聯性不高..."

            def generate_apology_stream():
                if "v1/chat/completions" in adapter.get_final_stream_endpoint():
                    yield "data: [DONE]\n\n"
                else:
                    yield f"{json.dumps({'message': {'content': apology_text}, 'done': True})}\n"
            return Response(generate_apology_stream(), content_type='application/x-ndjson')
    else:
        final_messages.append(
            {"role": "system", "content": base_system_prompt})
        original_messages = client_request_json.get("messages", [])
        final_messages.extend([msg for msg in original_messages if msg.get(
            "role") not in ["system", "developer"]])

    forward_payload = client_request_json.copy()
    forward_payload['messages'] = final_messages
    forward_payload['model'] = THINKING_MODEL
    try:
        ollama_response = requests.post(
            f"{OLLAMA_BASE_URL}{adapter.get_final_stream_endpoint()}", json=forward_payload, stream=True)
        ollama_response.raise_for_status()
        return Response(stream_forwarder(ollama_response), status=ollama_response.status_code, content_type=ollama_response.headers.get('content-type'))
    except requests.exceptions.RequestException as e:
        return Response(json.dumps({"error": f"Error forwarding: {e}", "status_code": 502}), status=502, content_type='application/json')

def create_fused_prompt_with_weights(selected_experts: list) -> str:
    if not selected_experts:
        return EXPERT_PROMPTS.get("Assistant", "")
    sorted_experts = sorted(selected_experts, key=lambda x: {
                            "High": 0, "Medium": 1, "Low": 2}.get(x[1], 99))
    lead_expert_name, _ = sorted_experts[0]
    prompt = EXPERT_PROMPTS.get(lead_expert_name, "")
    if len(sorted_experts) > 1:
        prompt += "\nIncorporate perspectives from:\n"
        for name, influence in sorted_experts[1:]:
            prompt += f"- {name} ({influence})\n"
    return prompt

if __name__ == '__main__':
    print("="*60)
    print("  Universal Adapter Proxy Started - Final Unified Version")
    print("  Listening on: http://localhost:5000")
    print("="*60)
    app.run(host='0.0.0.0', port=5000)
