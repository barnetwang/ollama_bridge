from adapters import find_adapter
import os
import logging
import json
import base64
import threading
from datetime import datetime, date
from flask import Flask, request, Response
import requests
from googleapiclient.discovery import build
from dotenv import load_dotenv
import trafilatura

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)

USAGE_FILE = "usage.json"
api_usage_lock = threading.RLock()


def load_api_usage():
    with api_usage_lock:
        if not os.path.exists(USAGE_FILE):
            usage_data = {"google_search": {
                "count": 0, "daily_limit": 100, "reset_date": datetime.now().date()}}
            save_api_usage(usage_data)
            return usage_data
        try:
            with open(USAGE_FILE, 'r', encoding='utf-8') as f:
                usage_data = json.load(f)
                usage_data["google_search"]["reset_date"] = datetime.fromisoformat(
                    usage_data["google_search"]["reset_date"]).date()
                return usage_data
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"{USAGE_FILE} 文件格式錯誤，將重新創建。")
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

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
THINKING_MODEL = os.getenv("THINKING_MODEL", "gpt-oss:20b")
VISION_MODEL = os.getenv("VISION_MODEL", "gemma3:4b")


def load_prompts_from_directory(directory: str) -> dict:
    prompts = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(base_dir, directory)
    if not os.path.isdir(prompts_dir):
        logger.warning(f"Prompt 目錄不存在: {prompts_dir}")
        return {}
    for filename in os.listdir(prompts_dir):
        if filename.endswith(".txt"):
            prompt_name = os.path.splitext(filename)[0]
            with open(os.path.join(prompts_dir, filename), 'r', encoding='utf-8') as f:
                prompts[prompt_name] = f.read().strip()
    logger.info(f"成功從 '{directory}' 目錄載入 {len(prompts)} 個專家角色。")
    return prompts


EXPERT_PROMPTS = load_prompts_from_directory("prompts")
if "Assistant" not in EXPERT_PROMPTS:
    EXPERT_PROMPTS["Assistant"] = "You are a helpful AI assistant."


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


def create_error_response(message: str, error_type: str = "api_error", status_code: int = 500) -> Response:
    logger.error(f"生成錯誤回應 (HTTP {status_code}): {message}")
    error_payload = {
        "error": {
            "type": error_type,
            "message": message,
            "status_code": status_code
        }
    }
    return Response(json.dumps(error_payload), status=status_code, mimetype='application/json')


def call_llm(messages: list, stream: bool = False):
    payload = {"model": THINKING_MODEL, "messages": messages, "stream": stream}
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=stream, timeout=180)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"內部 LLM 調用失敗: {e}", exc_info=True)
        return None


def generate_search_query(original_question: str) -> str:
    logger.info("正在將問題轉換為搜尋關鍵字...")
    prompt = (f"You are a search engine optimization expert... convert the user's question into a concise, keyword-based search query...\n\n"
              f"### USER QUESTION ###\n\"{original_question}\"\n\n### OPTIMIZED SEARCH QUERY ###")
    payload = {"model": THINKING_MODEL, "prompt": prompt,
               "stream": False, "options": {"temperature": 0.0}}
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=45)
        response.raise_for_status()
        optimized_query = response.json().get(
            "response", original_question).strip().replace("\"", "")
        logger.info(
            f"原始問題: '{original_question}' -> 優化後查詢: '{optimized_query}'")
        return optimized_query
    except requests.exceptions.RequestException as e:
        logger.error(f"查詢優化失敗: {e}. 將使用原始問題進行搜尋。", exc_info=True)
        return original_question


def perform_google_search(query: str, max_results: int = 5) -> list[dict]:
    logger.info(f"正在執行 Google 網路搜尋: '{query}'")
    with api_usage_lock:
        today = datetime.now().date()
        if API_USAGE["google_search"]["reset_date"] != today:
            API_USAGE["google_search"]["count"] = 0
            API_USAGE["google_search"]["reset_date"] = today
        if API_USAGE["google_search"]["count"] >= API_USAGE["google_search"]["daily_limit"]:
            logger.warning("Google Search API 每日免費額度已用盡。")
            return []
        API_USAGE["google_search"]["count"] += 1
        save_api_usage(API_USAGE)
        logger.info(
            f"Google Search API 使用次數: {API_USAGE['google_search']['count']}/{API_USAGE['google_search']['daily_limit']}")

    api_key = os.getenv('GOOGLE_API_KEY')
    search_engine_id = os.getenv('GOOGLE_CSE_ID')
    if not api_key or not search_engine_id:
        logger.error("GOOGLE_API_KEY 或 GOOGLE_CSE_ID 環境變數未設定。")
        return []

    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=search_engine_id, num=max_results).execute()
        items = res.get('items', [])
        if not items:
            logger.info("Google 網路搜尋沒有找到相關結果。")
            return []
        structured_results = [{"title": item.get('title', 'N/A'), "link": item.get(
            'link', 'N/A'), "snippet": item.get('snippet', 'N/A')} for item in items]
        logger.info(f"Google 搜尋成功，找到 {len(structured_results)} 條結果。")
        return structured_results
    except Exception as e:
        logger.error(f"Google 網路搜尋失敗: {e}", exc_info=True)
        return []


def generate_search_context(search_results: list[dict], question: str) -> str:
    logger.info(f"正在處理 {len(search_results)} 條搜尋結果以生成上下文...")
    if not search_results:
        return ""
    final_context = "--- CONTEXTUAL SOURCES ---\n"
    for i, result in enumerate(search_results):
        final_context += f"[Source {i+1}]\nTitle: {result['title']}\nURL: {result['link']}\nContent Snippet: {result['snippet']}\n\n"

    deep_browse_content = ""
    for i, result in enumerate(search_results[:3]):
        url = result.get('link')
        title = result.get('title', 'N/A')
        if not url:
            continue
        logger.info(f"深度瀏覽 {i+1}/3: 正在嘗試連結: {url}")
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                logger.warning(f"無法從 {url} 下載內容。")
                continue
            page_main_text = trafilatura.extract(
                downloaded, include_comments=False, include_tables=True)
            if page_main_text and len(page_main_text) > 100:
                summary_prompt = (f"Please read the main content from '{title}' and extract ONLY the key points relevant to: '{question}'.\n\n"
                                  f"--- WEBPAGE MAIN CONTENT ---\n{page_main_text[:4000]}\n\n--- RELEVANT KEY POINTS SUMMARY ---")
                summary_response = call_llm(
                    [{"role": "user", "content": summary_prompt}], stream=False)
                if summary_response:
                    summary = summary_response.json().get("message", {}).get("content", "")
                    if summary:
                        deep_browse_content += f"[Deep Dive Summary for Source {i+1}: {title}]\n{summary}\n\n"
                        logger.info(f"成功對 {url} 進行深度總結。")
            else:
                logger.warning(f"未能從 {url} 提取到有效的主要內容。")
        except Exception as e:
            logger.error(f"深度瀏覽 {i+1}/3 失敗: {url}, 原因: {e}", exc_info=False)
            continue
    if deep_browse_content:
        final_context += "--- DEEP DIVE SUMMARIES ---\n" + deep_browse_content
    return final_context


def is_context_relevant(context: str, original_question: str) -> bool:
    if not context:
        return False
    logger.info("正在評估上下文的有效性...")
    check_prompt = (f"You are a fact-checker... determine if the provided CONTEXT contains enough information to directly answer the USER'S QUESTION... Answer ONLY with 'Yes' or 'No'.\n\n"
                    f"--- USER'S QUESTION ---\n{original_question}\n\n--- CONTEXT ---\n{context}\n\n--- VERDICT (Yes/No) ---")
    payload = {"model": THINKING_MODEL, "prompt": check_prompt,
               "stream": False, "options": {"temperature": 0.0, "top_p": 0.1}}
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=45)
        response.raise_for_status()
        verdict = response.json().get("response", "No").strip().lower()
        logger.info(f"關聯性檢查判斷結果: {verdict}")
        return "yes" in verdict
    except requests.exceptions.RequestException as e:
        logger.error(f"關聯性檢查失敗: {e}", exc_info=True)
        return False


def handle_vision_request(adapter, user_prompt, image_base64, final_system_prompt):
    logger.info("進入圖文處理流程...")
    vision_payload = {"model": VISION_MODEL, "prompt": "Describe this image in detail.", "images": [
        image_base64], "stream": False}
    try:
        vision_response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=vision_payload, timeout=300)
        vision_response.raise_for_status()
        image_description = vision_response.json().get(
            "response", "Could not get a description.")
    except requests.exceptions.RequestException as e:
        return create_error_response(f"調用視覺模型出錯: {e}", "vision_model_error", 502)

    new_messages = [{"role": "system", "content": f"{final_system_prompt}\n\nImage Description: '{image_description}'."}, {
        "role": "user", "content": user_prompt}]
    thinking_payload = {"model": THINKING_MODEL,
                        "messages": new_messages, "stream": True}
    try:
        thinking_response = requests.post(
            f"{OLLAMA_BASE_URL}{adapter.get_final_stream_endpoint()}", json=thinking_payload, stream=True)
        thinking_response.raise_for_status()
        logger.info("將視覺模型描述與問題傳遞給思考模型，並流式傳輸回應。")
        return Response(stream_forwarder(thinking_response), status=thinking_response.status_code, content_type=thinking_response.headers.get('content-type'))
    except requests.exceptions.RequestException as e:
        return create_error_response(f"調用思考模型出錯: {e}", "thinking_model_error", 502)


@app.route('/<path:subpath>', methods=['POST', 'OPTIONS'])
def intelligent_proxy(subpath):
    if request.method == 'OPTIONS':
        return Response(status=200)

    if not (request.method == 'POST' and ("v1/chat/completions" in subpath or "api/chat" in subpath)):
        logger.info(f"進入通用轉發器處理 {request.method} /{subpath}...")
        target_url = f"{OLLAMA_BASE_URL}/{subpath}"
        try:
            resp = requests.request(method=request.method, url=target_url, headers={k: v for (
                k, v) in request.headers if k.lower() != 'host'}, data=request.get_data(), params=request.args, stream=True)
            return Response(stream_forwarder(resp), status=resp.status_code, content_type=resp.headers.get('content-type'))
        except requests.exceptions.RequestException as e:
            return create_error_response(f"通用轉發失敗: {e}", "forwarding_error", 502)

    client_request_json = request.get_json()
    if not client_request_json.get("stream", False):
        logger.info("檢測到非流式請求，進入通用轉發器...")
        target_url = f"{OLLAMA_BASE_URL}/{subpath}"
        try:
            resp = requests.post(target_url, headers={k: v for (
                k, v) in request.headers if k.lower() != 'host'}, json=client_request_json)
            return Response(resp.content, status=resp.status_code, content_type=resp.headers.get('content-type'))
        except requests.exceptions.RequestException as e:
            return create_error_response(f"非流式轉發失敗: {e}", "forwarding_error", 502)

    logger.info(f"--- [STEP 1] 接收到【流式】請求: {subpath} ---")
    adapter_class = find_adapter(subpath)
    if not adapter_class:
        return create_error_response(f"不支援的 API 路徑: {subpath}", "unsupported_path", 404)

    adapter = adapter_class(client_request_json)
    logger.info(f"==> [STEP 2] 使用適配器: {adapter.name}")
    try:
        user_prompt, core_question, image_base64 = adapter.parse()
    except Exception as e:
        return create_error_response(f"適配器解析失敗: {e}", "adapter_error", 400)

    logger.info(f"--- [STEP 3] 初始解析出的核心問題: '{core_question[:200]}...' ---")
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

    logger.info("==> [STEP 4] 請求思考模型進行角色選擇與權重分配...")
    expert_list = list(EXPERT_PROMPTS.keys())
    pre_selected_experts = set()
    expert_keywords = {
        'Writer': ['作家', 'Writer'],
        'UX_UI_Developer': ['使用者體驗', '介面開發者', 'UX/UI', 'UX_UI_Developer'],
        'Cyber_Security_Specialist': ['網路安全', 'Cyber_Security_Specialist'],
        'Legal_Advisor': ['法律顧問', 'Legal_Advisor'],
        'Relationship_Coach': ['關係教練', 'Relationship_Coach'],
        'Philosopher': ['哲學家', 'Philosopher'],
        'Doctor': ['醫師', '醫學專家', 'Doctor'],
        'Financial_Analyst': ['財務分析師', 'Financial_Analyst']
        # ... 可繼續添加其他專家的關鍵詞
    }
    scan_area = original_question[:256]
    for expert, keywords in expert_keywords.items():
        if any(keyword.lower() in original_question.lower() for keyword in keywords):
            pre_selected_experts.add(expert)

    logger.info(f"程式碼預選專家 (僅掃描前256字符): {list(pre_selected_experts)}")
    context_preview = ""
    if search_results:
        context_preview += "### AVAILABLE INFORMATION PREVIEW ###\n"
        for i, result in enumerate(search_results[:3]):
            context_preview += f"- Title {i+1}: {result.get('title', 'N/A')}\n  Snippet: {result.get('snippet', 'N/A')}\n"
    remaining_experts = [
        exp for exp in expert_list if exp not in pre_selected_experts]
    decision_prompt = (
        f"You are a strict and efficient Chief of Staff. Your task is to finalize an expert team.\n\n"
        f"**A pre-selection has already been made based on the user's explicit request. The following experts are MANDATORY for this task:**\n"
        f"{list(pre_selected_experts) if pre_selected_experts else 'None'}\n\n"
        f"**Your tasks are:**\n"
        f"1.  **Assign Influence Levels:** Assign an influence level (High, Medium, or Low) to all mandatory experts.\n"
        f"2.  **Select Additional Experts (if necessary):** Analyze the user's request to see if any OTHER experts are needed from the list below. Do NOT re-select the mandatory experts.\n"
        f"3.  **Combine and Finalize:** Create a single, final comma-separated list of all chosen experts and their influence levels.\n\n"
        f"### List of Additional Experts to Consider ###\n"
        f"{remaining_experts}\n\n"
        f"### User Request ###\n"
        f"\"{original_question}\"\n\n"
        f"Your output MUST BE a single, comma-separated list of 'Expert (Influence)' pairs, including both the mandatory and any additional experts you selected."
    )

    selected_experts_with_weights = []
    if pre_selected_experts:
        selected_experts_with_weights = [
            (expert, 'Medium') for expert in pre_selected_experts]
    try:
        decision_payload = {"model": THINKING_MODEL,
                            "prompt": decision_prompt, "stream": False}
        decision_response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=decision_payload, timeout=90)
        decision_response.raise_for_status()
        response_text = decision_response.json().get("response", "").strip()
        parts = [p.strip() for p in response_text.split(',')]
        parsed_experts = [(p.split('(')[0].strip(), p.split('(')[1].replace(')', '').strip(
        )) for p in parts if '(' in p and ')' in p and p.split('(')[0].strip() in expert_list]
        if parsed_experts:
            selected_experts_with_weights = parsed_experts
    except requests.exceptions.RequestException as e:
        logger.warning(f"AI 輔助決策失敗: {e}. 將僅使用預選專家。", exc_info=True)
        if not pre_selected_experts:
            selected_experts_with_weights = [("Assistant", "High")]

    logger.info(
        f"--- [STEP 5] 模型決策: 選擇專家團隊 -> {selected_experts_with_weights} ---")

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

    if image_base64 and not search_context:
        return handle_vision_request(adapter, user_prompt, image_base64, base_system_prompt)

    final_messages = []
    if search_context and original_question:
        if is_context_relevant(search_context, original_question):
            citation_instruction = (
                "**CRITICAL INSTRUCTIONS (You must follow BOTH):**\n"
                "1.  **In-line Citations:** ... `[Source X]`.\n"
                "2.  **Final Reference List:** ... `References` or `資料來源`... list every source you cited... format MUST be exactly as follows:\n"
                "    *   [Source 1] - Title of the first article (URL: the_full_url_here)"
            )
            final_system_prompt = f"{base_system_prompt}\n\n{citation_instruction}\n\n{search_context}"
            final_messages.append(
                {"role": "system", "content": final_system_prompt})
            final_messages.append(
                {"role": "user", "content": original_question})
        else:
            logger.warning("上下文關聯性檢查未通過，生成標準回覆。")
            apology_text = "我進行了網路搜尋，但找到的資料似乎與您提出的問題關聯性不高..."

            def generate_apology_stream():
                yield f"data: {json.dumps({'choices': [{'delta': {'content': apology_text}}]})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(generate_apology_stream(), content_type='text/event-stream')
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
        return create_error_response(f"最終請求轉發失敗: {e}", "forwarding_error", 502)


if __name__ == '__main__':
    logger.info("="*60)
    logger.info(
        "  Universal Adapter Proxy Started - Refactored & Compatible Version")
    logger.info(f"  Thinking Model: {THINKING_MODEL}")
    logger.info(f"  Vision Model: {VISION_MODEL}")
    logger.info("  Listening on: http://localhost:5000")
    logger.info("="*60)
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000)
    except ImportError:
        logger.warning(
            "Waitress not found. Falling back to Flask's development server.")
        app.run(host='0.0.0.0', port=5000)
