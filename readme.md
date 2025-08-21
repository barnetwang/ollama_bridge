# AI 代理伺服器

這是一個基於 Python 和 Flask 的智慧代理伺服器，專為大型語言模型（LLM）後端（如 Ollama）設計。其核心特色是採用了**通用適配器架構 (Universal Adapter Architecture)**，使其能夠輕鬆擴充以支援來自不同客戶端（如 LobeChat、Cherry Studio 等）的 API 格式。

此代理伺服器能夠智慧地判斷使用者請求是否包含圖片，並自動協調一個「思考模型」和一個「視覺模型」來處理多模態請求，最終將格式化的回應串流回原始的客戶端。

---

## 架構設計

本專案的核心是**將「核心業務邏輯」與「API 適配邏輯」分離**。

1.  **核心邏輯 (Core Logic)**: 這是此專案最有價值的部分，負責處理所有與模型互動的通用任務，例如：
    *   判斷請求是否需要圖像分析。
    *   調用視覺模型 (`VISION_MODEL`) 來描述圖片。
    *   組合圖片描述與使用者問題，調用思考模型 (`THINKING_MODEL`) 進行最終回答。
    *   處理流式回應。

2.  **適配器 (Adapters)**: 針對每一個需要支援的客戶端應用程式 (AP)，都有一個專屬的適配器。每個適配器只負責兩件事：
    *   **請求翻譯**: 將特定客戶端的 API 請求格式（“方言”）翻譯成代理伺服器內部的標準格式。
    *   **回應翻譯**: 將代理伺服器的標準流式回應，轉換為原始客戶端期望的格式和 API 端點。

這種設計使得核心邏輯保持穩定、乾淨，同時讓新增對其他客戶端的支援變得非常簡單和低風險。

## 主要功能

- **智慧多模態處理**: 自動識別圖片並採用「看圖說話 -> 思考總結」的流程處理圖文請求。
- **通用適配器架構**: 可透過在 `adapters.py` 中新增適配器，快速支援新的 API 格式。
- **客戶端支援**: 目前內建支援：
  - **LobeChat** (`/api/chat`)
  - **OpenAI 相容客戶端** (如 Cherry Studio) (`/v1/chat/completions`)
- **模型自訂**: 可在 `proxy_server.py` 中輕鬆更換用於思考和視覺的 Ollama 模型。
- **流式回應**: 將最終答案以串流方式傳回客戶端，提升使用者體驗。

## 安裝與設定

1.  **前置需求**:
    *   請確保您的系統已安裝 [Ollama](https://ollama.com/)。
    *   下載本專案所需的模型。預設模型為：
      ```bash
      ollama pull gpt-oss:20b # 思考模型
      ollama pull gemma3:4b    # 視覺模型
      ```
      *您可以在 `proxy_server.py` 中修改 `THINKING_MODEL` 和 `VISION_MODEL` 變數。*

2.  **安裝依賴**:
    在專案根目錄下，透過 `requirements.txt` 安裝所需的 Python 套件。
    ```bash
    pip install -r requirements.txt
    ```

## 如何使用

1.  **啟動代理伺服器**:
    ```bash
    python proxy_server.py
    ```
    成功啟動後，您會看到伺服器正在 `http://localhost:5000` 上監聽。

2.  **設定您的客戶端**:
    將您的 AI 客戶端（如 LobeChat）的 Ollama API 位址指向此代理伺服器。

    - **對於 LobeChat**:
      - API 位址: `http://localhost:5000/api/chat`
      - 模型名稱: `gpt-oss:20b` (或其他您在代理中設定的思考模型)

    - **對於 OpenAI 相容客戶端**:
      - API 位址: `http://localhost:5000/v1`
      - 模型名稱: `gpt-oss:20b`

## 如何支援一個新的客戶端

假設您想支援一個名為 "Future Assist" 的新客戶端，只需遵循以下步驟：

1.  **偵錯其 API 格式**: 了解 "Future Assist" 呼叫的 API 路徑（例如 `/api/v3/chat`）以及它傳送的 JSON 結構。

2.  **建立新適配器**: 在 `adapters.py` 檔案中，建立一個繼承自 `BaseAdapter` 的新類別。
    ```python
    class FutureAssistAdapter(BaseAdapter):
        name = "future_assist"

        def parse(self):
            # 在此實現解析 "Future Assist" JSON 格式的邏輯
            # ...
            return user_prompt, image_base64
    ```

3.  **註冊新適配器**: 在 `adapters.py` 底部的 `ADAPTERS` 字典中，將新客戶端的 API 路徑關鍵字與您剛建立的適配器類別關聯起來。
    ```python
    ADAPTERS = {
        "api/chat": LobeChatAdapter,
        "v1/chat/completions": CherryStudioAdapter,
        "api/v3/chat": FutureAssistAdapter  # 新增此行
    }
    ```

4.  **重啟伺服器**: 重新啟動 `proxy_server.py` 即可生效。就是這麼簡單！

---
## ⚖️ 授權與感謝

此專案為開源項目，使用了多個第三方函式庫。
請仔細閱讀其授權條款，在商業用途前務必審視清楚。

# Adapter AI Proxy (English Version)

This is an intelligent proxy server built with Python and Flask, designed for Large Language Model (LLM) backends like Ollama. Its core feature is the **Universal Adapter Architecture**, which allows it to be easily extended to support API formats from various clients (e.g., LobeChat, Cherry Studio).

This proxy can intelligently determine if a user's request includes an image, automatically coordinate a "thinking model" and a "vision model" to handle multimodal requests, and stream the formatted response back to the original client.

---

## Architecture

The core principle of this project is the **separation of "Core Logic" from "API Adaptation Logic."**

1.  **Core Logic**: This is the most valuable part of the project, responsible for handling all common tasks related to model interaction, such as:
    *   Determining if a request requires image analysis.
    *   Calling the vision model (`VISION_MODEL`) to describe an image.
    *   Combining the image description with the user's question and calling the thinking model (`THINKING_MODEL`) for a final answer.
    *   Handling streaming responses.

2.  **Adapters**: For each client application (AP) you want to support, there is a dedicated adapter. Each adapter is responsible for only two things:
    *   **Request Translation**: Translating the specific client's API request format (its "dialect") into the proxy's internal standard format.
    *   **Response Translation**: Converting the proxy's standard streaming response back into the format and API endpoint expected by the original client.

This design keeps the core logic stable and clean, while making it simple and low-risk to add support for new clients.

## Key Features

- **Intelligent Multimodal Handling**: Automatically detects images and uses a "describe-then-think" workflow for vision-language tasks.
- **Universal Adapter Architecture**: Quickly support new API formats by adding a new adapter in `adapters.py`.
- **Client Support**: Built-in support for:
  - **LobeChat** (`/api/chat`)
  - **OpenAI-compatible clients** (like Cherry Studio) (`/v1/chat/completions`)
- **Customizable Models**: Easily swap the thinking and vision models in `proxy_server.py`.
- **Streaming Responses**: Streams the final answer back to the client for a better user experience.

## Installation and Setup

1.  **Prerequisites**:
    *   Ensure you have [Ollama](https://ollama.com/) installed on your system.
    *   Pull the models required for this project. The default models are:
      ```bash
      ollama pull gpt-oss:20b # Thinking Model
      ollama pull gemma3:4b    # Vision Model
      ```
      *You can change the `THINKING_MODEL` and `VISION_MODEL` variables in `proxy_server.py`.*

2.  **Install Dependencies**:
    In the project's root directory, install the required Python packages using `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

1.  **Start the Proxy Server**:
    ```bash
    python proxy_server.py
    ```
    Once started, you will see the server listening on `http://localhost:5000`.

2.  **Configure Your Client**:
    Point your AI client (e.g., LobeChat) to this proxy server's address.

    - **For LobeChat**:
      - API Endpoint: `http://localhost:5000/api/chat`
      - Model Name: `gpt-oss:20b` (or your configured thinking model)

    - **For OpenAI-Compatible Clients**:
      - API Base URL: `http://localhost:5000/v1`
      - Model Name: `gpt-oss:20b`

## How to Support a New Client

Let's say you want to support a new client called "Future Assist." Just follow these steps:

1.  **Investigate its API Format**: Find out the API path (e.g., `/api/v3/chat`) and the JSON structure that "Future Assist" sends.

2.  **Create a New Adapter**: In the `adapters.py` file, create a new class that inherits from `BaseAdapter`.
    ```python
    class FutureAssistAdapter(BaseAdapter):
        name = "future_assist"

        def parse(self):
            # Implement the logic to parse the "Future Assist" JSON format here
            # ...
            return user_prompt, image_base64
    ```

3.  **Register the New Adapter**: In the `ADAPTERS` dictionary at the bottom of `adapters.py`, map the new client's API path keyword to the adapter class you just created.
    ```python
    ADAPTERS = {
        "api/chat": LobeChatAdapter,
        "v1/chat/completions": CherryStudioAdapter,
        "api/v3/chat": FutureAssistAdapter  # Add this line
    }
    ```

4.  **Restart the Server**: Relaunch `proxy_server.py` to apply the changes. It's that simple!

## 🙏 License & Acknowledgements

This project is open-source and uses various third-party libraries.
Please review their licenses carefully before using this code for commercial purposes.