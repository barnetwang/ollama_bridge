# 智慧型 AI 代理伺服器 (An Intelligent AI Proxy Server)

這是一個基於 Python 和 Flask 的智慧代理伺服器，專為大型語言模型（LLM）後端（如 Ollama）設計。它不僅僅是一個請求轉發器，而是一個**智慧、多模態、多角色的 AI 代理 (AI Agent)**。

它的核心能力是**理解使用者意圖**，並**動態地組織一個虛擬專家團隊**來生成高品質的、符合情境的回答。它還能無縫處理圖文混合請求，並通過**通用適配器架構**輕鬆擴展以支援多種前端應用。

---

## ✨ 主要功能 (Key Features)

- **🧠 動態專家角色融合 (Dynamic Expert Persona Fusion)**:
  能根據用戶的跨領域問題（例如「寫一篇關於達文西手術的小說」），自動選擇並融合多位專家（如 `Doctor` + `Writer`）的能力，生成兼具深度與創意的回答。

- **🔧 @前綴工具調用 (Prefix-based Tool Calling)**:
  通過簡單的 `@網路搜尋` 指令，即可觸發代理伺服器執行 Google 搜尋，將即時的網路資訊作為上下文提供給 LLM，使其能夠回答關於最新事件的問題。

- **🖼️ 無縫多模態處理 (Seamless Multimodal Handling)**:
  自動識別圖片並採用「看圖說話 -> 專家思考總結」的流程，能夠將專家角色能力應用於圖文混合的問答中。

- **🔌 通用適配器架構 (Universal Adapter Architecture)**:
  可透過在 `adapters.py` 中新增適配器，快速支援新的 API 格式，目前內建支援 **LobeChat** 和 **OpenAI 相容客戶端** (如 Cherry Studio)。

- **✍️ 易於維護的 Prompt 庫 (Externalized Prompt Library)**:
  所有專家角色的能力和行為準則都定義在 `prompts/` 資料夾下的 `.txt` 檔案中，新增或修改 AI 的角色完全無需更動程式碼。

- **🚀 流式回應 (Streaming Responses)**:
  將最終答案以串流方式傳回客戶端，提供流暢的使用者體驗。

## 🏛️ 架構設計 (Architecture)

本專案的核心是**將「核心業務邏輯」與「API 適配邏輯」徹底分離**。

1.  **核心邏輯 (`proxy_server.py`)**: 負責所有高階智慧任務：
    *   **意圖識別**: 判斷使用者是否想調用工具（如 `@網路搜尋`）。
    *   **專家決策**: 調用 LLM 分析問題，決定需要哪一位或哪幾位專家來協作回答。
    *   **Prompt 融合**: 動態地將專家 Prompt 和工具搜尋結果融合成一個強大的「元 Prompt」。
    *   **多模態協調**: 處理圖文請求的特殊流程。

2.  **適配器 (`adapters.py`)**: 針對每一個客戶端，都有一個專屬的適配器，負責：
    *   **請求翻譯**: 將客戶端的獨特 JSON 格式翻譯成內部標準格式。
    *   **回應端點選擇**: 告訴核心邏輯應使用哪個 API 端點 (`/api/chat` 或 `/v1/chat/completions`) 來回應，確保格式兼容。

3.  **Prompt 庫 (`prompts/`)**:
    *   以獨立 `.txt` 檔案形式存放所有專家角色的定義。

## 🛠️ 安裝與設定 (Installation & Setup)

#### 1. 前置需求 (Prerequisites)

- **Ollama**: 請確保您的系統已安裝 [Ollama](https://ollama.com/)。
- **Ollama 模型**: 下載本專案所需的模型。
  ```bash
  # 思考模型 (Thinking Model)
  ollama pull gpt-oss:20b
  # 視覺模型 (Vision Model)
  ollama pull gemma3:4b
  ```
  *您可以在 `proxy_server.py` 的頂部修改 `THINKING_MODEL` 和 `VISION_MODEL` 變數。*

#### 2. Google 搜尋 API (可選，但推薦)

- 若要啟用 `@網路搜尋` 功能，您需要一個 Google Custom Search API 金鑰。
  1.  在 [Google Cloud Console](https://console.cloud.google.com/) 啟用 "Custom Search API"。
  2.  建立一個 **API 金鑰**。
  3.  在 [可程式化搜尋引擎](https://programmablesearchengine.google.com/controlpanel/all) 建立一個搜尋引擎（選擇「在整個網路上搜尋」），並獲取其 **搜尋引擎 ID**。
  4.  在專案根目錄下，建立一個 `.env` 檔案，並填入您的金鑰：
      ```.env
      GOOGLE_API_KEY="您的_API_金鑰"
      GOOGLE_CSE_ID="您的_搜尋引擎_ID"
      ```

#### 3. 安裝 Python 依賴

- 在專案根目錄下，建立 `requirements.txt` 檔案：
  ```txt
  Flask
  requests
  python-dotenv
  google-api-python-client
  ```
- 執行安裝：
  ```bash
  pip install -r requirements.txt
  ```

#### 4. 建立 Prompt 庫

- 在專案根目錄下，建立一個名為 `prompts` 的資料夾。
- 在此資料夾內，建立您的專家角色檔案（檔名必須為**英文**）。例如：`Doctor.txt`, `Writer.txt`, `Assistant.txt`。

## 🚀 如何使用 (How to Use)

1.  **啟動代理伺服器**:
    ```bash
    python proxy_server.py
    ```
    伺服器將在 `http://localhost:5000` 上監聽。

2.  **設定您的客戶端**:
    - **對於 LobeChat (推薦)**:
      - API 端點: `http://localhost:5000`
      - 模型名稱: `gpt-oss:20b` (或其他您設定的思考模型)
    - **對於 OpenAI 相容客戶端**:
      - API Base URL: `http://localhost:5000/v1`
      - 模型名稱: `gpt-oss:20b`

3.  **開始對話**:
    - **常規問題**: 「達文西手術是什麼？」 -> 將由 `Doctor` 角色回答。
    - **跨領域問題**: 「寫一篇關於達文西手術的小說」 -> 將由 `Doctor` 和 `Writer` 團隊協同回答。
    - **網路搜尋**: 「@網路搜尋 特斯拉 Cybertruck 的特點」 -> 將觸發 Google 搜尋並基於即時資訊回答。

## 🧩 如何擴展 (How to Extend)

### 新增一位專家

1.  在 `prompts` 資料夾中，新增一個 `.txt` 檔案，例如 `Historian.txt`。
2.  在檔案內，編寫這位專家的角色定義和行為準則。
3.  重啟代理伺服器即可！

### 支援一個新的客戶端

1.  **偵錯** "Future Assist" 的 API 路徑和 JSON 結構。
2.  在 `adapters.py` 中，為其**建立**一個新的 `FutureAssistAdapter` 類。
3.  在 `adapters.py` 的 `ADAPTER_REGISTRY` 列表中，**註冊**這個新適配器。
4.  重啟伺服器即可生效！

---
## ⚖️ 授權與感謝 (License & Acknowledgements)

此專案為開源項目，使用了多個第三方函式庫。請仔細閱讀其授權條款，在商業用途前務必審視清楚。

# Adapter AI Proxy (English Version)

# 智慧型 AI 代理伺服器 (An Intelligent AI Proxy Server)

這是一個基於 Python 和 Flask 的智慧代理伺服器，專為大型語言模型（LLM）後端（如 Ollama）設計。它不僅僅是一個請求轉發器，而是一個**智慧、多模態、多角色的 AI 代理 (AI Agent)**。

它的核心能力是**理解使用者意圖**，並**動態地組織一個虛擬專家團隊**來生成高品質的、符合情境的回答。它還能無縫處理圖文混合請求，並通過**通用適配器架構**輕鬆擴展以支援多種前端應用。

---
---

<br>

# [ English Version ]

This is an intelligent proxy server built with Python and Flask, designed for Large Language Model (LLM) backends like Ollama. It functions as an **intelligent, multimodal, multi-persona AI Agent** that goes beyond simple request forwarding.

Its core capability is to **understand user intent** and **dynamically assemble a virtual team of experts** to generate high-quality, context-aware responses. It seamlessly handles mixed text-and-image requests and is easily extensible to support various front-end applications through its **Universal Adapter Architecture**.

---

## ✨ Key Features

- **🧠 Dynamic Expert Persona Fusion**:
  Based on a user's cross-disciplinary query (e.g., "write a short story about Da Vinci surgery"), it can automatically select and fuse the capabilities of multiple experts (like `Doctor` + `Writer`) to generate a response with both depth and creativity.

- **🔧 Prefix-based Tool Calling**:
  Using a simple `@web_search` command, the proxy can trigger a Google Search to provide the LLM with real-time information, enabling it to answer questions about recent events.

- **🖼️ Seamless Multimodal Handling**:
  Automatically detects images and uses a "describe-then-think" workflow, applying the selected expert persona's skills to answer complex vision-language questions.

- **🔌 Universal Adapter Architecture**:
  Quickly support new API formats by adding adapters in `adapters.py`. It comes with built-in support for **LobeChat** and **OpenAI-compatible clients** (like Cherry Studio).

- **✍️ Externalized & Easy-to-Maintain Prompt Library**:
  All expert personas, including their skills and behavioral guidelines, are defined in simple `.txt` files within the `prompts/` directory. Adding or modifying the AI's roles requires no code changes.

- **🚀 Streaming Responses**:
  Streams the final answer back to the client for a smooth, real-time user experience.

## 🏛️ Architecture

The core principle of this project is the **complete separation of "Core Logic" from "API Adaptation Logic."**

1.  **Core Logic (`proxy_server.py`)**: Handles all high-level intelligent tasks:
    *   **Intent Recognition**: Detects if the user wants to invoke a tool (e.g., `@web_search`).
    *   **Expert Decision**: Calls the LLM to analyze the query and decide which expert(s) are needed for the task.
    *   **Prompt Fusion**: Dynamically combines expert prompts and tool results into a powerful "meta-prompt."
    *   **Multimodal Coordination**: Manages the specialized workflow for text-and-image requests.

2.  **Adapters (`adapters.py`)**: A dedicated adapter exists for each client, responsible for:
    *   **Request Translation**: Translating the client's unique JSON format ("dialect") into the proxy's internal standard format.
    *   **Response Endpoint Selection**: Informing the core logic which API endpoint (`/api/chat` or `/v1/chat/completions`) to use for the response, ensuring format compatibility.

3.  **Prompt Library (`prompts/`)**:
    *   Contains all expert persona definitions as individual `.txt` files.

## 🛠️ Installation & Setup

#### 1. Prerequisites

- **Ollama**: Ensure you have [Ollama](https://ollama.com/) installed on your system.
- **Ollama Models**: Pull the required models.
  ```bash
  # Thinking Model
  ollama pull gpt-oss:20b
  # Vision Model
  ollama pull gemma3:4b
  ```
  *You can change the `THINKING_MODEL` and `VISION_MODEL` variables at the top of `proxy_server.py`.*

#### 2. Google Search API (Optional, but recommended)

- To enable the `@web_search` feature, you need a Google Custom Search API key.
  1.  Enable "Custom Search API" in the [Google Cloud Console](https://console.cloud.google.com/).
  2.  Create an **API Key**.
  3.  Create a search engine in the [Programmable Search Engine](https://programmablesearchengine.google.com/controlpanel/all) console (select "Search the entire web") and get its **Search engine ID**.
  4.  Create a `.env` file in the project root and add your credentials:
      ```.env
      GOOGLE_API_KEY="YOUR_API_KEY"
      GOOGLE_CSE_ID="YOUR_SEARCH_ENGINE_ID"
      ```

#### 3. Install Python Dependencies

- Create a `requirements.txt` file in the project root:
  ```txt
  Flask
  requests
  python-dotenv
  google-api-python-client
  ```
- Run the installation:
  ```bash
  pip install -r requirements.txt
  ```

#### 4. Create the Prompt Library

- Create a folder named `prompts` in the project root.
- Inside this folder, create your expert persona files with **English filenames**. For example:
    - `prompts/Doctor.txt`
    - `prompts/Writer.txt`
    - `prompts/Software_Engineer.txt`
    - `prompts/Assistant.txt` (This is the default fallback persona and is required).

## 🚀 How to Use

1.  **Start the Proxy Server**:
    ```bash
    python proxy_server.py
    ```
    The server will be listening on `http://localhost:5000`.

2.  **Configure Your Client**:
    - **For LobeChat (Recommended)**:
      - API Endpoint: `http://localhost:5000`
      - Path: `api/chat`
      - Model Name: `gpt-oss:20b` (or your configured thinking model)
    - **For OpenAI-Compatible Clients**:
      - API Base URL: `http://localhost:5000/v1`
      - Model Name: `gpt-oss:20b`

3.  **Start Chatting**:
    - **Normal Query**: "What is Da Vinci surgery?" -> Will be answered by the `Doctor` persona.
    - **Cross-Disciplinary Query**: "Write a short story about Da Vinci surgery" -> Will be answered by a `Doctor` and `Writer` team.
    - **Web Search**: "@web_search What are the features of the Tesla Cybertruck?" -> Will trigger a Google Search and answer based on real-time information.

## 🧩 How to Extend

### Adding a New Expert

This is the easiest way to extend the agent's capabilities:
1.  Add a new `.txt` file to the `prompts` folder, e.g., `Historian.txt`.
2.  Write the persona definition and behavioral guidelines inside the file.
3.  Restart the proxy server. The decision-making model will now automatically consider "Historian" as a potential team member!

### Supporting a New Client

1.  **Investigate** the new client's API path (e.g., `/api/v3/chat`) and JSON structure.
2.  **Create** a new `FutureAssistAdapter` class in `adapters.py`.
3.  **Register** this new adapter in the `ADAPTER_REGISTRY` list in `adapters.py`.
4.  Restart the server. It's that simple!

---
## ⚖️ License & Acknowledgements

This is an open-source project that utilizes various third-party libraries. Please review their respective licenses carefully before using this code for commercial purposes.
