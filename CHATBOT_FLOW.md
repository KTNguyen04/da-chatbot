# Chatbot Flow (End-to-End)

Tài liệu này mô tả **flow của chatbot** trong repo `da-chatbot`: từ lúc người dùng nhập câu hỏi → hệ thống quyết định nhánh xử lý → gọi LLM/RAG/tools → chạy code sandbox → trả kết quả **text / biểu đồ / cả hai**.

> Nguồn flow chính: `app.py` (`process_chat()`), `prompt.py` (`process_prompt()`), `agent.py` (`AgentAI.chat()`), `rag_docs_manager.py`, `tool_executor.py`, `tools/*`.

---

## Sơ đồ tổng quan (Mermaid)

```mermaid
flowchart TD
  U[User nhập câu hỏi<br/>Streamlit chat_input] --> A[app.py: process_chat()]

  A --> B[STEP 1: should_answer_normally()<br/>intent routing / short-circuit]

  B -->|Greeting / Help| SC1[Trả lời tĩnh<br/>(không code)]
  B -->|Dataset overview| SC2[answer_dataset_overview(df)<br/>(không code)]
  B -->|Context query| SC3[RAG docs + LLM text<br/>build_rag_context() + _stream_llm_response()]
  B -->|Direct chat| SC4[LLM text-only<br/>(không code)]

  B -->|Visualization / data analysis| P0[Đi vào luồng phân tích]

  P0 -->|Price-driver (optional)| TI0[Tool Insight precompute<br/>run_tool_insight()]
  P0 --> P1[prompt.py: process_prompt()<br/>metadata + graph routing + RAG inject + code instructions]
  P1 --> AG[agent.py: AgentAI.chat()<br/>LLM sinh Python code]
  AG --> EX[Sandbox exec_code()<br/>restricted imports + DF_1.. + GEOJSON]
  EX --> R[result = text OR df OR figure OR {figure,analysis}]

  R --> TI1[STEP 5: Tool Insight (optional)<br/>run_tool_insight()]
  TI1 --> OUT[STEP 6: Normalize & render<br/>text / plotly fig / both]
  R -->|tool insight skipped| OUT

  OUT --> SAVE[Lưu st.session_state.messages + last_code]
  SAVE --> RR[st.rerun()]
  RR --> A
```

---

## Step-by-step chi tiết

## 0) Khởi tạo app và dữ liệu

- **Entrypoint**: `app.py: main()`
- **Dataset**: load cố định 1 CSV qua `load_main_dataset()` và đưa vào dict `data = {"DF_1___...": df}`.
- **LLM chính (text/code)**: `ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=slider)`
- **Agent runtime**: `get_llm_agent()` tạo `AgentAI(data=[df], llm=llm, ...)` và lưu vào `st.session_state["agent_var"]`.

**Dùng gì → để làm gì**
- **`load_main_dataset()`** → *để có DataFrame “nguồn sự thật”* cho mọi phân tích.
- **`ChatOllama`** → *để dùng LLM cho 2 việc*: (1) phân loại intent (fallback) + (2) sinh code Python khi cần phân tích/vẽ.
- **`AgentAI`** → *để chạy LLM theo contract “trả code” và thực thi code trong sandbox* (thay vì trả lời tự do).

**Input/Output**
- **Input**: dataset CSV cố định (`datasets/vietnam_housing_dataset_cleaned.csv`) + LLM endpoint (`OLLAMA_BASE_URL`).
- **Output**: `data` dict cho prompt builder, và `agent_var` để gọi `agent.chat(...)` về sau.

Kết quả: mọi câu hỏi phân tích đều có thể tham chiếu `DF_1` trong sandbox code; UI chỉ cần truyền `data` vào `process_prompt()`.

---

## 1) Nhận input và ghi lịch sử chat

Trong `app.py: process_chat()`:

- User nhập câu hỏi qua `st.chat_input(...)`.
- App lưu message vào history: `{"role":"user","question": user_question}`.
- Có bước rewrite nếu user trả lời follow-up ngắn sau dataset overview để router nhận diện ý định “phân tích”.

**Dùng gì → để làm gì**
- **`st.chat_input(...)`** → *để nhận input* ở UI dạng chat.
- **`st.session_state.messages`** → *để lưu memory hội thoại* (history) cho routing và prompt building.
- **`_rewrite_followup_question_for_analysis(...)`** → *để “mở rộng” câu trả lời ngắn* thành câu hỏi có chủ đích phân tích (giúp router nhận ra chart/statistics).

**Logic rewrite cụ thể**
- Chỉ rewrite nếu câu trước đó (assistant) có gợi ý kiểu: `"vẽ biểu đồ/xem thống kê mô tả cho cột nào"` và user trả lời ngắn (vd `"Price"`, `"Area"`…).
- Rewrite thành mẫu: `Vẽ biểu đồ phù hợp và thống kê mô tả cho cột '<q>'...`.

---

## 2) STEP 1 — Intent routing / short-circuit (`should_answer_normally`)

Mục tiêu: **tránh luồng “sinh code + exec”** khi câu hỏi không cần chart/tính toán.

`should_answer_normally(user_question, llm, messages)` trả về `(should_short_circuit, hint)`:

### 2.1) Greeting / Help (text-only)
- Trả lời tĩnh bằng `_stream_text(...)`.
- Không chạy RAG, không chạy agent.

**Dùng gì → để làm gì**
- **Regex `_is_greeting()`** → *bắt chào hỏi nhanh* để giảm latency (không gọi LLM).
- **Keyword match `_contains_any(..., help_keywords)`** → *bắt câu hỏi “bạn làm gì/cách dùng”* để trả lời hướng dẫn.

### 2.2) Dataset overview (text-only)
- Trả lời tĩnh bằng `answer_dataset_overview(df)` (số dòng, số cột, preview tên cột).

**Dùng gì → để làm gì**
- **Keyword match `dataset_keywords`** → *bắt intent hỏi danh sách cột/dataset gồm gì*.
- **`answer_dataset_overview(df)`** → *trả lời nhanh* bằng thống kê đơn giản (shape + list cột), rồi hỏi lại user muốn phân tích cột nào.

### 2.3) Context query (RAG + LLM text-only)
- Dùng `rag_docs_manager.build_rag_context(question, top_k, min_similarity)`
- Dựng prompt “giải thích ý nghĩa/ngữ cảnh” và gọi `_stream_llm_response(llm, rag_prompt)`
- Không sinh code.

**Dùng gì → để làm gì**
- **Keyword match `context_keywords`** → *bắt câu hỏi kiểu “cột X nghĩa là gì / mô tả / nguồn gốc…”*.
- **RAG `build_rag_context()`** → *lấy ngữ cảnh từ tài liệu trong `rag_docs/`* để tránh LLM bịa.
- **Prompt “Không sinh code”** → *đảm bảo output chỉ là giải thích ngôn ngữ tự nhiên*.

**Prompt dùng trong nhánh này (tóm tắt)**
- Ép: “Không sinh code. Chỉ giải thích…” và kèm block `<rag_docs_context>...</rag_docs_context>` nếu có.

### 2.4) Direct chat (LLM text-only)
- Gọi `_stream_llm_response` với prompt ép “không sinh code”.

**Dùng gì → để làm gì**
- **Fallback `classify_intent_llm()`** (LLM-based) → *khi keyword match không chắc*, dùng LLM phân loại intent.
- **Prompt “Do NOT generate any code…”** → *giữ câu trả lời dạng chat*, không kéo vào luồng code-agent.

**Prompt dùng trong nhánh này (tóm tắt)**
- “Do NOT generate any code… Answer in Vietnamese only.”

### 2.5) Visualization / data analysis (đi tiếp luồng phân tích)
Nếu nhận diện câu hỏi có intent vẽ biểu đồ / thống kê / so sánh / ranking… thì **không short-circuit** và chuyển sang Step 3.

**Dùng gì → để làm gì**
- **Keyword match `visualization_keywords`** → *bắt tín hiệu* “vẽ/biểu đồ/plot/chart/so sánh/top/xếp hạng/nhiều nhất…” để đi vào luồng phân tích.
- Nếu keyword match không rõ, **LLM classifier `classify_intent_llm()`** vẫn có thể trả `visualization_request` để đi tiếp.

---

## 3) STEP 2 — (Optional) Tool Insight precompute cho “price-driver”

Trong `app.py`, nếu:
- `ENABLE_TOOL_INSIGHT = True`
- và `_is_price_driver_question(...) == True`

thì app chạy **tool-calling insight** trước khi gọi code-agent:

- Gọi `run_tool_insight(user_question, df, ollama_base_url, model, lang, verbose)`
- Lưu ra `precomputed_tool_insight` để ghép vào output sau đó.

**Dùng gì → để làm gì**
- **Heuristic `_is_price_driver_question()`** (keyword match: “ảnh hưởng/tác động/diện tích/phòng ngủ/…” + “giá”) → *nhận diện nhóm câu hỏi cần số liệu định lượng + ngoại lệ*.
- **Tool executor `run_tool_insight()`** → *buộc LLM gọi tool để tính số liệu thật* (corr/slope/median theo nhóm/bins…) trước khi agent sinh code vẽ.

**Tại sao cần precompute?**
- Nhánh price-driver thường cần “bằng chứng số liệu” + “ngoại lệ” + “kết luận có điều kiện”. Tool insight cho số liệu đáng tin cậy, sau đó app ghép vào phần analysis.

Ý nghĩa:
- Tool insight lấy số liệu thật từ DataFrame thông qua tool-calling (giảm hallucination).
- Đặc biệt hữu ích cho các câu hỏi “yếu tố nào ảnh hưởng giá…”.

---

## 4) STEP 3 — Build prompt cho code-agent (`prompt.py: process_prompt`)

`process_prompt(session_msgs, user_question, data, llm)` dựng một prompt gồm các phần chính:

### 4.1) `<metadata>` (dataset schema + sample)
- `df.info()` output vào buffer
- random sample N dòng (theo `st.session_state["sample_var"]`)

**Dùng gì → để làm gì**
- **`df.info()`** → *cho LLM biết schema/dtypes/missingness*.
- **Sample rows** → *cho LLM thấy giá trị thực tế* (để chọn đúng cột, đúng kiểu biểu đồ).

### 4.2) Graph routing (`define_graph_type`)
- Nếu “price-driver” → ép `graph_type = "scatter_2d_plot"`
- Còn lại gọi `classify_intent(llm, question, history)` để lấy `graph_type`.

**Dùng gì → để làm gì**
- **Match price-driver trong `define_graph_type()`** → *route thẳng sang scatter* (phù hợp quan hệ 2 biến Area–Price…).
- **LLM router `classify_intent()`** → *trả JSON* có `graph_type` (và intent/target_col nếu parse được).

**Prompt router `classify_intent()` (tóm tắt)**
- Yêu cầu LLM trả **một JSON**:
  - `"intent": "data_analysis" | "metadata_query" | "general_chat"`
  - `"graph_type": "... | none"`
  - `"target_col": "... | none"`
- Input cho prompt: `history` + `question`.

### 4.3) “Poor-man’s RAG” theo loại biểu đồ (`ragdata/*.txt`)
- Đọc file `ragdata/<graph_type>.txt` nếu có
- fallback `ragdata/base_ref.txt`
- Inject vào prompt trong tag `<code_ref>` như “gợi ý code plotly”.

**Dùng gì → để làm gì**
- **File templates trong `ragdata/`** → *chuẩn hoá style code Plotly* (update_layout/update_traces, cấu hình trục/title/legend…).
- **`<code_ref>`** → *đưa “mẫu code đúng” vào prompt* để LLM bám theo, giảm lỗi vẽ/format.

### 4.4) Semantic RAG từ `rag_docs/` (ChromaDB)
- `rag_docs_manager.build_rag_context(...)` trả về block `<rag_docs_context>...</rag_docs_context>`
- Inject ngay sau `<metadata>` trong prompt.

**Dùng gì → để làm gì**
- **ChromaDB + sentence-transformers** (`rag_docs_manager.py`) → *lấy đoạn docs liên quan theo semantic similarity* (vd mô tả dataset, playbook phân tích…).
- **`min_similarity`** → *lọc kết quả kém liên quan*; nếu không có doc vượt ngưỡng thì fallback top-1 để vẫn có context.

### 4.5) Output contract: agent phải set biến `result`
`process_prompt()` ép LLM sinh code theo contract:

- **Plot-type**: `result = {"figure": fig, "analysis": analysis}`
- **Table**: `result = <pandas dataframe>`
- **Non-plot**: `result = "<text tiếng Việt>"`

**Dùng gì → để làm gì**
- **Contract `result`** → *để UI hiểu và render đúng loại output*, và để agent “tự kiểm tra” output trước khi trả.
- **Giải pháp “plot = dict figure+analysis”** → *đảm bảo có cả hình lẫn diễn giải định lượng*.
- **Hint “PRICE DRIVER HINT” / “PROVINCE RANKING HINT”** (nếu match keyword) → *ép agent nêu caveat và định lượng* theo đúng nhóm câu hỏi.

---

## 5) STEP 4 — LLM sinh code + sandbox exec (`agent.py: AgentAI.chat`)

`AgentAI.chat(prompt)` chạy vòng lặp tối đa `max_attempts`:

### 5.1) LLM trả code
- `llm.invoke(prompt)` → lấy `response.content`
- Regex trích block ```python ... ```
- Lưu `self.last_code`

**Dùng gì → để làm gì**
- **Regex `r"```python(.*?)```"`** → *bóc tách phần code để chạy*, bỏ qua phần text thừa.
- **`self.last_code` + UI expander** → *giúp debug/giải trình* (user có thể xem code đã chạy).

### 5.2) Sandbox execute
`exec_code(code)`:
- Tạo env qua `create_isolated_env()`:
  - inject `DF_1`, `DF_2`, ... (từ list dataframes)
  - inject `GEOJSON`
  - override `__import__` bằng `restricted_import` (chặn import ngoài whitelist)
- `exec(code, env, context)` và lấy `context["result"]`
- chặn `plotly.Figure.show()` (bắt buộc trả fig object thay vì show)

**Dùng gì → để làm gì**
- **Whitelist import (`restricted_import`)** → *chặn code nguy hiểm / import không mong muốn*.
- **Inject `DF_1..`** → *cho code truy cập trực tiếp DataFrame* mà không đọc file.
- **Chặn `fig.show()`** → *tránh side-effect*, buộc trả figure object để Streamlit render.

### 5.3) Error-retry
Nếu code lỗi:
- thêm `<code_error>` (kèm line numbers) và `<message_error>` vào prompt
- gọi lại LLM để tự sửa code (tăng temperature sau nửa số attempt).

**Dùng gì → để làm gì**
- **`<code_error>` + line numbers** → *đưa lỗi chính xác* để LLM sửa đúng dòng.
- **`<message_error>` (exception type/trace/message)** → *giúp LLM biết nguyên nhân* (KeyError, ImportError, …).
- **Retry loop** → *tăng tỷ lệ “tự sửa lỗi”* mà không cần người dùng can thiệp.

Kết quả từ agent là một trong các type:
- `str` (text)
- `go.Figure`
- `dict` (đặc biệt `{"figure": fig, "analysis": str}`)
- `pd.DataFrame`
- hoặc “EXCEPTION ERROR: ...”

---

## 6) STEP 5 — Tool-calling insight (optional) (`tool_executor.py`)

Sau khi có `response` (và/hoặc `figure`), `app.py` có thể gọi tool insight lần nữa:

### 6.1) Tool registry (`tools/__init__.py`)
- Auto-load tất cả module trong `tools/`
- Mỗi module cung cấp:
  - `TOOL_DEFINITIONS` (schema function-calling)
  - `TOOL_FUNCTIONS` (implementation Python)

**Dùng gì → để làm gì**
- **Tool registry auto-discovery** → *thêm tool mới không cần sửa engine* (chỉ cần tạo file `tools/*.py`).
- **Schema `TOOL_DEFINITIONS`** → *dạy LLM khi nào gọi tool và arguments cần gì*.
- **Implementation `TOOL_FUNCTIONS`** → *thực thi tính toán thật* (pandas/numpy) trên DataFrame.

### 6.2) Tool executor (`run_tool_insight`)
`run_tool_insight(...)`:
- Inject DataFrame vào tool modules (vd `tools/housing_tools.set_dataframe(df)`)
- Gọi Ollama `/api/chat` với `tools=[...]`
- Nếu LLM trả `tool_calls`:
  - parse arguments
  - gọi Python function tương ứng
  - gửi lại kết quả tool vào messages (role=`tool`)
- Lặp tối đa `MAX_TOOL_ROUNDS`
- Trả về **một đoạn insight text** (tiếng Việt) để ghép vào output.

**Dùng gì → để làm gì**
- **Ollama `/api/chat` + tools** → *thực hiện function-calling multi-round*: LLM quyết định gọi tool nào, nhận kết quả, rồi tổng hợp insight.
- **`MAX_TOOL_ROUNDS`** → *chặn vòng lặp vô hạn*.
- **System prompt `_build_system_prompt()`** → *gợi ý tool ưu tiên* theo keyword (province ranking / price drivers).

**Prompt tool-insight (tóm tắt)**
- System: liệt kê tool names + rules (“luôn gọi tool khi cần số liệu”).
- User: “Hãy gọi tool phù hợp để lấy số liệu, sau đó trả lời insight ngắn gọn…”

Các tool “housing” tiêu biểu ở `tools/housing_tools.py`:
- `get_province_ranking`
- `describe_numeric_column`
- `compare_mean_by_group`
- `analyze_price_drivers`, `price_vs_size_summary`, `structure_group_price_compare`, ...

---

## 7) STEP 6 — Chuẩn hoá output & render UI (text / chart / both)

Trong `app.py`:

### 7.1) Nếu response là string
- stream text
- nếu có tool insight: ghép thêm `---` + “🔧 Tool Insight”
- nếu lỗi EXCEPTION: fallback hiển thị tool insight (nếu có)

**Dùng gì → để làm gì**
- **`_stream_text()`** → *trải nghiệm streaming* (word-by-word) thay vì đợi render 1 lần.
- **Ghép tool insight** → *tăng độ tin cậy định lượng* (số liệu từ tool) ngay cả khi agent trả text.

### 7.2) Nếu response là dict có `figure`
- render `st.plotly_chart(response["figure"])`
- tạo `combined_insight = agent_analysis + tool_insight_text` rồi gán lại vào `response["analysis"]`
- stream `combined_insight`

**Dùng gì → để làm gì**
- **Contract `{figure, analysis}`** → *UI render được cả biểu đồ lẫn diễn giải*.
- **`combined_insight`** → *gộp “agent analysis” (từ code) + “tool insight” (từ tool-calling)* thành narrative cuối.

### 7.3) Nếu response là `go.Figure`
- render figure
- nếu có tool insight, chuẩn hoá thành `{"figure": fig, "analysis": tool_insight}`

**Dùng gì → để làm gì**
- **Chuẩn hoá về dict** → *giữ format nhất quán trong history* để lần rerun sau vẫn render đúng.

### 7.4) Lưu session & rerun
Cuối cùng:
- append `{"role":"assistant","response": ...}` vào `st.session_state.messages`
- lưu `st.session_state["last_code"] = llm_agent.get_last_code()`
- `st.rerun()` để UI render history nhất quán.

**Dùng gì → để làm gì**
- **`st.session_state.messages`** → *memory + replay UI* (mỗi rerun Streamlit sẽ render lại toàn history).
- **`last_code` expander** → *debug/giải trình* code đã chạy.
- **`st.rerun()`** → *đảm bảo UI hiển thị đồng bộ* sau khi stream xong và lưu state.

---

## Ghi chú: “RAG” trong repo có 2 lớp

- **Lớp 1 (ragdata/)**: mapping `graph_type` → code-hints (txt), dùng để hướng dẫn Plotly/code style.
- **Lớp 2 (rag_docs/ + ChromaDB)**: semantic search cho docs (md/txt/py), inject `<rag_docs_context>` vào prompt hoặc dùng riêng cho “context query”.

