# prompt.py
"""
Prompt builder tích hợp RAG docs tiêu chuẩn (ChromaDB).

Thay đổi so với bản gốc:
  1. Import rag_docs_manager ở đầu file.
  2. Trong process_prompt(): gọi build_rag_context() và inject vào prompt_main.
  3. define_graph_type() mở rộng nhận diện câu hỏi về tỉnh/thành và ranking.
"""

import os
import sys
import traceback
import streamlit as st

from io import StringIO
from colorama import Fore

# ── [NEW] RAG docs ────────────────────────────────────────────────────────────
from rag_docs_manager import build_rag_context, sync_index

# ─────────────────────────────────────────────────────────────────────────────


def classify_intent(llm: object, question: str, history: str) -> dict:
    """
    Sử dụng LLM để phân loại mục đích người dùng (Intent Classification).
    Thay thế cho việc match từ khóa cứng nhắc.
    """
    prompt = f"""
    Phân tích câu hỏi của người dùng về dữ liệu bất động sản và trả về JSON.
    
    Các loại Intent:
    1. 'data_analysis': Cần tính toán, vẽ biểu đồ, thống kê (ví dụ: "vẽ biểu đồ giá", "trung bình diện tích").
    2. 'metadata_query': Hỏi về ý nghĩa dữ liệu, định nghĩa cột (ví dụ: "cột address là gì?", "giá tính bằng đơn vị nào?").
    3. 'general_chat': Chào hỏi, tán gẫu.

    Trả về format JSON duy nhất:
    {{
        "intent": "data_analysis" | "metadata_query" | "general_chat",
        "graph_type": "box_plot" | "bar_chart" | "treemap_plot" | "none",
        "target_col": "tên cột liên quan hoặc none"
    }}

    Lịch sử: {history}
    Câu hỏi: {question}
    """
    try:
        import json
        import re

        response = llm.invoke(prompt).content
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return {"intent": "data_analysis", "graph_type": "none", "target_col": "none"}


def define_graph_type(llm: object, question_user: str, hist_questions: str) -> str:
    q = question_user.lower()
    price_driver_keywords = [
        "ảnh hưởng",
        "tác động",
        "liệu",
        "có luôn",
        "nhà to hơn",
        "diện tích",
        "phòng ngủ",
        "phòng tắm",
        "số tầng",
        "mặt tiền",
    ]
    if any(kw in q for kw in price_driver_keywords) and (
        "giá" in q or "price" in q
    ):
        return "scatter_2d_plot"

    # Hàm này giờ gọi classify_intent để lấy graph_type
    res = classify_intent(llm, question_user, hist_questions)
    return res.get("graph_type", "none")


# Old function header replaced by new logic below
def _deprecated_define_graph_type(
    llm: object, question_user: str, hist_questions: str
) -> str:
    """
    Phân tích câu hỏi qua LLM và xác định loại biểu đồ cho RAG routing.
    Mở rộng nhận diện câu hỏi về tỉnh/thành, xuất hiện nhiều nhất, xếp hạng.
    """

    prompt_template = f"""
    <question>
    Answer the following question with a single specific word.
    {question_user}
    Answer with just one word.
    </question>
    
    <hist_questions>
    {hist_questions}
    </hist_questions>

    Analyze the question within the <question> tags and determine which type of graph is most appropriate for 
    the question, according to the options listed in <allowed_words>.
    
    Analyze message history within the <hist_questions> tags for more context in response.

    <allowed_words>
    answer "treemap_plot" if has in question (treemap, biểu đồ treemap)
    answer "sunburst_plot" if has in question (sunburst, biểu đồ tròn phân cấp)
    answer "density_contour_plot" if has in question (density contour, đường đồng mức mật độ, contour mật độ)
    answer "violin_plot" if has in question (violin, biểu đồ violin, đàn violin)
    answer "candle_plot" if has in question (candle, nến, ohlc, candlestick, nến nhật)
    answer "heatmap" if has in question (heatmap, bản đồ nhiệt, ma trận tương quan, matrix)
    answer "scatter_2d_plot" if has in question (scatter, điểm phân tán, biểu đồ điểm)
    answer "bubble_plot" if has in question (bubble, bong bóng, biểu đồ bong bóng)
    answer "scatter_3d_plot" if has in question (scatter 3d, điểm 3d, biểu đồ 3 chiều)
    answer "surface_plot" if has in question (surface, bề mặt, biểu đồ bề mặt, 3d surface)
    answer "bar_plot" if has in question (bar, cột, biểu đồ cột, biểu đồ thanh)
                     OR if the question asks about province/city frequency or ranking, like:
                     "tỉnh nào xuất hiện nhiều nhất", "tỉnh nào có nhiều nhất", "thành phố nào nhiều nhất",
                     "tỉnh nào ít nhất", "phân bổ theo tỉnh", "bao nhiêu bất động sản ở mỗi tỉnh",
                     "top tỉnh thành", "xếp hạng tỉnh", "province nào", "cái nào nhiều nhất/ít nhất",
                     "tỉnh/thành nào chiếm nhiều nhất/ít nhất", "xếp hạng", "top", "so sánh",
                     "cao nhất", "thấp nhất", "nhiều nhất", "ít nhất", "loại nào", "khu vực nào",
                     "nơi nào", "which ... is the highest/most/least/lowest/top/bottom"
    answer "line_plot" if has in question (line, đường, biểu đồ đường, xu hướng, theo thời gian)
    answer "histogram_plot" if has in question (histogram, phân phối, tần suất, tần số, phân bố)
    answer "pie_plot" if has in question (pie, tròn, biểu đồ tròn, tỷ lệ phần trăm, cơ cấu)
    answer "box_plot" if has in question (box, hộp, biểu đồ hộp, tứ phân vị, box plot)
    answer "area_plot" if has in question (area, diện tích, biểu đồ vùng, miền)
    answer "choroplethmap_plot" if has in question (bản đồ choropleth, bản đồ tô màu, choropleth)
    answer "densitymap_plot" if has in question (bản đồ mật độ, density map)
    answer "scattermap_plot" if has in question (bản đồ điểm, scatter map, điểm trên bản đồ)
    answer "polar_plot" if has in question (radar, polar, biểu đồ radar, biểu đồ nhện, mạng nhện)
    answer "table" if has in question (bảng dataframe, bảng dữ liệu, hiển thị bảng)
    answer "table_plotly" if has in question (bảng plotly, bảng đồ họa, table plotly)
    answer "base_ref" if there is nothing related to the question.
    </allowed_words>

    Use context information for accuracy in the answer.
    DECLARE ONLY one WORD in response according with <allowed_words> tags.
    
    Response examples:
    question: 'vẽ biểu đồ cột theo tỉnh thành' 
    response: 'bar_plot'
    
    question: 'tỉnh nào xuất hiện nhiều nhất trong dataset?'
    response: 'bar_plot'
    
    question: 'tỉnh nào có nhiều bất động sản nhất?' 
    response: 'bar_plot'

    question: 'thống kê số lượng bất động sản theo từng tỉnh'
    response: 'bar_plot'
    
    question: 'phân phối giá nhà theo histogram'
    response: 'histogram_plot'
    
    """

    response = llm.invoke(prompt_template)
    return response.content.strip()


def process_prompt(
    session_msgs: list[dict], user_question: str, data: dict, llm: object
) -> str:
    """
    Xử lý, RAG routing và tạo final prompt.
    [NEW] Luôn chạy thêm semantic RAG từ rag_docs/ và inject <rag_docs_context>.
    """

    # ── Process data for prompt <metadata> ────────────────────────────────────
    try:
        output_parts = []
        for index, (name, df) in enumerate(data.items(), start=1):
            buffer = StringIO()
            df.info(buf=buffer)
            df_info = buffer.getvalue()
            df_sample = df.sample(st.session_state["sample_var"]).to_csv(
                path_or_buf=None, index=False
            )
            df_name = f"DF_{index}"
            output_parts.append(
                f"\n<{df_name}>\n[INFO {df_name}]:\n{df_info}[SAMPLES {df_name}]:\n{df_sample}</{df_name}>\n"
            )
        metadata = "\n".join(output_parts)
    except Exception as e:
        data = {}
        exception_name = type(e).__name__
        track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
        message_ = "WARNING! Sample data error, provided only columns as samples"
        st.error(f"{message_}  <{exception_name}: {track_line}>")
        metadata = str(df.columns)

    # ── Code in context ────────────────────────────────────────────────────────
    if st.session_state["context_code_var"]:
        st.session_state["context_code_var"] = False
        context_code = st.session_state["last_code"]
    else:
        context_code = "No code returned for context."

    user_questions = [item for item in session_msgs if item["role"] == "user"]
    questions_text = "\n".join([item["question"] for item in user_questions])

    # ── RAG routing (ragdata/ - poor man's RAG, giữ nguyên) ───────────────────
    graph_type = define_graph_type(llm, user_question, questions_text)
    dir_path = "./ragdata"
    files = os.listdir("./ragdata")
    for file_ in files:
        file_name, file_ext = os.path.splitext(file_)
        if file_name == graph_type and file_ext == ".txt":
            file_path = os.path.join(dir_path, file_)
            with open(file_path, "r") as code:
                params_plot = f"\n\n<code_ref>\n{code.read()}\n</code_ref>"
            break
        else:
            with open("ragdata/base_ref.txt") as code:
                params_plot = f"\n\n<code_ref>\n{code.read()}\n</code_ref>"

    # ── [NEW] Semantic RAG từ rag_docs/ ───────────────────────────────────────
    try:
        rag_docs_context = build_rag_context(
            user_question, top_k=3, min_similarity=0.25
        )
        if rag_docs_context:
            print(
                f"\n{Fore.LIGHTMAGENTA_EX}[RAGDocs] Context injected from rag_docs/{Fore.RESET}"
            )
        else:
            print(
                f"\n{Fore.LIGHTBLACK_EX}[RAGDocs] No relevant docs found (similarity < 0.25){Fore.RESET}"
            )
    except Exception as e:
        rag_docs_context = ""
        print(f"\n{Fore.LIGHTRED_EX}[RAGDocs] Error: {e}{Fore.RESET}")
    # ─────────────────────────────────────────────────────────────────────────

    price_driver_keywords = [
        "ảnh hưởng",
        "tác động",
        "liệu",
        "có luôn",
        "nhà to hơn",
        "diện tích",
        "phòng ngủ",
        "phòng tắm",
        "số tầng",
        "mặt tiền",
    ]
    q_lower = user_question.lower()
    is_price_driver_question = any(kw in q_lower for kw in price_driver_keywords) and (
        "giá" in q_lower or "price" in q_lower
    )

    # ── Xác định loại kết quả trả về ──────────────────────────────────────────
    if graph_type in [
        "scatter_2d_plot",
        "bubble_plot",
        "scatter_3d_plot",
        "bar_plot",
        "line_plot",
        "histogram_plot",
        "pie_plot",
        "box_plot",
        "area_plot",
        "choroplethmap_plot",
        "densitymap_plot",
        "scattermap_plot",
        "table_plotly",
        "polar_plot",
        "surface_plot",
        "heatmap",
        "candle_plot",
        "violin_plot",
        "density_contour_plot",
        "sunburst_plot",
        "treemap_plot",
    ]:
        result_instruction = """
# For plot-type answers:
# - Create the Plotly figure (store it in a variable named `fig`).
# - Create a short Vietnamese analysis/insight string in a variable named `analysis`.
# - The `analysis` MUST reference the computed values used in the chart (e.g., top-1 entity, highest/lowest value, or main comparison).
# - Set `result = {"figure": fig, "analysis": analysis}`.
# Do NOT set `result` to only the figure.
"""
        lang_code = st.session_state.get("lang_var", "en")
        lang_for_text = "Vietnamese" if lang_code == "vi" else lang_code

        # ── [PROVINCE RANKING] Inject thêm hướng dẫn khi câu hỏi về tỉnh ────
        province_hint = ""
        province_keywords = [
            "tỉnh nào",
            "thành phố nào",
            "province",
            "xuất hiện nhiều nhất",
            "phân bổ theo tỉnh",
            "bao nhiêu bất động sản",
            "top tỉnh",
            "xếp hạng tỉnh",
        ]
        if any(kw in user_question.lower() for kw in province_keywords):
            province_hint = """
        # PROVINCE RANKING HINT:
        # Câu hỏi này liên quan đến đếm số lượng bất động sản theo tỉnh/thành.
        # Cột cần dùng: 'Province' (tên tỉnh/thành phố).
        # Công thức chuẩn:
        #   province_counts = df['Province'].value_counts().reset_index()
        #   province_counts.columns = ['Province', 'Count']
        #   province_counts['Percentage'] = (province_counts['Count'] / len(df) * 100).round(2)
        # Vẽ bar chart ngang (orientation='h') sắp xếp từ cao xuống thấp.
        # Hiển thị text count trên từng bar.
"""

        prompt_context = f"""
        {province_hint}
        {"""
        # PRICE DRIVER HINT:
        # Câu hỏi này yêu cầu phân tích mức độ ảnh hưởng của diện tích/cấu trúc đến giá.
        # Bắt buộc nêu bằng chứng định lượng, ngoại lệ và kết luận có điều kiện.
        # Không khẳng định quan hệ nhân quả tuyệt đối, phải ghi rõ 'tương quan không đồng nghĩa nhân quả'.
        # Ưu tiên scatter (Area-Price) + so sánh nhóm theo Floors/Bedrooms/Bathrooms/Frontage.
        """ if is_price_driver_question else ""}
        For plots, ONLY use the "Plotly" library and bring fig object into the result variable.
        The template should ONLY be "plotly", when not requested.
        All texts in titles, axis titles, legends, hovers, etc., SET to language "{lang_for_text}".
        SET Legend title according to legend data, when not requested.
        On the x and y axes, place large words or numbers representing dates at 45 degrees, when not requested.
        The title color must follow the template standard, when not requested.
        DEFINE text in "xaxis_title" or "yaxis_title" according to <main_question>, removing symbols and special characters.
        DECLARE the chart title according to <main_question>.
        Follow the underlying tags below EXACTLY, ALWAYS guided by the <main_question> tag.
        SET the update_traces() and update_layout() configs BASED in context within the <code_ref> tags for <main_question>.
        {params_plot}"""

    elif graph_type == "table":
        result_instruction = """
# For table requests:
# - Return ONLY a pandas dataframe in `result`.
"""
        prompt_context = """
        Any table or list request by the <main_question> tag, the response must ONLY be a return with
        dataframe and NEVER a graph.
        """
    else:
        result_instruction = """
# For non-plot requests:
# - Set `result` to the final answer text (string) in Vietnamese.
"""
        with open("ragdata/base_ref.txt") as code:
            params_plot = f"\n\n<code_ref>\n{code.read()}\n</code_ref>"
        prompt_context = f"""
        For plots, ONLY use the "Plotly" library and bring fig object into the result variable.
        {params_plot}"""

    print(f"\n\n{Fore.LIGHTGREEN_EX}STARTING RUNTIME...{Fore.RESET}")
    print(f"\n{Fore.LIGHTBLUE_EX}GRAPH TYPE BASE:{Fore.RESET} {graph_type}")

    # ── Language instruction ───────────────────────────────────────────────────
    if st.session_state["lang_var"] == "en":
        lang = "Respond ONLY in <en> language. Configure all graphics parameters for the <en> language."
    elif st.session_state["lang_var"] == "pt-BR":
        lang = "Respond ONLY in <pt-BR> language. Configure all graphics parameters for the <pt-BR> language."
    else:
        lang = "Respond ONLY in Vietnamese language. Configure all graphics parameters for Vietnamese language."

    # ── Final prompt (rag_docs_context inject ngay sau metadata) ──────────────
    prompt_main = f"""
    
        <metadata>
        {metadata}
        </metadata>
        {rag_docs_context}
        
        DEFINE DF_* according to the <main_question> using the variables already declared <DF_1, DF_2, DF_*, ...>. 
        A priori assumes DF as DF_1.
        
        ```python
        # TODO: import the necessary dependencies.
        import pandas as pd 
        ...
            
        df = pd.DataFrame(DF_*)
            
        # Complete your code here.
        ...
        
        # bring the result here.
        {result_instruction}
        result = None
        ```
    
    
        Answer the <main_question> tag concisely and accurately. For greater accuracy in your answers, follow the context EXACTLY
        provided by the <guidelines> and <last_code> tags, always driven by the <main_question> tag request and the data
        in the <metadata> tag. Answers with numbers ONLY with two decimal places. In case of error, DEBUG the code according to
        <message_error> tag and the wrong code tag <code_error>.
        {"For price-driver questions, your final analysis MUST include: (1) quantified evidence, (2) at least one exception/counter-example, (3) a conditional conclusion that avoids absolute causality claims." if is_price_driver_question else ""}
                    
                        
        <guidelines>
        Follow the guidelines below and use the <messages_history> and <last_code> tags to improve understanding only as context:        
        {prompt_context}
        </guidelines>
            
            
        <messages_history>
        {questions_text}
        </messages_history>


        <main_question>
        {user_question}
        </main_question>
        
        
        <last_code>
        ```python
        {context_code}
        ```
        </last_code>
        


        Variable `GEOJSON: list[dict]` is already declared.

        Generate python code and return full updated code.
        
        {lang}
        
        """

    return prompt_main
