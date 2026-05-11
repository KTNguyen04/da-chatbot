import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time 
import traceback
from streamlit_ace import st_ace
import re
import io
from contextlib import redirect_stdout

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Chatbot Phân tích Dữ liệu", layout="wide", page_icon="🤖")

# --- KHỞI TẠO SESSION STATE ---
if 'generated_code' not in st.session_state: st.session_state.generated_code = ""
if 'original_code' not in st.session_state: st.session_state.original_code = ""
if 'ai_explanation' not in st.session_state: st.session_state.ai_explanation = ""
if 'ai_idea' not in st.session_state: st.session_state.ai_idea = ""
if 'current_intent' not in st.session_state: st.session_state.current_intent = ""
if 'table_query' not in st.session_state: st.session_state.table_query = ""
if 'suggestions' not in st.session_state: st.session_state.suggestions = []
if 'df' not in st.session_state: st.session_state.df = None
if 'api_key_valid' not in st.session_state: st.session_state.api_key_valid = None
if 'code_executed_success' not in st.session_state: st.session_state.code_executed_success = False
if 'retry_query' not in st.session_state: st.session_state.retry_query = ""

# =======================================================================
# --- QUẢN LÝ LỊCH SỬ (FILE JSON LOCAL) ---
# =======================================================================
HISTORY_FILE = "analysis_history.json"

def load_history():
    if not os.path.exists(HISTORY_FILE): return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return []

def save_to_history(data):
    history = load_history()
    history.append(data)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# =======================================================================
# --- HÀM TƯƠNG TÁC AI (GEMINI) ---
# =======================================================================
def validate_api_key(key):
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        model.generate_content("test")
        return True
    except:
        return False

def generate_ai_response_text(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    return model.generate_content(prompt).text

def generate_ai_response_stream(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt, stream=True)
    for chunk in response: yield chunk.text

FALLBACK_SUGGESTIONS = [
    {"title": "Thống kê tổng quan", "request": "Tính các chỉ số thống kê mô tả (trung bình, trung vị) cho các cột số."},
    {"title": "Kiểm tra dữ liệu thiếu", "request": "Tìm và đếm số lượng giá trị thiếu (NaN/Null) trong mỗi cột."},
    {"title": "Mối tương quan", "request": "Tính ma trận tương quan giữa các biến số và giải thích."}
]

def do_generate_suggestions(df, api_key):
    start_time = time.time()
    with st.spinner('🤖 AI đang quét dữ liệu và suy nghĩ ý tưởng...'):
        if not api_key or not st.session_state.api_key_valid:
            st.warning("⚠️ API Key chưa hợp lệ. Sử dụng gợi ý dự phòng (Fallback).")
            st.session_state.suggestions = FALLBACK_SUGGESTIONS
            return

        prompt = f"""
            Cột dữ liệu: {list(df.columns)}. Sample: {df.head(2).to_string()}
            Đề xuất 3 hướng phân tích dữ liệu thực tế. BẮT BUỘC TRẢ VỀ CHỈ JSON LIST:
            [ {{"title": "Tên gợi ý", "request": "Mô tả yêu cầu"}} ]
            """
        try:
            res_text = generate_ai_response_text(prompt, api_key)
            match = re.search(r'\[\s*{.*}\s*\]', res_text, re.DOTALL)
            if match:
                st.session_state.suggestions = json.loads(match.group(0))
                st.success(f"💡 Đã tạo gợi ý xong trong {time.time() - start_time:.1f}s")
            else:
                raise ValueError("JSON parse error")
        except Exception as e:
            st.error("⚠️ Lỗi API/Mạng. Tự động dùng Gợi ý dự phòng (Fallback).")
            st.session_state.suggestions = FALLBACK_SUGGESTIONS

# =======================================================================
# --- DIALOG HIỂN THỊ LỊCH SỬ (MODAL MỜ) ---
# =======================================================================
@st.dialog("📜 Xem & Chỉnh sửa Lịch sử", width="large")
def show_history_dialog(entry):
    st.markdown(f"**Yêu cầu:** {entry['question']}")
    
    edited_hist_code = st_ace(
        value=entry.get('code', ''), language="python", theme="github", 
        keybinding="vscode", font_size=14, min_lines=15, key=f"ace_{entry['time']}"
    )
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("💾 Lưu cập nhật code vào Log", use_container_width=True):
            if edited_hist_code.strip() == entry.get('code', '').strip():
                st.info("ℹ️ Code không có sự thay đổi nào để lưu.")
            else:
                history = load_history()
                for idx, e in enumerate(history):
                    if e['time'] == entry['time']:
                        history[idx]['code'] = edited_hist_code
                        break
                with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=4)
                st.success("✅ Đã cập nhật thành công!")
    with col_b:
        if st.button("🚀 Thực thi lại mã nguồn", use_container_width=True):
            try:
                plt.clf(); plt.close('all')
                exec(edited_hist_code, globals(), {'df': st.session_state.df, 'plt': plt, 'sns': sns, 'pd': pd, 'st': st})
                fig = plt.gcf()
                if len(fig.get_axes()) > 0: st.pyplot(fig)
            except Exception as e:
                st.error(f"⚠️ Lỗi: {str(e)}")

# =======================================================================
# --- SIDEBAR ---
# =======================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    with st.expander("🔑 Hướng dẫn lấy API Key"):
        st.markdown("1. Vào [Google AI Studio](https://aistudio.google.com/app/apikey).\n2. Bấm **Create API key**.\n3. Dán vào ô bên dưới.")
    
    api_key_input = st.text_input("Nhập Gemini API Key (Bấm Enter):", type="password")
    
    # Xác thực Key & Hiển thị thông báo biến mất
    if api_key_input != st.session_state.get('last_key', ''):
        st.session_state.last_key = api_key_input
        if api_key_input:
            is_valid = validate_api_key(api_key_input)
            st.session_state.api_key_valid = is_valid
            if is_valid:
                msg = st.empty()
                msg.success("✅ Hợp lệ!")
                time.sleep(2)
                msg.empty()

    if st.session_state.api_key_valid is False and api_key_input:
        st.error("❌ API key không hợp lệ")
    elif not api_key_input:
        st.warning("⚠️ Hãy nhập API Key để bắt đầu.")

    st.divider()
    st.header("📂 Tải Dữ liệu")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success(f"✅ Đã tải: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

# =======================================================================
# --- GIAO DIỆN CHÍNH (NẾU CÓ DATA) ---
# =======================================================================
if st.session_state.df is not None:
    df = st.session_state.df
    st.title("🤖 Chatbot Phân tích Dữ liệu")

    with st.expander("📖 Hướng dẫn sử dụng ứng dụng"):
        st.markdown("""
        **Chào mừng bạn đến với Chatbot Phân tích Dữ liệu!** Dưới đây là luồng thao tác cơ bản:

        1. **Tải dữ liệu:** Nhấp vào nút tải file ở thanh công cụ bên trái (Sidebar) để tải lên tập dữ liệu `.csv` của bạn.
        
        2. **Khám phá ý tưởng (Tùy chọn):** Nếu chưa có hướng phân tích, hãy nhấp vào nút **"💡 Mở Gợi ý Phân tích (AI)"** (ở Sidebar hoặc dưới đáy màn hình). AI sẽ quét file và đề xuất các ý tưởng cho bạn.
        
        3. **Tương tác & Nhập yêu cầu:** Nhập yêu cầu vào ô chat (Ví dụ: *"Vẽ biểu đồ giá nhà theo quận..."* hoặc *"Lọc ra các nhà có 3 phòng ngủ..."*). Sau đó nhấn **Enter** hoặc nút **🚀 AI Lập trình** để hệ thống xử lý.
        
        4. **Xử lý Kết quả & Chỉnh sửa:**
           - **Tra cứu / Lọc dữ liệu:** AI sẽ trả về câu giải thích dạng văn bản kèm theo bảng số liệu.
           - **Vẽ biểu đồ:** AI sẽ sinh ra mã nguồn Python. Bạn có thể trực tiếp chỉnh sửa các thông số (màu sắc, font chữ...) trong khung code và nhấn **✅ Chấp nhận & Thực thi trên Local** để vẽ biểu đồ.

        5. **Các công cụ tiện ích khác:**
           - **🧪 Môi trường Test Code:** Bật tính năng này để có một không gian nháp riêng. Bạn có thể tự do gõ code, in (`print`) kết quả hoặc vẽ thử biểu đồ độc lập với luồng chat.
           - **📜 Lịch sử phân tích:** Cho phép bạn xem lại các phiên làm việc cũ. Tại đây, bạn có thể chạy lại code cũ, sửa code và nhấn **💾 Lưu cập nhật code vào Log** để lưu lại những thay đổi mới.
        """)

    with st.expander("🔍 Xem dữ liệu gốc"):
            if st.checkbox("Hiển thị tất cả dữ liệu (⚠️ Cảnh báo: Có thể lag do dữ liệu lớn)"):
                st.dataframe(df)
            else:
                st.write("Đang hiển thị 10 dòng đầu tiên:")
                st.dataframe(df.head(10))
    # Form nhận yêu cầu
    data_context = f"Cột dữ liệu: {list(df.columns)}. Sample:\n{df.head(2).to_string()}"
    
    # Nổi bật ô nhập liệu
    st.markdown("### 💬 Nhập yêu cầu phân tích:")
    user_input = st.text_input(
        "Nhập câu hỏi của bạn tại đây:", 
        value=st.session_state.retry_query if st.session_state.retry_query else st.session_state.get('active_query', ''),
        placeholder="Ví dụ: Vẽ biểu đồ nhiệt thể hiện tương quan giữa biến giá nhà và diện tích",
        label_visibility="collapsed"
    )
    
    col_run, col_sandbox = st.columns([0.8, 0.2])
    with col_run:
        run_button = st.button("🚀 AI Lập trình", type="primary", use_container_width=True)
    with col_sandbox:
        toggle_sandbox = st.toggle("🧪 Bật ô Test Code")

    # Môi trường Sandbox (Test code tự do)
    if toggle_sandbox:
        st.info("🧪 **Môi trường Test Code:** Bạn có thể tự code và chạy thử tại đây. Hỗ trợ in text (`print`), hiển thị biểu đồ và bảng dữ liệu.")
        sandbox_code = st_ace(language="python", theme="github", font_size=14, min_lines=5, key="sandbox")
        
        if st.button("Chạy thử Code"):
            # Tạo một biến để "hứng" toàn bộ dữ liệu in ra từ hàm print()
            f = io.StringIO()
            
            with redirect_stdout(f): # Bắt đầu hứng output
                try:
                    plt.clf(); plt.close('all')
                    exec(sandbox_code, globals(), {'df': df, 'plt': plt, 'sns': sns, 'pd': pd, 'st': st})
                    
                    # 1. Xử lý và in ra Output Text (Terminal)
                    printed_output = f.getvalue()
                    if printed_output.strip():
                        st.markdown("**🖨️ Terminal Output:**")
                        st.code(printed_output, language="text")
                    
                    # 2. Xử lý và in ra Output Biểu đồ
                    fig = plt.gcf()
                    if len(fig.get_axes()) > 0: 
                        st.markdown("**📊 Biểu đồ Output:**")
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"⚠️ Lỗi thực thi Sandbox: {e}")

    # XỬ LÝ YÊU CẦU AI
    if st.session_state.get('auto_run'):
        run_button = True
        st.session_state.auto_run = False
        st.session_state.active_query = ""

    if run_button:
        st.session_state.code_executed_success = False # Reset trạng thái biểu đồ
        st.session_state.retry_query = ""

        if not st.session_state.api_key_valid:
            st.error("⚠️ Bạn cần nhập API Key hợp lệ ở thanh bên trái!")
        elif not user_input.strip():
            st.warning("⚠️ Chưa nhập yêu cầu. Vui lòng nhập để tiếp tục.")
        else:
            prompt = f"""
            Bạn là AI phân tích dữ liệu. Biến DataFrame thực tế là 'df'.
            NGỮ CẢNH: {data_context}
            YÊU CẦU: {user_input}

            QUY TẮC NGHIÊM NGẶT:
            1. KHÔNG bịa dữ liệu. KHÔNG lấy hình từ mạng.
            2. CODE PHẢI COMMENT TIẾNG VIỆT RÕ RÀNG.
            3. KHÔNG DÙNG plt.show(). Chỉ gán vào ax/fig.

            TRẢ VỀ DUY NHẤT 1 BLOCK JSON NHƯ SAU:
            {{
                "intent": "CHART" hoặc "TABLE" hoặc "TEXT",
                "idea": "Ý tưởng triển khai ngắn gọn (1-2 câu)",
                "explanation": "Giải thích kết quả phân tích bằng text chi tiết",
                "code": "Mã python vẽ biểu đồ (chỉ khi intent=CHART)",
                "table_query": "Chuỗi mã python lọc df (chỉ khi intent=TABLE)"
            }}
            """
            
            with st.chat_message("assistant"):
                start_time = time.time()
                placeholder = st.empty()
                full_response = ""
                
                try:
                    for chunk in generate_ai_response_stream(prompt, api_key_input):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                    
                    json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
                    if json_match:
                        res_json = json.loads(json_match.group(0))
                        st.session_state.current_intent = res_json.get("intent", "TEXT")
                        st.session_state.ai_idea = res_json.get("idea", "")
                        st.session_state.ai_explanation = res_json.get("explanation", "")
                        st.session_state.table_query = res_json.get("table_query", "")
                        
                        code = res_json.get("code", "")
                        if code:
                            st.session_state.generated_code = code
                            st.session_state.original_code = code

                        save_to_history({
                            "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "question": user_input, "code": code, "explanation": st.session_state.ai_explanation
                        })
                        st.success(f"⏱ Hoàn thành trong {time.time() - start_time:.1f}s")
                        st.rerun()
                    else:
                        raise ValueError("Invalid JSON")
                except Exception as e:
                    st.error(f"⚠️ Lỗi hệ thống: {str(e)}")
                    st.session_state.retry_query = user_input
                    if st.button("🔄 Thử lại (Retry)"):
                        st.rerun()

    # TRÌNH BÀY KẾT QUẢ THEO INTENT
    intent = st.session_state.get('current_intent', '')

    if intent == "TEXT":
        st.divider()
        st.info(f"💡 **AI Ý tưởng:** {st.session_state.ai_idea}")
        st.success(f"💬 **AI Giải thích:**\n{st.session_state.ai_explanation}")

    elif intent == "TABLE":
        st.divider()
        st.info(f"💡 **AI Ý tưởng:** {st.session_state.ai_idea}")
        try:
            result_df = eval(st.session_state.table_query, {"df": df, "pd": pd})
            st.dataframe(result_df)
            st.success(f"💬 **AI Giải thích:**\n{st.session_state.ai_explanation}")
        except Exception as e:
            st.error(f"⚠️ Lỗi trích xuất bảng: {str(e)}")

    elif intent == "CHART" and st.session_state.generated_code:
        st.divider()
        st.info(f"💡 **AI Ý tưởng:** {st.session_state.ai_idea}")
        
        # Thao tác Mã nguồn (Gọn gàng trên 1 dòng)
        st.write("### 📝 Trạng thái chờ: Mã nguồn")
        action = st.radio("Thao tác mã nguồn:", ["Chỉnh sửa (Mặc định)", "Copy Code", "Xóa toàn bộ", "Reset về ban đầu"], horizontal=True, label_visibility="collapsed")
        
        if action == "Copy Code":
            st.code(st.session_state.generated_code, language="python")
        elif action == "Xóa toàn bộ":
            if st.button("⚠️ Xác nhận Xóa"): 
                st.session_state.generated_code = ""
                st.rerun()
        elif action == "Reset về ban đầu":
            if st.button("🔄 Xác nhận Reset"):
                st.session_state.generated_code = st.session_state.original_code
                st.rerun()

        edited_code = st_ace(
            value=st.session_state.generated_code, language="python", theme="github", 
            keybinding="vscode", font_size=14, min_lines=15, key="main_ace"
        )
        
        if st.button("✅ Chấp nhận & Thực thi trên Local", type="primary"):
            st.session_state.generated_code = edited_code # Cập nhật code mới nhất
            try:
                plt.clf(); plt.close('all')
                exec(edited_code, globals(), {'df': df, 'plt': plt, 'sns': sns, 'pd': pd, 'st': st})
                fig = plt.gcf() 
                if len(fig.get_axes()) > 0:
                    st.pyplot(fig)
                    st.session_state.code_executed_success = True
                else:
                    st.warning("Code chạy xong nhưng không vẽ biểu đồ.")
            except Exception as e:
                st.session_state.code_executed_success = False
                error_msg = traceback.format_exc()
                st.error(f"⚠️ Đã xảy ra lỗi thực thi:\n```\n{str(e)}\n```")
                
                # Tính năng Auto-Fix
                if st.button("✨ Nhờ AI sửa lỗi giúp"):
                    with st.spinner("AI đang tìm lỗi và viết lại code..."):
                        fix_prompt = f"Code Python bị lỗi:\n{edited_code}\nLỗi Traceback:\n{error_msg}\nSửa lỗi và TRẢ VỀ DUY NHẤT ĐOẠN CODE PYTHON (bắt đầu bằng comment ### FIXED: [lý do sửa])."
                        try:
                            fixed_code_raw = generate_ai_response_text(fix_prompt, api_key_input)
                            cleaned_fix = re.sub(r"```python|```", "", fixed_code_raw).strip()
                            st.session_state.generated_code = cleaned_fix
                            st.success("✅ Đã sửa lỗi! Vui lòng ấn Thực thi lại.")
                            time.sleep(1.5)
                            st.rerun()
                        except Exception as ex:
                            st.error("AI không thể tự sửa lỗi lúc này.")

        # CHỈ HIỂN THỊ KHI CHẠY CODE THÀNH CÔNG (Giải thích & Bộ lọc)
        if st.session_state.get('code_executed_success'):
            st.success(f"💬 **AI Giải thích kết quả:**\n{st.session_state.ai_explanation}")
            
            with st.expander("🎨 Tùy chỉnh Biểu đồ (Không cần Code)"):
                st.info("Chỉnh sửa các thông số dưới đây và ấn 'Cập nhật' để vẽ lại.")
                c_t1, c_t2, c_t3 = st.columns(3)
                custom_title = c_t1.text_input("Đổi Tên biểu đồ:", value="")
                custom_xl = c_t2.text_input("Đổi Tên trục X:", value="")
                custom_yl = c_t3.text_input("Đổi Tên trục Y:", value="")
                
                if st.button("🔄 Cập nhật biểu đồ mới"):
                    plt.clf(); plt.close('all')
                    exec(st.session_state.generated_code, globals(), {'df': df, 'plt': plt, 'sns': sns, 'pd': pd, 'st': st})
                    fig = plt.gcf()
                    ax = fig.gca()
                    if custom_title: ax.set_title(custom_title)
                    if custom_xl: ax.set_xlabel(custom_xl)
                    if custom_yl: ax.set_ylabel(custom_yl)
                    st.pyplot(fig)

else:
    # Nếu chưa up data, hiện thông báo mờ
    st.title("🤖 Chatbot Phân tích Dữ liệu")
    st.info("👋 Chào mừng! Vui lòng tải lên file CSV ở thanh bên trái (Sidebar) để bắt đầu phân tích.")

# =======================================================================
# --- FOOTER: QUICK ACCESS (LUÔN HIỂN THỊ) ---
# =======================================================================
st.divider()
st.markdown("### 📌 Quick Access (Truy cập nhanh)")

f_col1, f_col2 = st.columns(2)
with f_col1:
    if st.button("💡 Mở Gợi ý Phân tích (AI)", use_container_width=True):
        st.session_state.show_b_sug = not st.session_state.get('show_b_sug', False)
        st.session_state.show_b_hist = False 
        st.rerun()
with f_col2:
    if st.button("📜 Mở Lịch sử Phân tích", use_container_width=True):
        st.session_state.show_b_hist = not st.session_state.get('show_b_hist', False)
        st.session_state.show_b_sug = False 
        st.rerun()

if st.session_state.get('show_b_sug'):
    with st.container(border=True):
        if st.session_state.df is None:
            st.warning("⚠️ Vui lòng tải lên file dữ liệu CSV ở Sidebar trước khi lấy gợi ý.")
        else:
            if not st.session_state.suggestions:
                if st.button("🚀 Tạo gợi ý mới ngay", key="btn_f_sug"):
                    do_generate_suggestions(st.session_state.df, st.session_state.get('last_key', ''))
                    st.rerun()
            else:
                for i, item in enumerate(st.session_state.suggestions):
                    if st.button(f"👉 {item['title']}", key=f"f_sug_{i}", use_container_width=True):
                        st.session_state.active_query = item['request']
                        st.session_state.auto_run = True 
                        st.session_state.show_b_sug = False
                        st.rerun()
                if st.button("🔄 Làm mới gợi ý"):
                    do_generate_suggestions(st.session_state.df, st.session_state.get('last_key', ''))
                    st.rerun()

if st.session_state.get('show_b_hist'):
    with st.container(border=True):
        hist = load_history()
        if not hist: st.info("Chưa có lịch sử.")
        else:
            for entry in list(reversed(hist))[:5]: 
                if st.button(f"🕒 {entry['time']} | {entry['question'][:50]}...", key=f"f_hist_{entry['time']}", use_container_width=True):
                    if st.session_state.df is None:
                        st.warning("⚠️ Vui lòng tải file dữ liệu lên trước khi xem chi tiết code lịch sử.")
                    else:
                        show_history_dialog(entry) # Mở Modal Dialog