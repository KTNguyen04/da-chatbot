# AI Data Analysis Chatbot 🚀


## 🛠️ Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.11 trở lên.
- (Tùy chọn) [Ollama](https://ollama.com/) nếu muốn chạy model local.

### 2. Cài đặt môi trường

Sử dụng `uv` (khuyên dùng) hoặc `pip`:

**Dùng uv:**
```bash
# Cài đặt dependencies
uv sync
```

**Dùng pip:**
```bash
# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường

Tạo file `.env` tại thư mục gốc của dự án và thêm các API Key cần thiết:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

## 🚀 Cách chạy ứng dụng

Khởi chạy giao diện Streamlit:

```bash
streamlit run app.py
```

Sau khi chạy, truy cập địa chỉ `http://localhost:8501` trên trình duyệt.

