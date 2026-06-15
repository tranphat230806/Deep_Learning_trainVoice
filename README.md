<div align="center">

# SmartHome Hybrid AI (Voice Recognition & Control)

Hệ thống điều khiển nhà thông minh kết hợp đa mô hình AI: **Xác thực Sinh trắc học Giọng nói**, **Điều khiển cục bộ (Local)** bằng sóng âm, và **Quản gia ảo (Cloud)** với khả năng hiểu ngôn ngữ tự nhiên.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![SpeechBrain](https://img.shields.io/badge/AI-SpeechBrain-FF5A5F?logo=ai&logoColor=white)](https://speechbrain.github.io/)
[![HuggingFace](https://img.shields.io/badge/Model-PhoWhisper-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_Pro-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![HTML5](https://img.shields.io/badge/Frontend-HTML/CSS/JS-E34F26?logo=html5&logoColor=white)](https://developer.mozilla.org/)

</div>

---

## 🌟 Kiến trúc hệ thống (Hybrid AI)

Dự án này sử dụng mô hình AI Lai (Hybrid AI) để tối ưu hoá tốc độ và bảo mật cho Smart Home:

1. **Xác thực Giọng nói (Voice Biometrics)**
   - Sử dụng **SpeechBrain** (mô hình `spkrec-ecapa-voxceleb`) để trích xuất *speaker embedding* từ giọng nói.
   - So sánh Cosine Similarity với giọng chủ nhà được thu sẵn (`my_voice/a1.wav`).
   - Nếu độ khớp >= 80%, hệ thống mở khóa (Phase 2).

2. **Điều khiển Cục bộ (Local AI - Tốc độ cao)**
   - Sử dụng mạng Neural Network **LSTM** nội bộ.
   - Trích xuất đặc trưng **MFCC** bằng `librosa`.
   - Lắng nghe liên tục, nhận diện các lệnh ngắn (Up/Down/On/Off) để điều khiển cửa, đèn, quạt, máy lạnh siêu tốc (< 1s) mà không cần Internet.

3. **Quản gia ảo (Cloud AI - Hiểu ngữ nghĩa)**
   - Chuyển đổi giọng nói tiếng Việt sang văn bản (STT) siêu nhanh qua **PhoWhisper** chạy nội bộ.
   - Gửi văn bản này cho **Google Gemini 1.5 Flash** để phân tích ngữ nghĩa (NLP).
   - Gemini trả về JSON chứa cấu trúc phần cứng cần điều khiển (ví dụ: `{"device": "ac", "action": "on"}`) kèm câu phản hồi tự nhiên.

---

## 📂 Cấu trúc thư mục

- `Backend/`
  - `app_smarthome.py`: File chính chạy Flask Server (API), tải tất cả các mô hình AI.
  - `models/`: Thư mục chứa các weights AI.
    - `LSTM_Advanced/`: File pth và mapping label cho mô hình LSTM Local.
    - `spkrec_model/`: Mô hình nhận diện giọng nói của SpeechBrain.
    - `whisper-small/`: Mô hình PhoWhisper (đã được tải về máy).
  - `my_voice/`: Chứa file `a1.wav` lưu trữ mẫu giọng chủ nhà để đối chiếu.
- `Frontend/`
  - `home.html`: Giao diện Smart Home trực quan (Vanilla HTML/CSS/JS kết hợp TailwindCSS từ CDN), có mô phỏng phòng khách 3D và giao diện Chat.
  - `dashboard.html`: Giao diện báo cáo kết quả đánh giá mô hình.
- `.env`: File chứa API Key của Google Gemini.

---

## 🚀 Hướng dẫn cài đặt & Chạy dự án

### 1) Thiết lập Backend (Flask API)

> API mặc định chạy tại `http://localhost:5002`.

Tạo môi trường ảo và cài đặt các thư viện cần thiết:

```bash
cd Backend
python -m venv venv
venv\Scripts\activate

# Cài đặt thư viện
pip install flask flask-cors python-dotenv torch torchvision torchaudio numpy librosa soundfile transformers speechbrain google-genai google-generativeai
```

**Cấu hình biến môi trường:**
Tạo file `.env` ở trong thư mục `Backend/` với nội dung:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Chạy Server:**
```bash
python app_smarthome.py
```
> **Lưu ý:** Lần đầu chạy, hệ thống sẽ tự động tải các mô hình `SpeechBrain` và `PhoWhisper` từ Hugging Face nếu chúng chưa có trong thư mục `models/`. Xin kiên nhẫn.

### 2) Khởi động Frontend (Giao diện)

Dự án Frontend được code thuần (Vanilla) nên bạn **không cần cài đặt Node.js hay build code**. 
Chỉ cần mở file `Frontend/home.html` bằng trình duyệt (hoặc dùng tiện ích *Live Server* trong VSCode) là có thể sử dụng ngay.

---

## 🔌 Danh sách API chính

- `POST /api/verify_voice`: 
  - Input: FormData chứa file audio (key `audio`).
  - Logic: SpeechBrain quét và đối chiếu với file gốc `my_voice/a1.wav`.
  - Output: `{"match": true/false, "score": 85}`

- `POST /api/local_command`:
  - Input: FormData chứa file audio (key `audio`).
  - Logic: Trích xuất MFCC, LSTM dự đoán lệnh ngắn.
  - Output: `{"status": "success", "intent": "on", "confidence": 95.5}`

- `POST /api/cloud_command`:
  - Input: FormData chứa file audio (key `audio`).
  - Logic: PhoWhisper (Vi -> En) -> Text -> Gemini (LLM) -> JSON.
  - Output: `{"status": "success", "transcription": "...", "ai_response": {"hardware_command": {...}, "speech_reply": "..."}}`

---

## ⚠️ Ghi chú quan trọng

- **Lỗi `WinError 1314` khi nạp SpeechBrain**: Dự án đã được fix bằng cách ép sử dụng `LocalStrategy.COPY` thay vì Symlink mặc định, không còn yêu cầu quyền Admin trên Windows.
- **Microphone**: Frontend bắt buộc phải chạy qua giao thức `http://localhost` hoặc `https://` thì trình duyệt mới cấp quyền truy cập Microphone.

---

## 📝 Giấy phép

Dự án phục vụ mục đích học tập và nghiên cứu mô hình Deep Learning trong điều khiển thiết bị IoT (Smart Home).
