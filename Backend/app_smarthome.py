import os
import tempfile
import numpy as np
import soundfile as sf
import librosa
import google.generativeai as genai
# =================================================================
# 💉 BƯỚC LÀM NÓNG (WARM-UP): KÍCH HOẠT LIBROSA TRƯỚC KHI GỌI SPEECHBRAIN
# (Cách này giải quyết triệt để 100% mọi lỗi k2, flair, wordemb...)
# =================================================================
print("⏳ Đang làm nóng hệ thống âm thanh (Bypass Windows Bug)...")
_temp_dir = tempfile.gettempdir()
_dummy_path = os.path.join(_temp_dir, "dummy_warmup.wav")
# 1. Tạo một đoạn âm thanh im lặng dài 10 mini-giây
sf.write(_dummy_path, np.zeros(10), 16000, subtype='PCM_16')
# 2. Ép librosa đọc file này để nó chạy hết các hàm quét hệ thống ngay bây giờ
_ = librosa.load(_dummy_path, sr=16000)
# 3. Xóa file rác
os.remove(_dummy_path)
print("✅ Librosa đã an toàn! Bắt đầu nạp AI...")
# =================================================================

# BÂY GIỜ MỚI ĐƯỢC PHÉP NẠP SPEECHBRAIN (Nó sẽ không thể cắn librosa được nữa)
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from speechbrain.inference.speaker import SpeakerRecognition

app = Flask(__name__)
CORS(app)

models = {}
LOCAL_PHOWHISPER_PATH = "models/PhoWhisper-medium"

def load_smarthome_models():
    print(f"⏳ Đang nạp PhoWhisper cục bộ từ thư mục: {LOCAL_PHOWHISPER_PATH}...")
    models['w_vi_proc'] = WhisperProcessor.from_pretrained(LOCAL_PHOWHISPER_PATH)
    models['w_vi_model'] = WhisperForConditionalGeneration.from_pretrained(LOCAL_PHOWHISPER_PATH)
    models['w_vi_model'].eval()

    print("⏳ Đang nạp hệ thống Sinh trắc học Giọng nói (SpeechBrain)...")
    models['voice_auth'] = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="models/spkrec_model" 
    )
    print("✅ Đã kích hoạt toàn bộ AI cho Smart Home!")

# ==========================================
# API 1: XÁC THỰC GIỌNG NÓI CHỦ NHÂN
# ==========================================
@app.route('/api/verify_voice', methods=['POST'])
def verify_voice():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio"}), 400
        
    audio_file = request.files['audio']
    temp_dir = tempfile.gettempdir()
    
    raw_path = os.path.join(temp_dir, "raw_auth.webm")
    audio_file.save(raw_path)
    
    # --- FILE GIỌNG NÓI CHỦ NHÂN ---
    # --- ĐỔI THÀNH FILE WAV ĐỂ WINDOWS ĐỌC MƯỢT NHẤT ---
    master_voice_path = "my_voice/a1.wav" 
    
    try:
        # 1. ĐỌC ÂM THANH BẰNG LIBROSA
        y_user, _ = librosa.load(raw_path, sr=16000)
        y_master, _ = librosa.load(master_voice_path, sr=16000)
        
        # 2. CHUYỂN THÀNH TENSOR VÀ ĐẨY LÊN CÙNG THIẾT BỊ VỚI MODEL (CPU/GPU)
        # Lấy địa chỉ phần cứng mà AI đang nằm (ví dụ: 'cuda:0' hoặc 'cpu')
        ai_device = models['voice_auth'].device 
        
        # Thêm .to(ai_device) để ship dữ liệu từ CPU lên GPU
        tensor_user = torch.tensor(y_user).unsqueeze(0).float().to(ai_device)
        tensor_master = torch.tensor(y_master).unsqueeze(0).float().to(ai_device)
        
        # 3. TRÍCH XUẤT TRỰC TIẾP TỪ CÁC LỚP MẠNG
        with torch.no_grad():
            f_user = models['voice_auth'].mods.compute_features(tensor_user)
            f_user = models['voice_auth'].mods.mean_var_norm(f_user, torch.ones(f_user.shape[0]).to(ai_device))
            emb_user = models['voice_auth'].mods.embedding_model(f_user)
            
            f_master = models['voice_auth'].mods.compute_features(tensor_master)
            f_master = models['voice_auth'].mods.mean_var_norm(f_master, torch.ones(f_master.shape[0]).to(ai_device))
            emb_master = models['voice_auth'].mods.embedding_model(f_master)
        
        # 4. TỰ TOÁN HỌC ĐO ĐỘ KHỚP (COSINE SIMILARITY)
        import torch.nn.functional as F
        score = F.cosine_similarity(emb_user.view(1, -1), emb_master.view(1, -1))
        similarity_value = score.item()
        print(f"🕵️ Điểm AI gốc (Cosine): {similarity_value:.3f}") # In ra Terminal để bạn theo dõi
        
        percent_score = int(max(0, min(100, (similarity_value * 150) + 35)))
        is_matched = percent_score >= 75
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Lỗi xử lý âm thanh: {e}")
        return jsonify({"error": "Audio processing failed"}), 500
        
    finally:
        if os.path.exists(raw_path): 
            os.remove(raw_path)
        
    return jsonify({
        "match": is_matched,
        "score": percent_score
    })

# ==========================================
# API 2: ĐIỀU KHIỂN THIẾT BỊ BẰNG PHOWHISPER
# ==========================================
@app.route('/api/command', methods=['POST'])
def run_command():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio data"}), 400
        
    audio_file = request.files['audio']
    temp_dir = tempfile.gettempdir()
    
    # 1. Lưu đúng chuẩn đuôi webm của trình duyệt gửi lên để librosa không bị lỗi
    raw_path = os.path.join(temp_dir, "raw_command.webm")
    audio_file.save(raw_path)
    
    try:
        # 2. Đọc âm thanh bằng librosa (Tự động ép về 16000Hz cho Whisper)
        y, sr = librosa.load(raw_path, sr=16000)
        
        # 3. Trích xuất đặc trưng giọng nói (Mel Spectrogram)
        feat = models['w_vi_proc'](y, sampling_rate=sr, return_tensors="pt").input_features
        
        # 💡 SỬA LỖI CHÍNH Ở ĐÂY: Lấy địa chỉ phần cứng của Whisper và ship dữ liệu lên đó (GPU)
        whisper_device = models['w_vi_model'].device
        feat = feat.to(whisper_device)
        
        # 4. Sinh văn bản (Dịch giọng nói thành Text)
        with torch.no_grad():
            ids = models['w_vi_model'].generate(feat)
            text = models['w_vi_proc'].batch_decode(ids, skip_special_tokens=True)[0].lower()
        
        print(f"🗣️  [Lệnh thu được (Gốc)]: {text}")

        # --- BƯỚC MỚI: TẨY RỬA VĂN BẢN (NLP NORMALIZATION) ---
        # Loại bỏ các dấu chấm, phẩy mà AI tự động thêm vào cuối câu
        import re
        clean_text = re.sub(r'[^\w\s]', '', text).strip()
        print(f"🧹  [Lệnh đã làm sạch]: {clean_text}")

        # --- BƯỚC MỚI: PHÂN TÍCH ĐA LỆNH TRÊN CÙNG CÂU NÓI ---
        ACTIONS = {
            "ON": ["bật", "mở", "lên"],
            "OFF": ["tắt", "đóng"]
        }
        DEVICES = {
            "lights": ["đèn", "sáng"],
            "fan": ["quạt"],
            "ac": ["điều hòa", "điều hoà", "lạnh", "ac", "máy lạnh"],
            "tv": ["tivi", "tv", "ti vi"],
            "door": ["cửa"]
        }

        # 1. Tách chuỗi theo các liên từ/dấu câu
        seps = [" và ", " rồi ", ", ", ","]
        segments = [clean_text]
        for sep in seps:
            new_segments = []
            for seg in segments:
                new_segments.extend(seg.split(sep))
            segments = new_segments

        # 2. Tách nhỏ tiếp khi gặp động từ hành động khác để xử lý chuỗi "bật đèn tắt quạt"
        action_words = ["bật", "mở", "lên", "tắt", "đóng"]
        final_segments = []
        for seg in segments:
            words = seg.strip().split()
            current_sub = []
            for word in words:
                if word in action_words and current_sub:
                    final_segments.append(" ".join(current_sub))
                    current_sub = [word]
                else:
                    current_sub.append(word)
            if current_sub:
                final_segments.append(" ".join(current_sub))

        # 3. Phân tích hành động và thiết bị cho từng đoạn nhỏ
        commands = []
        last_action = None
        for seg in final_segments:
            seg_action = None
            for act, keywords in ACTIONS.items():
                if any(k in seg for k in keywords):
                    seg_action = act
                    break
            
            seg_devices = []
            for dev, keywords in DEVICES.items():
                if any(k in seg for k in keywords):
                    seg_devices.append(dev)

            if seg_devices:
                # Nếu không tìm thấy hành động trong đoạn hiện tại, thừa kế từ hành động trước đó
                if not seg_action:
                    seg_action = last_action
                if seg_action:
                    for d in seg_devices:
                        commands.append({"device": d, "action": seg_action})
                    last_action = seg_action
            else:
                if seg_action:
                    last_action = seg_action

        # Tạo giá trị fallback để tương thích ngược với frontend cũ
        if commands:
            device = commands[0]["device"]
            action = commands[0]["action"]
        else:
            device = "UNKNOWN"
            action = "UNKNOWN"

        return jsonify({
            "text": text,
            "device": device,
            "action": action,
            "commands": commands
        })
    except Exception as e:
        import traceback
        traceback.print_exc() # In lỗi màu đỏ ra Terminal nếu có
        print(f"❌ Lỗi Whisper: {e}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Dọn rác sau khi xử lý xong
        if os.path.exists(raw_path):
            os.remove(raw_path)

# ==========================================
# API 3: XÁC NHẬN MỞ CỬA 3D VÀ UI
# ==========================================
@app.route('/verify', methods=['POST', 'GET'])
def verify_system():
    # Trả về tín hiệu True để Frontend biết đường mở cánh cửa 3D và bật đèn
    return jsonify({"verified": True})


genai.configure(api_key="AIzaSyDHtbwJNMOrnYejcPCGGkxLXbwwCsdxyGQ")

# 💡 SỬA Ở ĐÂY: Dùng bản Flash thế hệ mới nhất từ danh sách của bạn
gemini_model = genai.GenerativeModel('gemini-3.1-flash-lite') 

# ==========================================
# API 4: TRỢ LÝ ẢO GEMINI
# ==========================================
@app.route('/api/chatbot', methods=['POST'])
def chatbot_gemini():
    data = request.json
    user_text = data.get("text", "")
    
    if not user_text:
        return jsonify({"reply": "Tôi chưa nghe rõ bạn nói gì."})
        
    try:
        prompt = f"""Bạn là trợ lý ảo nhà thông minh tên Thông. Chủ nhà là Thành. 
        QUY TẮC TỐI THƯỢNG: 
        1. BẮT BUỘC trả lời 100% bằng Tiếng Việt có dấu chuẩn xác. 
        2. TUYỆT ĐỐI KHÔNG dùng bất kỳ từ tiếng Anh nào. 
        3. Trả lời cực kỳ ngắn gọn, thân thiện, dưới 2 câu.
        Câu hỏi của chủ nhà là: {user_text}"""
        
        response = gemini_model.generate_content(prompt)
        reply_text = response.text
        
        return jsonify({"reply": reply_text})
    except Exception as e:
        print(f"Lỗi Gemini: {e}")
        return jsonify({"reply": "Xin lỗi, đường truyền đến não bộ Gemini đang bị lỗi."}), 500

# --- Kéo xuống dưới cùng sẽ là đoạn này (bạn giữ nguyên) ---
if __name__ == '__main__':
    load_smarthome_models()
    app.run(host='0.0.0.0', port=5000, debug=False)