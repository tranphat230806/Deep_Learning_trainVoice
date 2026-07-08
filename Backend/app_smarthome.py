import os
import tempfile
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import WhisperProcessor, WhisperForConditionalGeneration


load_dotenv() 

# =================================================================
# 💉 BƯỚC LÀM NÓNG (WARM-UP): KÍCH HOẠT LIBROSA TRƯỚC KHI GỌI SPEECHBRAIN
# =================================================================
print("⏳ Đang làm nóng hệ thống âm thanh (Bypass Windows Bug)...")
_temp_dir = tempfile.gettempdir()
_dummy_path = os.path.join(_temp_dir, "dummy_warmup.wav")
sf.write(_dummy_path, np.zeros(10), 16000, subtype='PCM_16')
_ = librosa.load(_dummy_path, sr=16000)  # Librosa đi tuần tra lúc này
os.remove(_dummy_path)
print("✅ Librosa đã an toàn! Bắt đầu nạp AI...")

# =================================================================
# 💡 BÂY GIỜ MỚI ĐƯỢC IMPORT SPEECHBRAIN VÀO
# =================================================================
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

# ==========================================
# KHỞI TẠO HỆ THỐNG
# ==========================================
app = Flask(__name__)
CORS(app)

models = {}
LOCAL_PHOWHISPER_PATH = "models/whisper-small"
LSTM_MODEL_DIR = "models/LSTM_Advanced"

N_MFCC = 40
MAX_LEN = 100

# ==========================================
# CẤU TRÚC MẠNG LSTM (Cho luồng Cục bộ)
# ==========================================
class SpeechLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SpeechLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(torch.mean(lstm_out, dim=1))

def load_smarthome_models():
    # 1. Nạp LSTM (Local Control)
    print("⏳ Đang nạp hệ thống Local AI (LSTM)...")
    try:
        with open(os.path.join(LSTM_MODEL_DIR, "label_map.json"), "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        models['lstm_idx_to_class'] = {v: k for k, v in class_to_idx.items()}
        
        models['lstm'] = SpeechLSTM(N_MFCC, 128, len(class_to_idx))
        models['lstm'].load_state_dict(torch.load(os.path.join(LSTM_MODEL_DIR, "speech_lstm.pth"), map_location='cpu', weights_only=True))
        models['lstm'].eval()
        print("✅ Đã nạp Local AI (LSTM) thành công!")
    except Exception as e:
        print(f"⚠️ Lỗi nạp LSTM: {e}")

    # 2. Nạp PhoWhisper
    print(f"⏳ Đang nạp PhoWhisper cục bộ từ thư mục: {LOCAL_PHOWHISPER_PATH}...")
    try:
        models['w_vi_proc'] = WhisperProcessor.from_pretrained(LOCAL_PHOWHISPER_PATH)
        models['w_vi_model'] = WhisperForConditionalGeneration.from_pretrained(LOCAL_PHOWHISPER_PATH)
        models['w_vi_model'].eval()
    except Exception as e:
         print(f"⚠️ Lỗi nạp PhoWhisper: {e}")



    # 3. Nạp SpeechBrain
    print("⏳ Đang nạp hệ thống Sinh trắc học Giọng nói (SpeechBrain)...")
    try:
        models['voice_auth'] = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="models/spkrec_model",
            local_strategy=LocalStrategy.COPY
        )
        print("✅ Đã kích hoạt toàn bộ AI cho Smart Home!")
    except Exception as e:
        print(f"⚠️ Lỗi nạp SpeechBrain: {e}")


# ==========================================
# HÀM TRÍCH XUẤT ĐẶC TRƯNG CHO LSTM
# ==========================================
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20) 
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=N_MFCC).T
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    if len(mfcc) > MAX_LEN: mfcc = mfcc[:MAX_LEN, :]
    else: mfcc = np.pad(mfcc, pad_width=((0, MAX_LEN - len(mfcc)), (0, 0)), mode='constant')
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)


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
    
    master_voice_path = "my_voice/a1.wav" 
    
    try:
        y_user, _ = librosa.load(raw_path, sr=16000)
        y_master, _ = librosa.load(master_voice_path, sr=16000)
        
        ai_device = models['voice_auth'].device 
        
        tensor_user = torch.tensor(y_user).unsqueeze(0).float().to(ai_device)
        tensor_master = torch.tensor(y_master).unsqueeze(0).float().to(ai_device)
        
        with torch.no_grad():
            f_user = models['voice_auth'].mods.compute_features(tensor_user)
            f_user = models['voice_auth'].mods.mean_var_norm(f_user, torch.ones(f_user.shape[0]).to(ai_device))
            emb_user = models['voice_auth'].mods.embedding_model(f_user)
            
            f_master = models['voice_auth'].mods.compute_features(tensor_master)
            f_master = models['voice_auth'].mods.mean_var_norm(f_master, torch.ones(f_master.shape[0]).to(ai_device))
            emb_master = models['voice_auth'].mods.embedding_model(f_master)
        
        import torch.nn.functional as F
        score = F.cosine_similarity(emb_user.view(1, -1), emb_master.view(1, -1))
        similarity_value = score.item()
        
        percent_score = int(max(0, min(100, (similarity_value * 150) + 35)))
        is_matched = percent_score >= 80
        
    except Exception as e:
        print(f"❌ Lỗi sinh trắc học: {e}")
        return jsonify({"error": "Audio processing failed"}), 500
    finally:
        if os.path.exists(raw_path): os.remove(raw_path)
        
    return jsonify({"match": is_matched, "score": percent_score})

# ==========================================
# API 2: ĐIỀU HƯỚNG NHANH (LOCAL LSTM)
# ==========================================
@app.route('/api/local_command', methods=['POST'])
def local_command():
    if 'audio' not in request.files: return jsonify({"error": "No audio"}), 400
    
    temp_path = os.path.join(tempfile.gettempdir(), "local_cmd.webm")
    request.files['audio'].save(temp_path)
    mfcc_data = extract_mfcc(temp_path)

    with torch.no_grad():
        out = models['lstm'](mfcc_data)
        prob = torch.nn.functional.softmax(out, dim=1).squeeze(0)
        max_prob = torch.max(prob).item() * 100
        intent = models['lstm_idx_to_class'][torch.argmax(prob).item()]

    if os.path.exists(temp_path): os.remove(temp_path)
    
    if max_prob >= 80.0:
        return jsonify({"status": "success", "intent": intent, "confidence": max_prob})
    else:
        return jsonify({"status": "rejected", "intent": "unknown", "message": "Độ tin cậy quá thấp"})

# ==========================================
# API 3: TRỢ LÝ ẢO TÍCH HỢP (PHOWHISPER + GEMINI FUNCTION CALLING)
# ==========================================
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel(os.getenv('MODEL_NAME', 'gemini-1.5-flash'))

@app.route('/api/cloud_command', methods=['POST'])
def cloud_command():
    if 'audio' not in request.files: return jsonify({"error": "No audio data"}), 400
        
    temp_path = os.path.join(tempfile.gettempdir(), "cloud_cmd.webm")
    request.files['audio'].save(temp_path)
    
    try:
        # 1. STT: Dịch giọng nói thành Text bằng PhoWhisper cục bộ
        y, sr = librosa.load(temp_path, sr=16000)
        feat = models['w_vi_proc'](y, sampling_rate=sr, return_tensors="pt").input_features
        
        whisper_device = models['w_vi_model'].device
        whisper_dtype = models['w_vi_model'].dtype
        feat = feat.to(device=whisper_device, dtype=whisper_dtype)
        
        with torch.no_grad():
            # ĐỔI THÀNH TIẾNG ANH (language="en") -> Tốc độ dịch sẽ cực nhanh
            ids = models['w_vi_model'].generate(
                feat,
                task="transcribe",
                language="en", 
                pad_token_id=models['w_vi_model'].config.eos_token_id
            )
            transcription = models['w_vi_proc'].batch_decode(ids, skip_special_tokens=True)[0].strip()
        
        print(f"🗣️  [User said]: {transcription}")

        # 2. LLM: Prompt bằng Tiếng Anh giúp Gemini phản xạ tức thời
        system_prompt = """
        You are a smart home assistant. The user has 5 devices: "light", "fan", "ac", "door", "tv".
        Analyze the speech and RETURN EXACTLY THIS JSON STRUCTURE (no markdown):
        {
            "hardware_command": {
                "device": "device name (or null)",
                "action": "'on' or 'off' (or null)"
            },
            "speech_reply": "Your friendly reply in English (under 2 sentences)"
        }
        User's speech: 
        """
        response = gemini_model.generate_content(system_prompt + transcription)
        raw_text = response.text.replace("```json", "").replace("```", "").strip()
        
        try:
            gemini_data = json.loads(raw_text)
        except:
             gemini_data = {
                 "hardware_command": {"device": None, "action": None},
                 "speech_reply": raw_text
             }

        return jsonify({
            "status": "success",
            "transcription": transcription,
            "ai_response": gemini_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc() 
        return jsonify({"status": "error", "message": str(e)}), 500
        
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

if __name__ == '__main__':
    load_smarthome_models()
    # Chạy cổng 5002 để khớp với Frontend home.html
    app.run(host='0.0.0.0', port=5002, debug=False)