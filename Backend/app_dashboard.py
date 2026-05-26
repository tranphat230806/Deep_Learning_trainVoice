import os
import time
import json
import torch
import torch.nn as nn
import librosa
import numpy as np
import re
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import WhisperProcessor, WhisperForConditionalGeneration

app = Flask(__name__)
CORS(app)

N_MFCC = 40
MAX_LEN = 100
models = {}

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(torch.mean(out, dim=1))

class SpeechLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SpeechLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(torch.mean(lstm_out, dim=1))

def load_dashboard_models():
    print("⏳ [API 1] Đang nạp các mô hình Tiếng Anh phục vụ Đua tốc độ...")
    with open("models/RNN_Baseline/label_map.json", "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    models['idx_to_class'] = {v: k for k, v in class_to_idx.items()}

    models['rnn'] = SimpleRNN(N_MFCC, 64, len(class_to_idx))
    models['rnn'].load_state_dict(torch.load("models/RNN_Baseline/simple_rnn.pth", weights_only=True))
    models['rnn'].eval()

    models['lstm'] = SpeechLSTM(N_MFCC, 128, len(class_to_idx))
    models['lstm'].load_state_dict(torch.load("models/LSTM_Advanced/speech_lstm.pth", weights_only=True))
    models['lstm'].eval()

    models['w_en_proc'] = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    models['w_en_model'] = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    models['w_en_model'].eval()
    print("✅ [API 1] Đã nạp xong RNN, LSTM, Whisper-tiny.en!")

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    if len(mfcc) > MAX_LEN: mfcc = mfcc[:MAX_LEN, :]
    else: mfcc = np.pad(mfcc, pad_width=((0, MAX_LEN - len(mfcc)), (0, 0)), mode='constant')
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file found"}), 400
        
    audio_file = request.files['audio']
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_dashboard_bench.wav")
    audio_file.save(temp_path)
    
    # ==========================================
    # TIỀN XỬ LÝ (Trích xuất đặc trưng chung, KHÔNG BẤM GIỜ)
    # ==========================================
    mfcc_data = extract_mfcc(temp_path)

    # 🏎️ 1. ĐO TỐC ĐỘ RNN (Chỉ đo thời gian suy luận AI)
    t0 = time.time()
    with torch.no_grad():
        rnn_out = models['rnn'](mfcc_data)
        rnn_prob = torch.nn.functional.softmax(rnn_out, dim=1)
        rnn_acc = torch.max(rnn_prob).item() * 100
        rnn_ans = models['idx_to_class'][torch.max(rnn_out, 1)[1].item()]
    t_rnn = (time.time() - t0) * 1000

    # 🏎️ 2. ĐO TỐC ĐỘ LSTM (Chỉ đo thời gian suy luận AI)
    t0 = time.time()
    with torch.no_grad():
        lstm_out = models['lstm'](mfcc_data)
        lstm_prob = torch.nn.functional.softmax(lstm_out, dim=1)
        lstm_acc = torch.max(lstm_prob).item() * 100
        lstm_ans = models['idx_to_class'][torch.max(lstm_out, 1)[1].item()]
    t_lstm = (time.time() - t0) * 1000

    # 🏎️ 3. ĐO TỐC ĐỘ WHISPER (Nó dùng thư viện riêng nên phải đo từ đầu)
    t0 = time.time()
    y, sr = librosa.load(temp_path, sr=16000)
    feat = models['w_en_proc'](y, sampling_rate=sr, return_tensors="pt").input_features
    with torch.no_grad():
        ids = models['w_en_model'].generate(feat)
        w_ans = models['w_en_proc'].batch_decode(ids, skip_special_tokens=True)[0]
        w_ans = re.sub(r'[^\w\s]', '', w_ans.lower()).strip()
    t_w = (time.time() - t0) * 1000
    w_acc = None

    # ... (Phần trả về JSON giữ nguyên)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return jsonify({
        "rnn": {"latency": round(t_rnn, 2), "result": rnn_ans, "accuracy": round(rnn_acc, 2)},
        "lstm": {"latency": round(t_lstm, 2), "result": lstm_ans, "accuracy": round(lstm_acc, 2)},
        "phowhisper": {"latency": round(t_w, 2), "result": w_ans, "accuracy": w_acc}
    })

if __name__ == '__main__':
    load_dashboard_models()
    app.run(host='0.0.0.0', port=5001, debug=False)