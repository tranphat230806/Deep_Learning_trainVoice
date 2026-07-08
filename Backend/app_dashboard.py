import os
import time
import json
import torch
import torch.nn as nn
import librosa
import numpy as np
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

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
    print("⏳ [API Dashboard] Đang nạp mô hình RNN và LSTM...")
    with open("models/RNN_Baseline/label_map.json", "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    models['idx_to_class'] = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    models['rnn'] = SimpleRNN(N_MFCC, 64, num_classes)
    models['rnn'].load_state_dict(torch.load("models/RNN_Baseline/simple_rnn.pth", map_location='cpu', weights_only=True))
    models['rnn'].eval()

    models['lstm'] = SpeechLSTM(N_MFCC, 128, num_classes)
    models['lstm'].load_state_dict(torch.load("models/LSTM_Advanced/speech_lstm.pth", map_location='cpu', weights_only=True))
    models['lstm'].eval()
    
    print(f"✅ Đã nạp xong!")

@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file found"}), 400
        
    audio_file = request.files['audio']
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_dashboard_bench.wav")
    audio_file.save(temp_path)
    
    y, sr = librosa.load(temp_path, sr=16000)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # 1. TRÍCH XUẤT CHO TOÀN BỘ CÂU (TỔNG THỂ)
    mfcc_full = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=N_MFCC).T
    mfcc_full = (mfcc_full - np.mean(mfcc_full)) / (np.std(mfcc_full) + 1e-8)
    if len(mfcc_full) > MAX_LEN: mfcc_full = mfcc_full[:MAX_LEN, :]
    else: mfcc_full = np.pad(mfcc_full, pad_width=((0, MAX_LEN - len(mfcc_full)), (0, 0)), mode='constant')
    tensor_full = torch.tensor(mfcc_full, dtype=torch.float32).unsqueeze(0)

    # 2. CẮT LÁT THỜI GIAN (SLIDING WINDOW)
    window_size = 16000 
    step_size = 16000   
    
    timeline_lstm = []
    timeline_rnn = [] 
    
    for start in range(0, len(y_trimmed), step_size):
        end = start + window_size
        y_window = y_trimmed[start:end]
        
        if len(y_window) < 16000 * 0.2: 
            continue
            
        time_label = f"{start/16000:.1f}s - {end/16000:.1f}s"
        
        mfcc_win = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=N_MFCC).T
        mfcc_win = (mfcc_win - np.mean(mfcc_win)) / (np.std(mfcc_win) + 1e-8)
        if len(mfcc_win) > MAX_LEN: mfcc_win = mfcc_win[:MAX_LEN, :]
        else: mfcc_win = np.pad(mfcc_win, pad_width=((0, MAX_LEN - len(mfcc_win)), (0, 0)), mode='constant')
        tensor_win = torch.tensor(mfcc_win, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            rnn_win_out = models['rnn'](tensor_win)
            rnn_win_prob = torch.nn.functional.softmax(rnn_win_out, dim=1).squeeze(0)
            r_idx = torch.argmax(rnn_win_prob).item()
            timeline_rnn.append({
                "time": time_label,
                "predicted": models['idx_to_class'][r_idx],
                "confidence": round(rnn_win_prob[r_idx].item() * 100, 1)
            })

        with torch.no_grad():
            lstm_win_out = models['lstm'](tensor_win)
            lstm_win_prob = torch.nn.functional.softmax(lstm_win_out, dim=1).squeeze(0)
            l_idx = torch.argmax(lstm_win_prob).item()
            timeline_lstm.append({
                "time": time_label,
                "predicted": models['idx_to_class'][l_idx],
                "confidence": round(lstm_win_prob[l_idx].item() * 100, 1)
            })

    with torch.no_grad():
        _ = models['rnn'](tensor_full)
        _ = models['lstm'](tensor_full)

    t0 = time.time()
    with torch.no_grad():
        rnn_out = models['rnn'](tensor_full)
        rnn_prob = torch.nn.functional.softmax(rnn_out, dim=1).squeeze(0)
        rnn_ans = models['idx_to_class'][torch.argmax(rnn_prob).item()]
        rnn_probs_dict = {models['idx_to_class'][i]: round(rnn_prob[i].item() * 100, 2) for i in range(len(rnn_prob))}
    t_rnn = (time.time() - t0) * 1000

    t0 = time.time()
    with torch.no_grad():
        lstm_out = models['lstm'](tensor_full)
        lstm_prob = torch.nn.functional.softmax(lstm_out, dim=1).squeeze(0)
        lstm_ans = models['idx_to_class'][torch.argmax(lstm_prob).item()]
        lstm_probs_dict = {models['idx_to_class'][i]: round(lstm_prob[i].item() * 100, 2) for i in range(len(lstm_prob))}
    t_lstm = (time.time() - t0) * 1000

    # ==============================================================
    # 💡 BẢN VÁ LỖI: GHI ĐÈ KẾT QUẢ TỪ CỬA SỔ TRƯỢT LÊN TRÊN CÙNG
    # ==============================================================
    def get_best_window_intent(timeline, default_ans):
        # Lọc ra các từ khóa (khác noise) có độ tự tin >= 80%
        valid_steps = [s for s in timeline if s['predicted'] != 'noise' and s['confidence'] >= 80.0]
        if valid_steps:
            # Nếu có, lấy kết quả tự tin nhất vứt lên làm kết quả cuối cùng
            best_step = max(valid_steps, key=lambda x: x['confidence'])
            return best_step['predicted']
        return default_ans

    # Áp dụng hàm vừa viết
    final_rnn_ans = get_best_window_intent(timeline_rnn, rnn_ans)
    final_lstm_ans = get_best_window_intent(timeline_lstm, lstm_ans)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return jsonify({
        "rnn": {"latency": round(t_rnn, 2), "result": final_rnn_ans, "probs": rnn_probs_dict, "timeline": timeline_rnn},
        "lstm": {"latency": round(t_lstm, 2), "result": final_lstm_ans, "probs": lstm_probs_dict, "timeline": timeline_lstm}
    })

if __name__ == '__main__':
    load_dashboard_models()
    app.run(host='0.0.0.0', port=5001, debug=False)