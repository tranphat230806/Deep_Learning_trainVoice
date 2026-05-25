import os
import json
import torch
import torch.nn as nn
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# ==========================================
# 1. CẤU HÌNH 
# ==========================================
MODEL_DIR = "models/RNN_Baseline"
N_MFCC = 40
MAX_LEN = 100

# 💡 BƯỚC 1: SỬA LẠI THÀNH MẠNG SimpleRNN (Khớp 100% với lúc Train)
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        avg_out = torch.mean(out, dim=1) 
        final_out = self.fc(avg_out)
        return final_out

# ==========================================
# 2. HÀM THU ÂM TRỰC TIẾP TỪ MICROPHONE
# ==========================================
def record_from_mic(duration=2, fs=16000, filename="temp_mic_record.wav"):
    print(f"\n🎙️  BẮT ĐẦU THU ÂM TRONG {duration} GIÂY...")
    print("🗣️ Hãy nói tiếng Anh (VD: 'yes', 'no', 'stop', 'on', 'off')...")
    print("Hoặc ngồi im để test nhãn 'noise'...")
    
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait() 
    
    print("✅ Đã ghi âm xong! Đang phân tích dữ liệu...")
    write(filename, fs, myrecording)
    return filename

# ==========================================
# 3. HÀM RÚT TRÍCH ĐẶC TRƯNG 
# ==========================================
def extract_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
        
        # 💡 BƯỚC 2: BẮT BUỘC PHẢI CÓ DÒNG CHUẨN HÓA NÀY
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        if len(mfcc) > MAX_LEN:
            mfcc = mfcc[:MAX_LEN, :]
        else:
            pad_width = MAX_LEN - len(mfcc)
            mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
        return mfcc
    except Exception as e:
        print(f"❌ Lỗi đọc file audio: {e}")
        return None

# ==========================================
# 4. HÀM DỰ ĐOÁN CHÍNH
# ==========================================
def test_model(test_file):
    label_path = os.path.join(MODEL_DIR, "label_map.json")
    if not os.path.exists(label_path):
        print("❌ Chưa tìm thấy file label_map.json!")
        return
        
    with open(label_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Khởi tạo mô hình RNN với hidden_size=64 (giống hệt lúc train)
    model = SimpleRNN(input_size=N_MFCC, hidden_size=64, num_classes=len(class_to_idx))
    
    # Load file weights của RNN
    model_path = os.path.join(MODEL_DIR, "simple_rnn.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    mfcc_data = extract_mfcc(test_file)
    if mfcc_data is None:
        return
    
    tensor_data = torch.tensor(mfcc_data, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor_data)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    result_class = idx_to_class[predicted_idx.item()]
    confidence_score = confidence.item() * 100
    
    print("=" * 50)
    print(f"🤖 AI NGHE ĐƯỢC LỆNH:  >> {result_class.upper()} <<")
    print(f"🎯 Độ tự tin (Confidence): {confidence_score:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    while True:
        try:
            # Dừng chương trình chờ bạn ấn Enter
            ans = input("\n🔴 NHẤN [ENTER] ĐỂ THU ÂM (Gõ 'q' để thoát): ")
            if ans.lower() == 'q':
                break
                
            # Chạy hàm thu âm 2 giây
            file_vua_thu = record_from_mic(duration=2)
            test_model(file_vua_thu)
            
            if os.path.exists(file_vua_thu):
                os.remove(file_vua_thu)
        except KeyboardInterrupt:
            break