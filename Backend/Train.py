import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

# ==========================================
# 1. CẤU HÌNH SIÊU THAM SỐ (HYPERPARAMETERS)
# ==========================================
DATASET_DIR = "Dataset"
MODEL_DIR = "models/MyModel"
EPOCHS = 30           # Số vòng lặp học (Có thể tăng lên 50 nếu độ chính xác chưa cao)
BATCH_SIZE = 32       # Số lượng file âm thanh đưa vào học cùng lúc
LEARNING_RATE = 0.001
N_MFCC = 40           # Lấy 40 đặc trưng âm thanh
MAX_LEN = 100         # Độ dài khung thời gian chuẩn (Cắt/bù các file cho bằng nhau)

# Đảm bảo thư mục lưu model tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (DATASET & DATALOADER)
# ==========================================
def extract_mfcc(file_path):
    """ Hàm đọc file audio và chuyển thành ma trận đặc trưng MFCC """
    try:
        # Load audio với tần số 16000Hz chuẩn
        y, sr = librosa.load(file_path, sr=16000)
        # Rút trích MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        # Chuyển vị trí ma trận: Từ (Đặc trưng, Thời gian) -> (Thời gian, Đặc trưng)
        mfcc = mfcc.T 
        
        # Padding: Các file âm thanh có độ dài ngắn khác nhau, ta ép về 1 kích thước MAX_LEN
        if len(mfcc) > MAX_LEN:
            mfcc = mfcc[:MAX_LEN, :] # Cắt bớt nếu quá dài
        else:
            pad_width = MAX_LEN - len(mfcc)
            mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant') # Bù số 0 nếu quá ngắn
            
        return mfcc
    except Exception as e:
        print(f"Lỗi đọc file {file_path}: {e}")
        return None

class SpeechDataset(Dataset):
    def __init__(self, data_dir):
        self.features = []
        self.labels = []
        self.classes = []
        
        # Quét các thư mục con trong Dataset (VD: mo_den, tat_den)
        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if os.path.isdir(folder_path):
                self.classes.append(folder_name)
        
        # Tạo từ điển ánh xạ chữ sang số: {'mo_den': 0, 'tat_den': 1}
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print("🔍 Đang tải và trích xuất đặc trưng âm thanh...")
        for cls_name in self.classes:
            folder_path = os.path.join(data_dir, cls_name)
            label_idx = self.class_to_idx[cls_name]
            
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(folder_path, file_name)
                    mfcc_data = extract_mfcc(file_path)
                    
                    if mfcc_data is not None:
                        self.features.append(mfcc_data)
                        self.labels.append(label_idx)
                        
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        print(f"✅ Đã tải xong {len(self.labels)} file. Số nhãn (lệnh) phân loại: {len(self.classes)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # PyTorch yêu cầu dữ liệu phải ở dạng Tensor
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 3. XÂY DỰNG MẠNG NƠ-RON LSTM
# ==========================================
class SpeechLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SpeechLSTM, self).__init__()
        # Lớp LSTM: Giúp học chuỗi thời gian của âm thanh
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        # Lớp tuyến tính: Đưa ra dự đoán cuối cùng
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Đưa dữ liệu qua LSTM
        lstm_out, _ = self.lstm(x)
        # Chỉ lấy kết quả ở bước thời gian cuối cùng để phân loại
        last_out = lstm_out[:, -1, :] 
        out = self.fc(last_out)
        return out

# ==========================================
# 4. HÀM CHẠY HUẤN LUYỆN (TRAINING LOOP)
# ==========================================
def train_model():
    dataset = SpeechDataset(DATASET_DIR)
    
    # ⚠️ CẢNH BÁO QUAN TRỌNG: Cần ít nhất 2 nhãn để phân loại
    if len(dataset.classes) < 2:
        print("❌ LỖI: Bạn phải có ít nhất 2 thư mục con trong folder Dataset (ví dụ: 'mo_den' và 'tat_den').")
        print("Mô hình AI không thể học phân loại nếu chỉ có 1 sự lựa chọn!")
        return

    # Lưu lại từ điển nhãn (Label map) để file Predict.py sau này biết dịch số 0, 1 ra chữ gì
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(dataset.class_to_idx, f, ensure_ascii=False, indent=4)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Khởi tạo mô hình
    model = SpeechLSTM(input_size=N_MFCC, hidden_size=128, num_classes=len(dataset.classes))
    
    # Hàm mất mát (Loss) và bộ tối ưu hóa (Optimizer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad() # Xóa rác đồ thị cũ
            
            # Dự đoán
            outputs = model(batch_features)
            
            # Tính sai số
            loss = criterion(outputs, batch_labels)
            
            # Học và cập nhật trọng số (Backpropagation)
            loss.backward()
            optimizer.step()
            
            # Tính toán độ chính xác (Accuracy)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(dataloader):.4f} - Accuracy: {accuracy:.2f}%")

# ... (các code ở trên giữ nguyên) ...

    # Lưu não bộ đã train xong
    model_path = os.path.join(MODEL_DIR, "speech_lstm.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n🎉 QUÁ TRÌNH HOÀN TẤT! Đã lưu mô hình tại: {model_path}")

# Gọi thẳng hàm thực thi ở cuối file
train_model()