import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CẤU HÌNH SIÊU THAM SỐ
# ==========================================
TRAIN_DIR = "Dataset/Train"
TEST_DIR = "Dataset/Test"
MODEL_DIR = "models/LSTM_Advanced"  # 💡 Lưu sang một thư mục mới
EPOCHS = 30           
BATCH_SIZE = 64       
LEARNING_RATE = 0.001
N_MFCC = 40           
MAX_LEN = 100         

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU
# ==========================================
def extract_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T 
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        if len(mfcc) > MAX_LEN:
            mfcc = mfcc[:MAX_LEN, :]
        else:
            pad_width = MAX_LEN - len(mfcc)
            mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
        return mfcc
    except Exception:
        return None

class SpeechDataset(Dataset):
    def __init__(self, data_dir, class_mapping=None):
        self.features = []
        self.labels = []
        
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = class_mapping if class_mapping else {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"🔍 Đang nạp dữ liệu từ: {data_dir}...")
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
        print(f"✅ Tải thành công {len(self.labels)} file âm thanh!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 3. MẠNG LSTM (VŨ KHÍ TỐI THƯỢNG)
# ==========================================
class SpeechLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SpeechLSTM, self).__init__()
        # 💡 SỬ DỤNG LSTM VỚI 2 LỚP (num_layers=2) VÀ DROPOUT ĐỂ THÔNG MINH HƠN
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        avg_out = torch.mean(lstm_out, dim=1) 
        final_out = self.fc(avg_out)
        return final_out

# ==========================================
# 4. HUẤN LUYỆN VÀ VẼ BIỂU ĐỒ
# ==========================================
def run_lstm():
    print("-" * 50)
    print("🧠 BƯỚC 1: CHUẨN BỊ TẬP DỮ LIỆU TRAIN & TEST (DÙNG CHUNG DATA VỚI RNN)")
    train_dataset = SpeechDataset(TRAIN_DIR)
    test_dataset = SpeechDataset(TEST_DIR, class_mapping=train_dataset.class_to_idx)
    
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(train_dataset.class_to_idx, f, ensure_ascii=False, indent=4)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Khởi tạo LSTM với hidden_size lớn hơn (128) để nhớ được nhiều đặc trưng phức tạp hơn
    model = SpeechLSTM(input_size=N_MFCC, hidden_size=128, num_classes=len(train_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history_loss = []
    history_acc = []

    print("\n" + "="*50)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MẠNG LSTM...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad() 
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        history_loss.append(avg_loss)

        # Chấm điểm ngay
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
        epoch_acc = 100 * correct / total
        history_acc.append(epoch_acc)
            
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] - Loss: {avg_loss:.4f} - Test Accuracy: {epoch_acc:.2f}%")

    print("\n" + "="*50)
    print(f"🏆 KẾT QUẢ CUỐI CÙNG TRÊN TẬP TEST KÍN (2.344 FILE):")
    print(f"👉 Thuật toán 2 (Mạng LSTM) đạt Độ chính xác: {history_acc[-1]:.2f}%")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "speech_lstm.pth"))
    
    # Vẽ biểu đồ
    print("\n📈 Đang vẽ biểu đồ huấn luyện LSTM...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS+1), history_loss, marker='o', color='purple', label='Train Loss')
    plt.title("Biểu đồ Suy giảm Sai số (LSTM)")
    plt.xlabel("Vòng lặp (Epochs)")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS+1), history_acc, marker='s', color='green', label='Test Accuracy')
    plt.title("Biểu đồ Tăng trưởng Độ chính xác (LSTM)")
    plt.xlabel("Vòng lặp (Epochs)")
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(MODEL_DIR, "lstm_training_chart.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"🎉 Đã lưu model và biểu đồ tại: {MODEL_DIR}")

if __name__ == "__main__":
    run_lstm()