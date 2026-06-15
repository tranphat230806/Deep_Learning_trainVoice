import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. CẤU HÌNH SIÊU THAM SỐ
# ==========================================
TRAIN_DIR = "Dataset/Train"
VAL_DIR = "Dataset/Val"       # 💡 Đã thêm tập Validation
TEST_DIR = "Dataset/Test"     # 💡 Tập Test độc lập
MODEL_DIR = "models/RNN_Baseline"
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
    except Exception as e:
        print(f"Lỗi đọc file {file_path}: {e}")
        return None

class SpeechDataset(Dataset):
    def __init__(self, data_dir, class_to_idx):
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx
        self.file_paths = []
        self.labels = []
        
        for label_name in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label_name)
            if not os.path.isdir(label_dir) or label_name not in class_to_idx:
                continue
            for file_name in os.listdir(label_dir):
                if file_name.endswith('.wav'):
                    self.file_paths.append(os.path.join(label_dir, file_name))
                    self.labels.append(class_to_idx[label_name])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mfcc = extract_mfcc(self.file_paths[idx])
        if mfcc is None: mfcc = np.zeros((MAX_LEN, N_MFCC))
        label = self.labels[idx]
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Tự động mapping nhãn (Quét từ tập Train)
classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
idx_to_class = {i: cls_name for cls_name, i in class_to_idx.items()}
with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(class_to_idx, f, ensure_ascii=False, indent=4)

# Load 3 tập dữ liệu riêng biệt
train_dataset = SpeechDataset(TRAIN_DIR, class_to_idx)
val_dataset = SpeechDataset(VAL_DIR, class_to_idx)
test_dataset = SpeechDataset(TEST_DIR, class_to_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. MÔ HÌNH VANILLA RNN
# ==========================================
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = torch.mean(out, dim=1) 
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleRNN(input_size=N_MFCC, hidden_size=64, num_classes=len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. VÒNG LẶP HUẤN LUYỆN (Sử dụng tập VAL)
# ==========================================
history_loss, history_acc = [], []
print(f"🚀 Bắt đầu huấn luyện Vanilla RNN trên {device}...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    history_loss.append(avg_loss)
    
    # 💡 Đánh giá trên tập Validation sau mỗi Epoch
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
    epoch_acc = 100 * correct / total
    history_acc.append(epoch_acc)
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] - Train Loss: {avg_loss:.4f} - Val Acc: {epoch_acc:.2f}%")

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "simple_rnn.pth"))

# ====================================================
# 5. VẼ BIỂU ĐỒ & ĐÁNH GIÁ TRÊN TẬP TEST KÍN (TEST_DIR)
# ====================================================
print("\n📈 Đang vẽ biểu đồ huấn luyện và chấm điểm trên tập Test...")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), history_loss, marker='o', color='red', label='Train Loss')
plt.title("Biểu đồ Suy giảm Sai số (RNN)")
plt.xlabel("Epochs"); plt.grid(True); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), history_acc, marker='s', color='blue', label='Val Accuracy')
plt.title("Biểu đồ Độ chính xác (RNN)")
plt.xlabel("Epochs"); plt.grid(True); plt.legend()
plt.savefig(os.path.join(MODEL_DIR, "rnn_training_chart.png"))

# Chấm điểm cuối cùng trên tập Test
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.to(device)
        outputs = model(batch_features)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.numpy())

print("\n📊 BÁO CÁO PHÂN LOẠI TRÊN TẬP TEST (RNN):")
print(classification_report(all_labels, all_preds, target_names=classes))

# Vẽ Ma trận nhầm lẫn
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=classes, yticklabels=classes)
plt.title("Ma trận nhầm lẫn - Confusion Matrix (RNN)")
plt.ylabel('Nhãn Thực tế')
plt.xlabel('Nhãn Dự đoán')
plt.savefig(os.path.join(MODEL_DIR, "rnn_confusion_matrix.png"))
print(f"✅ Hoàn tất! Ảnh lưu tại {MODEL_DIR}/")