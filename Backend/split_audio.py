import os
import shutil
import random

SOURCE_DIR = "Google_Speech_Data" 
TRAIN_DIR = "Dataset/Train"
TEST_DIR = "Dataset/Test"

# Chọn từ khóa đại diện
CLASSES = ["yes", "no", "stop", "on", "off", "noise"]

# Tỷ lệ chia: 90% cho Train, 10% cho Test
TRAIN_RATIO = 0.90

def prepare_dataset():
    print("🚀 BẮT ĐẦU CHIA DỮ LIỆU TỰ ĐỘNG (TỶ LỆ 90/10)...")
    
    # Tạo cấu trúc thư mục
    for c in CLASSES:
        os.makedirs(os.path.join(TRAIN_DIR, c), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, c), exist_ok=True)
        
    total_train = 0
    total_test = 0

    for c in CLASSES:
        src_folder = os.path.join(SOURCE_DIR, c)
        if not os.path.exists(src_folder):
            print(f"❌ Không tìm thấy thư mục: {src_folder}")
            continue
            
        # Lấy toàn bộ file .wav và trộn ngẫu nhiên để tránh thiên lệch
        files = [f for f in os.listdir(src_folder) if f.endswith('.wav')]
        random.shuffle(files)
        
        total_files_in_class = len(files)
        if total_files_in_class == 0:
            continue
            
        # Tính toán số lượng cắt lát
        train_count = int(total_files_in_class * TRAIN_RATIO)
        
        train_files = files[:train_count]
        test_files = files[train_count:] # Phần còn lại là Test

        print(f"⏳ Class '{c}': Tổng {total_files_in_class} file -> Cắt {len(train_files)} Train / {len(test_files)} Test...")
        
        # Copy file sang Train
        for f in train_files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(TRAIN_DIR, c, f))
            total_train += 1
            
        # Copy file sang Test
        for f in test_files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(TEST_DIR, c, f))
            total_test += 1

    print("\n🎉 HOÀN TẤT CHIA DỮ LIỆU ĐÓNG BĂNG!")
    print(f"✅ TỔNG SỐ FILE TRAIN: {total_train}")
    print(f"✅ TỔNG SỐ FILE TEST:  {total_test}")
    print("👉 Bây giờ, thư mục Train và Test đã sẵn sàng để so sánh các mạng!")

if __name__ == "__main__":
    prepare_dataset()