import os
import shutil
import random

SOURCE_DIR = "Google_Speech_Data" 
TRAIN_DIR = "Dataset/Train"
VAL_DIR = "Dataset/Val"     # Thêm thư mục Validation (Test trong Train)
TEST_DIR = "Dataset/Test"   # Thư mục Test độc lập (Test thực tế)

# Chọn từ khóa đại diện
CLASSES = ["up", "down", "on", "off", "noise"]

# Tỷ lệ chia: 80% Train, 10% Val, 10% Test
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

def prepare_dataset():
    print("🚀 BẮT ĐẦU CHIA DỮ LIỆU TỰ ĐỘNG (TỶ LỆ 80/10/10)...")
    
    # Tạo cấu trúc 3 thư mục
    for c in CLASSES:
        os.makedirs(os.path.join(TRAIN_DIR, c), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, c), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, c), exist_ok=True)
        
    total_train = 0
    total_val = 0
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
            
        # Tính toán mốc cắt lát (Slicing Index)
        train_end = int(total_files_in_class * TRAIN_RATIO)
        val_end = train_end + int(total_files_in_class * VAL_RATIO)
        
        # Cắt mảng (Array Slicing)
        train_files = files[:train_end]                 # 0 -> 80%
        val_files = files[train_end:val_end]            # 80% -> 90%
        test_files = files[val_end:]                    # 90% -> 100%

        print(f"⏳ Class '{c}': Tổng {total_files_in_class} file -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
        
        # Copy file sang Train
        for f in train_files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(TRAIN_DIR, c, f))
            total_train += 1
            
        # Copy file sang Validation
        for f in val_files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(VAL_DIR, c, f))
            total_val += 1
            
        # Copy file sang Test
        for f in test_files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(TEST_DIR, c, f))
            total_test += 1

    print("\n🎉 HOÀN TẤT CHIA DỮ LIỆU ĐÓNG BĂNG!")
    print(f"✅ TỔNG SỐ FILE TRAIN (80%):      {total_train}")
    print(f"✅ TỔNG SỐ FILE VALIDATION (10%): {total_val}")
    print(f"✅ TỔNG SỐ FILE TEST (10%):       {total_test}")
    print("Xong !!")

if __name__ == "__main__":
    prepare_dataset()