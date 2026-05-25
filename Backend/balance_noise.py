import os
import shutil

NOISE_DIR = "Google_Speech_Data/noise"

def oversample_noise():
    files = [f for f in os.listdir(NOISE_DIR) if f.endswith('.wav') and not f.startswith('copy_')]
    
    if len(files) == 0:
        print("❌ Không tìm thấy file noise gốc nào!")
        return
        
    print(f"🧬 Đang nhân bản {len(files)} file noise gốc lên 9 lần để cân bằng Data...")
    
    # 396 file * 9 lần copy = khoảng 3.564 file (bằng với số lượng các từ khóa khác)
    for i in range(9):
        for f in files:
            src = os.path.join(NOISE_DIR, f)
            dst = os.path.join(NOISE_DIR, f"copy_{i}_{f}")
            shutil.copy(src, dst)
            
    total_files = len([f for f in os.listdir(NOISE_DIR) if f.endswith('.wav')])
    print(f"🎉 HOÀN TẤT! Thư mục noise hiện tại đã có {total_files} file. Sẵn sàng đi Train!")

if __name__ == "__main__":
    oversample_noise()