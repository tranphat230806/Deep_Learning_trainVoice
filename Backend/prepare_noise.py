import os
import librosa
import soundfile as sf

# Thư mục gốc chứa nhiễu của Google
NOISE_DIR = "Google_Speech_Data/_background_noise_"
# Thư mục đích chứa các file nhiễu đã băm nhỏ (Ngang hàng với yes, no, stop...)
OUTPUT_DIR = "Google_Speech_Data/noise"

TARGET_SR = 16000
CHUNK_LENGTH = 1.0 # 1 giây

def split_noise():
    print("🔪 ĐANG BĂM CÁC FILE NHIỄU DÀI THÀNH TỪNG KHÚC 1 GIÂY...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    chunk_samples = int(TARGET_SR * CHUNK_LENGTH)
    count = 0

    for file_name in os.listdir(NOISE_DIR):
        if file_name.endswith('.wav'):
            file_path = os.path.join(NOISE_DIR, file_name)
            y, sr = librosa.load(file_path, sr=TARGET_SR)
            
            print(f" Đang cắt file: {file_name}...")
            # Cắt thành từng khúc 1 giây
            for i in range(0, len(y) - chunk_samples, chunk_samples):
                chunk = y[i:i + chunk_samples]
                out_name = os.path.join(OUTPUT_DIR, f"noise_{count:04d}.wav")
                sf.write(out_name, chunk, TARGET_SR)
                count += 1
                
    print("\n🎉 HOÀN TẤT!")
    print(f"✅ Đã tạo thành công {count} file noise 1 giây.")
    print("👉 Bây giờ thư mục 'noise' đã sẵn sàng để trở thành 1 nhãn (Class) mới!")

if __name__ == "__main__":
    split_noise()