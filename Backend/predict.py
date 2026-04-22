import os
import time
import sounddevice as sd
import soundfile as sf
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# CẤU HÌNH THU ÂM
# ==========================================
SR = 16000          # Tần số lấy mẫu chuẩn cho Whisper (16kHz)
DURATION = 5        # Số giây thu âm mỗi lần
TEMP_WAV = "temp_mic.wav"

# ==========================================
# KHỞI TẠO MODEL PHO-WHISPER
# ==========================================
print("Đang tải model PhoWhisper vào RAM...")
transcriber = pipeline("automatic-speech-recognition", model="./models/PhoWhisper-medium")
print("✅ Model đã sẵn sàng!\n")

def record_audio():
    """Hàm mở micro và thu âm"""
    input("Bấm [ENTER] để bắt đầu nói...")
    print(f"🎤 Đang thu âm ({DURATION} giây) - Hãy nói lệnh của bạn!")
    
    # Mở mic thu âm
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait() 
    
    # Ghi ra file .wav tạm
    sf.write(TEMP_WAV, audio, SR)
    print("✅ Đã thu xong! Đang chuyển cho AI xử lý...\n")
    return TEMP_WAV

def recognize_audio(file_path):
    """Hàm đưa file vào model để nhận diện"""
    start_time = time.time()
    
    # Chạy AI
    result = transcriber(file_path)
    
    end_time = time.time()
    
    print("=" * 40)
    print("KẾT QUẢ NHẬN DIỆN:")
    print(f"Câu lệnh: {result['text']}")
    print(f"Thời gian xử lý: {round(end_time - start_time, 2)} giây")
    print("=" * 40 + "\n")

# ==========================================
# LUỒNG CHẠY CHÍNH (MAIN LOOP)
# ==========================================
if __name__ == "__main__":
    try:
        while True:
            # 1. Thu âm
            wav_file = record_audio()
            
            # 2. Nhận diện chữ
            recognize_audio(wav_file)
            
            # 3. Dọn dẹp file tạm
            if os.path.exists(TEMP_WAV):
                os.remove(TEMP_WAV)
                
            # 4. Hỏi xem có muốn tiếp tục không
            tiep_tuc = input("Nhập 'q' để thoát, hoặc bấm [ENTER] để nói câu khác: ")
            if tiep_tuc.lower() == 'q':
                print("Đã thoát chương trình.")
                break
            print("\n")
            
    except KeyboardInterrupt:
        # Bắt sự kiện người dùng bấm Ctrl+C để thoát ngang
        print("\n👋 Đã thoát chương trình.")
        if os.path.exists(TEMP_WAV):
            os.remove(TEMP_WAV)