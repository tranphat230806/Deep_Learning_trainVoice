import sounddevice as sd
import soundfile as sf
import os
import time

def record_audio(filename, duration_minutes=3, sample_rate=16000):
    """
    Hàm thu âm liên tục trong thời gian chỉ định.
    - filename: Tên file lưu (VD: Raw_Data/mo_den.wav)
    - duration_minutes: Số phút thu (Mặc định 3)
    - sample_rate: Tần số lấy mẫu (16000 là chuẩn cho Whisper/LSTM)
    """
    duration_seconds = duration_minutes * 60
    
    print("="*50)
    print(f"🎤 CHUẨN BỊ THU ÂM: {filename}")
    print(f"⏱️  Thời lượng: {duration_minutes} phút ({duration_seconds} giây)")
    print("="*50)
    
    # Đếm ngược cho bạn có thời gian hít sâu
    for i in range(3, 0, -1):
        print(f"Bắt đầu trong {i}...")
        time.sleep(1)
        
    print("\n🔴 ĐANG THU ÂM... Bắt đầu đọc đi!")
    print("💡 Mẹo: Đọc - Nghỉ 1s - Đọc - Nghỉ 1s")
    
    try:
        # Bắt đầu thu
        mydata = sd.rec(int(duration_seconds * sample_rate), samplerate=sample_rate, channels=1)
        
        # Cái này giống như đồng hồ bấm giờ, cập nhật liên tục để bạn biết thu được bao lâu rồi
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            elapsed = int(time.time() - start_time)
            print(f"\r⏳ Đã thu được: {elapsed//60} phút {elapsed%60} giây / {duration_minutes} phút...", end="")
            time.sleep(1)
            
        sd.wait() # Đảm bảo thu xong hẳn mới chạy tiếp
        
        # Lưu file
        print(f"\n✅ Đã thu xong! Đang lưu file {filename}...")
        sf.write(filename, mydata, sample_rate)
        print("💾 Lưu thành công!")
        
    except KeyboardInterrupt:
        # Đã sửa thành sd.stop() để dừng ngay lập tức khi bấm Ctrl+C
        print("\n⚠️ Đã bị ngắt (Ctrl+C). Đang ép dừng micro...")
        sd.stop() 
        
        # Lưu file ngay tức thì
        sf.write(filename, mydata[:int((time.time() - start_time) * sample_rate)], sample_rate)
        print("💾 Lưu thành công phần bị ngắt!")

if __name__ == "__main__":
    # Đảm bảo thư mục tồn tại
    output_dir = "Raw_Data"
    os.makedirs(output_dir, exist_ok=True)
    
    # === BẠN CHỈ CẦN SỬA 2 DÒNG NÀY CHO MỖI LẦN THU ===
    # Đã xóa phần Raw_Data/ bị thừa ở đây
    TEN_FILE = os.path.join(output_dir, "mo_den_part1.wav") 
    THOI_GIAN = 3 # Đơn vị: Phút
    
    record_audio(TEN_FILE, duration_minutes=THOI_GIAN)