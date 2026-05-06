import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

def process_mixed_audio_folder(raw_folder, dataset_folder, prefix="moden"):
    # Tạo thư mục đích để chứa data chuẩn
    os.makedirs(dataset_folder, exist_ok=True)
    
    # Biến đếm tổng số file đã cắt được từ TẤT CẢ các file gốc
    total_saved = 0
    
    print(f"📂 Đang quét thư mục: {raw_folder}")
    
    # Duyệt qua từng file (wav, m4a, mp3...) có trong thư mục
    for filename in os.listdir(raw_folder):
        # Bỏ qua các file rác của hệ thống, chỉ lấy file âm thanh
        if filename.lower().endswith((".wav", ".m4a", ".mp3", ".mp4")):
            file_path = os.path.join(raw_folder, filename)
            print(f"\n⏳ Đang xử lý file: {filename} ...")
            
            try:
                # 1. Đọc file (Tự động nhận diện định dạng nhờ FFmpeg)
                audio = AudioSegment.from_file(file_path)
                
                # 2. Cắt khoảng lặng
                chunks = split_on_silence(
                    audio,
                    min_silence_len=400,  
                    silence_thresh=audio.dBFS - 11, 
                    keep_silence=200
                )
                
                print(f"✂️ File này cắt được {len(chunks)} đoạn. Đang lưu...")
                
                # 3. Lưu các đoạn đã cắt và tăng số thứ tự
                for chunk in chunks:
                    if len(chunk) > 500:
                        output_file = os.path.join(dataset_folder, f"{prefix}_{total_saved:04d}.wav")
                        
                        # Ép chuẩn: 1 Kênh, 16000Hz, đuôi WAV
                        chunk = chunk.set_frame_rate(16000).set_channels(1)
                        chunk.export(output_file, format="wav")
                        total_saved += 1
                        
            except Exception as e:
                print(f"❌ Bỏ qua file '{filename}' do lỗi: {e}")
                
    print(f"\n🎉 HOÀN TẤT! tổng số file đã được xử lý: {total_saved}")

if __name__ == "__main__":
   
    THU_MUC_CHUA_FILE_GOC = "Raw_Data" 
    
    # Thư mục chứa data sạch để train
    THU_MUC_LUU_DATASET = "Dataset/mo_den"
    
    # Tên tiền tố
    TEN_PREFIX = "moden"
    
    # Chạy máy xay
    process_mixed_audio_folder(THU_MUC_CHUA_FILE_GOC, THU_MUC_LUU_DATASET, prefix=TEN_PREFIX)