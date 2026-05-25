import os
import time
import torch
import librosa
import re
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ==========================================
# 1. CẤU HÌNH
# ==========================================
TEST_DIR = "Dataset/Test"
# Dùng bản Whisper nhỏ nhất chuyên tiếng Anh để so sánh tốc độ
MODEL_ID = "openai/whisper-tiny.en" 

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text).strip()
    return text

def run_whisper_benchmark():
    print("⏳ Đang tải bộ não Whisper từ OpenAI (Lần đầu sẽ mất chút thời gian tải model)...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval() # Bật chế độ làm bài thi
    
    classes = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    
    total_files = 0
    correct_predictions = 0
    total_latency = 0
    
    print("\n" + "="*50)
    print("⚖️ BẮT ĐẦU CHẤM ĐIỂM WHISPER TRÊN TẬP TEST...")
    print("="*50)

    with torch.no_grad():
        for cls_name in classes:
            folder_path = os.path.join(TEST_DIR, cls_name)
            files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
            
            print(f"🎙️ Đang chấm điểm thư mục: '{cls_name}' ({len(files)} file)...")
            
            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                
                try:
                    y, sr = librosa.load(file_path, sr=16000)
                    
                    # Tính thời gian bắt đầu suy luận
                    start_time = time.time()
                    
                    # Ép dữ liệu vào chuẩn của Whisper
                    input_features = processor(y, sampling_rate=sr, return_tensors="pt").input_features
                    predicted_ids = model.generate(input_features)
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    
                    # Tính thời gian kết thúc
                    end_time = time.time()
                    total_latency += (end_time - start_time)
                    
                    # Dọn dẹp text để so sánh
                    pred_clean = clean_text(transcription)
                    total_files += 1
                    
                    # 💡 Logic chấm điểm công bằng:
                    if cls_name == "noise":
                        # Với tạp âm, Whisper thường in ra rỗng, hoặc in ra rác. 
                        # Miễn là nó KHÔNG nhầm thành 5 lệnh điều khiển thì tính là Đúng.
                        if pred_clean not in ["yes", "no", "stop", "on", "off"]:
                            correct_predictions += 1
                    else:
                        # Với các lệnh, kiểm tra xem từ khóa có nằm trong câu trả lời không
                        if cls_name in pred_clean:
                            correct_predictions += 1
                            
                except Exception as e:
                    pass

    # ==========================================
    # 3. TỔNG KẾT KẾT QUẢ SO SÁNH
    # ==========================================
    accuracy = (correct_predictions / total_files) * 100
    avg_latency_ms = (total_latency / total_files) * 1000

    print("\n" + "="*50)
    print(f"🏆 KẾT QUẢ BENCHMARK CỦA WHISPER (TINY.EN):")
    print(f"👉 Số câu đúng: {correct_predictions} / {total_files}")
    print(f"👉 ĐỘ CHÍNH XÁC (ACCURACY): {accuracy:.2f}%")
    print(f"👉 TỐC ĐỘ PHẢN HỒI TRUNG BÌNH (LATENCY): {avg_latency_ms:.2f} ms / file")
    print("="*50)

if __name__ == "__main__":
    run_whisper_benchmark()