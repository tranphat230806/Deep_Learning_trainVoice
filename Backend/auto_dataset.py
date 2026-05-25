import os
import random
import numpy as np
import librosa
import soundfile as sf
import asyncio
import edge_tts
import time

# ==========================================
# 1. CẤU HÌNH DATASET
# ==========================================
DATASET_DIR = "Dataset"
AUGMENT_PER_VOICE = 50   # Số bản augment mỗi giọng. Tổng = giọng × câu × augment

# ──────────────────────────────────────────
# 10 nhãn lệnh điều khiển SmartHome
# ──────────────────────────────────────────
LENH_SMART_HOME = {
    "mo_den":       "Light on",
    "tat_den":      "Light off",
    "mo_quat":      "Fan on",
    "tat_quat":     "Fan off",
    "mo_tv":        "TV on",
    "tat_tv":       "TV off",
    "mo_may_lanh":  "AC on",
    "tat_may_lanh": "AC off",
    "mo_nhac":      "Music play",
    "tat_nhac":     "Music stop"
}

# ──────────────────────────────────────────
# Các biến thể câu nói cho mỗi lệnh
# Giúp model học được nhiều cách nói khác nhau
# ──────────────────────────────────────────
SENTENCE_VARIANTS = {
    "mo_den":       ["Light on", "Turn on the light", "Lights on please"],
    "tat_den":      ["Light off", "Turn off the light", "Lights off please"],
    "mo_quat":      ["Fan on", "Turn on the fan", "Start the fan"],
    "tat_quat":     ["Fan off", "Turn off the fan", "Stop the fan"],
    "mo_tv":        ["TV on", "Turn on the TV", "Switch on TV"],
    "tat_tv":       ["TV off", "Turn off the TV", "Switch off TV"],
    "mo_may_lanh":  ["AC on", "Turn on the AC", "Start air conditioner"],
    "tat_may_lanh": ["AC off", "Turn off the AC", "Stop air conditioner"],
    "mo_nhac":      ["Music play", "Play music", "Start music"],
    "tat_nhac":     ["Music stop", "Stop music", "Pause music"]
}

# ──────────────────────────────────────────
# 6 giọng đọc Edge TTS (3 Nam + 3 Nữ)
# Tạo sự đa dạng giọng cho model
# ──────────────────────────────────────────
VOICES = [
    "en-US-ChristopherNeural",  # Nam - trầm ấm
    "en-US-GuyNeural",          # Nam - trung tính
    "en-US-EricNeural",         # Nam - trẻ
    "en-US-AriaNeural",         # Nữ - thanh thoát
    "en-US-JennyNeural",        # Nữ - chuyên nghiệp
    "en-US-MichelleNeural",     # Nữ - ấm áp
]

# ==========================================
# 2. DATA AUGMENTATION (Nhân bản đa dạng)
# ==========================================
def augment_audio(y, sr):
    """Biến đổi ngẫu nhiên tín hiệu âm thanh để tạo đa dạng."""
    augmented = y.copy()

    # --- Thay đổi tốc độ nói (0.85x → 1.25x) ---
    speed = random.uniform(0.85, 1.25)
    augmented = librosa.effects.time_stretch(augmented, rate=speed)

    # --- Thay đổi cao độ giọng (-3 → +3 semitones) ---
    pitch = random.randint(-3, 3)
    if pitch != 0:
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=pitch)

    # --- Thêm nhiễu trắng ngẫu nhiên ---
    noise_level = random.uniform(0.002, 0.012)
    noise = np.random.normal(0, noise_level, len(augmented))
    augmented = augmented + noise

    # --- Thay đổi âm lượng (gain) ---
    gain = random.uniform(0.7, 1.4)
    augmented = augmented * gain

    # --- Clip để tránh vượt biên [-1, 1] ---
    augmented = np.clip(augmented, -1.0, 1.0)

    return augmented.astype(np.float32)


# ==========================================
# 3. TẢI GIỌNG NÓI TỪ EDGE TTS (ASYNC)
# ==========================================
async def create_tts_file(text, voice, output_path):
    """Gọi Microsoft Edge TTS để tạo file giọng nói."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


# ==========================================
# 4. HÀM SẢN XUẤT DATASET CHÍNH
# ==========================================
def generate_dataset():
    start_time = time.time()
    total_files = 0

    print("=" * 60)
    print("🚀 BẮT ĐẦU TẠO DATASET EDGE-TTS CHO SMART HOME")
    print("=" * 60)
    print(f"📂 Thư mục lưu: {DATASET_DIR}/")
    print(f"🏷️  Số nhãn:      {len(LENH_SMART_HOME)}")
    print(f"🎙️  Số giọng:     {len(VOICES)}")
    print(f"📝 Số câu/nhãn:   {len(list(SENTENCE_VARIANTS.values())[0])}")
    print(f"🧬 Augment/giọng: {AUGMENT_PER_VOICE}")

    expected = len(LENH_SMART_HOME) * len(VOICES) * len(list(SENTENCE_VARIANTS.values())[0]) * AUGMENT_PER_VOICE
    print(f"📊 Dự kiến tổng:  ~{expected} file lệnh + 300 noise")
    print("=" * 60)

    os.makedirs(DATASET_DIR, exist_ok=True)

    # ── Tạo dữ liệu cho từng nhãn lệnh ──
    for label_idx, (label, base_text) in enumerate(LENH_SMART_HOME.items(), 1):
        folder = os.path.join(DATASET_DIR, label)
        os.makedirs(folder, exist_ok=True)

        variants = SENTENCE_VARIANTS.get(label, [base_text])
        print(f"\n[{label_idx}/{len(LENH_SMART_HOME)}] 🎙️  Nhãn: {label}  →  \"{base_text}\"")
        print(f"   Biến thể câu: {variants}")

        file_count = 0

        for voice in VOICES:
            voice_short = voice.split("-")[-1].replace("Neural", "")

            for text in variants:
                # Tạo file TTS gốc
                temp_mp3 = f"_temp_{label}_{voice_short}.mp3"
                try:
                    asyncio.run(create_tts_file(text, voice, temp_mp3))
                except Exception as e:
                    print(f"   ⚠️  Lỗi TTS ({voice_short}, \"{text}\"): {e}")
                    continue

                # Đọc và resample về 16kHz
                try:
                    y_original, sr = librosa.load(temp_mp3, sr=16000)
                except Exception as e:
                    print(f"   ⚠️  Lỗi đọc audio: {e}")
                    if os.path.exists(temp_mp3):
                        os.remove(temp_mp3)
                    continue

                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)

                # Lưu bản gốc (không augment)
                original_path = os.path.join(folder, f"{label}_{file_count:05d}.wav")
                sf.write(original_path, y_original, sr)
                file_count += 1

                # Nhân bản với augmentation
                for _ in range(AUGMENT_PER_VOICE):
                    y_aug = augment_audio(y_original, sr)
                    aug_path = os.path.join(folder, f"{label}_{file_count:05d}.wav")
                    sf.write(aug_path, y_aug, sr)
                    file_count += 1

            print(f"   ✅ {voice_short}: {file_count} file tích lũy")

        total_files += file_count
        print(f"   📦 Tổng nhãn [{label}]: {file_count} file")

    # ── Tạo thư mục Noise ──
    print(f"\n{'=' * 60}")
    print("🌪️  Đang tạo dữ liệu Noise (tạp âm)...")
    noise_folder = os.path.join(DATASET_DIR, "noise")
    os.makedirs(noise_folder, exist_ok=True)

    noise_count = 300
    for i in range(noise_count):
        duration = random.uniform(0.8, 2.5)  # Độ dài ngẫu nhiên
        noise_type = random.choice(["white", "pink", "brown"])

        if noise_type == "white":
            signal = np.random.normal(0, 0.08, int(16000 * duration))
        elif noise_type == "pink":
            # Xấp xỉ pink noise bằng bộ lọc
            white = np.random.normal(0, 0.08, int(16000 * duration))
            b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
            a = [1.000000000, -2.494956002, 2.017265875, -0.522189400]
            from scipy.signal import lfilter
            signal = lfilter(b, a, white)
        else:  # brown
            signal = np.cumsum(np.random.normal(0, 0.003, int(16000 * duration)))
            signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.15

        signal = signal.astype(np.float32)
        sf.write(os.path.join(noise_folder, f"noise_{i:04d}.wav"), signal, 16000)

    total_files += noise_count

    # ── Kết quả ──
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"🎉 HOÀN TẤT TẠO DATASET!")
    print(f"   📊 Tổng số file: {total_files}")
    print(f"   ⏱️  Thời gian:    {elapsed:.1f}s ({elapsed/60:.1f} phút)")
    print(f"   📂 Lưu tại:      {os.path.abspath(DATASET_DIR)}/")
    print(f"{'=' * 60}")
    print(f"\n👉 Bước tiếp theo: chạy  python Train.py  để huấn luyện model!")


if __name__ == "__main__":
    generate_dataset()