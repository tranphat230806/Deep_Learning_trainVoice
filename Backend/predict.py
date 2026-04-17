import os
import glob
import numpy as np
import sounddevice as sd
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav

# =====================================
# CONFIG
# =====================================
MY_FOLDER = r"E:\LuuTruSinhVien\LuuTruPyThon\doAnDeep\Data\my_voice"
TEMP_FILE = "mic_test.wav"

SR = 16000
DURATION = 4
THRESHOLD = 0.78

# =====================================
# LOAD MODEL
# =====================================
print("Đang load model...")
encoder = VoiceEncoder()

# =====================================
# CHỈ LẤY FILE GỐC, BỎ _fixed
# =====================================
files = []

for f in glob.glob(os.path.join(MY_FOLDER, "*.wav")):
    name = os.path.basename(f).lower()

    if "_fixed" in name:
        continue

    if "test" in name:
        continue

    if "mic" in name:
        continue

    files.append(f)

print("Danh sách file dùng để học:")
for f in files:
    print("-", os.path.basename(f))

print("Tổng file:", len(files))

# =====================================
# ENROLL GIỌNG CỦA BẠN
# =====================================
embeds = []

for file in files:
    wav = preprocess_wav(file)
    embed = encoder.embed_utterance(wav)
    embeds.append(embed)

my_voice_embed = np.mean(embeds, axis=0)

print("Đã học xong giọng của bạn")

# =====================================
# RECORD MIC
# =====================================
input("Nhấn Enter để bắt đầu nói...")

print("🎤 Đang ghi âm...")

audio = sd.rec(
    int(DURATION * SR),
    samplerate=SR,
    channels=1,
    dtype="float32"
)

sd.wait()

sf.write(TEMP_FILE, audio, SR)

print("Đã ghi xong!")

# =====================================
# VERIFY
# =====================================
wav = preprocess_wav(TEMP_FILE)
test_embed = encoder.embed_utterance(wav)

score = np.dot(my_voice_embed, test_embed) / (
    np.linalg.norm(my_voice_embed) *
    np.linalg.norm(test_embed)
)

print("-" * 40)
print("Similarity Score:", round(float(score), 4))
print("Threshold       :", THRESHOLD)
print("-" * 40)

if score >= THRESHOLD:
    print("✅ XÁC NHẬN: ĐÚNG LÀ BẠN")
else:
    print("❌ KHÔNG PHẢI BẠN")