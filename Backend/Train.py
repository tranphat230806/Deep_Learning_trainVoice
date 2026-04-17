from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import glob
import tempfile
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

app = Flask(__name__)
CORS(app)

# =============================
# CONFIG
# =============================
MY_FOLDER = r"E:\LuuTruSinhVien\LuuTruPyThon\doAnDeep\Data\my_voice"
THRESHOLD = 0.78

print("Đang load AI model...")
encoder = VoiceEncoder()

# =============================
# ENROLL GIỌNG CỦA BẠN
# =============================
def build_my_voice():
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

    embeds = []

    for file in files:
        wav = preprocess_wav(file)
        embed = encoder.embed_utterance(wav)
        embeds.append(embed)

    return np.mean(embeds, axis=0)

my_voice_embed = build_my_voice()

print("Đã học giọng chủ nhà")

# =============================
# VERIFY API
# =============================
@app.route("/verify", methods=["POST"])
def verify():
    if "audio" not in request.files:
        return jsonify({
            "verified": False,
            "message": "Không có audio"
        })

    file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        wav = preprocess_wav(temp_path)
        test_embed = encoder.embed_utterance(wav)

        score = np.dot(my_voice_embed, test_embed) / (
            np.linalg.norm(my_voice_embed) *
            np.linalg.norm(test_embed)
        )

        verified = score >= THRESHOLD

        return jsonify({
            "verified": bool(verified),
            "score": float(score)
        })

    finally:
        os.remove(temp_path)

# =============================
if __name__ == "__main__":
    app.run(debug=True, port=5000)