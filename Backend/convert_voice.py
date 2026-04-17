from pydub import AudioSegment
import os

folder = r"E:\LuuTruSinhVien\LuuTruPyThon\doAnDeep\Data\my_voice"

for file in os.listdir(folder):
    print("Đang chạy convert_voice.py...")
    if file.endswith(".wav") or file.endswith(".mp3") or file.endswith(".m4a"):
        path = os.path.join(folder, file)
        
        try:
            audio = AudioSegment.from_file(path)

            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            audio = audio.set_sample_width(2)

            new_name = os.path.splitext(file)[0] + "_fixed.wav"
            save_path = os.path.join(folder, new_name)

            audio.export(save_path, format="wav")

            print("OK:", new_name)

        except Exception as e:
            print("Lỗi:", file, e)