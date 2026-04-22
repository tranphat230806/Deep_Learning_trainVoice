from flask import Flask, jsonify
from flask_cors import CORS
import random


# RNNs: Nhận diện giọng nói
# giới thiệu bài toán
# Giới thiệu bộ dữ liệu(kích thước, thời gian, lấy ở đâu,...)
# Có 3 dạng dữ liệu train, test, dữ liệu thực. Độ chính xác từ
#  dữ liệu gốc từ đó đánh giá. Lấy dữ liệu mẫu để test, so sánh các
#  thuật toán nếu dùng nhiều thuật toán, lấy mẫu thuật toán ngoài so 
# sánh các thuật toán đã làm. Nhớ có giới thiệu, sơ đồ biểu đồ, đánh giá thuật
#  toán khi lm xong kết quả.	
	


app = Flask(__name__)
CORS(app)

@app.route("/verify", methods=["POST"])
def verify():
    # giả lập AI xác thực giọng nói
    result = random.choice([True, False])

    return jsonify({
        "verified": result
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)