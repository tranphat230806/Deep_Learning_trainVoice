from flask import Flask, jsonify
from flask_cors import CORS
import random

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