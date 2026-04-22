from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import glob
import tempfile
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

app = Flask(__name__)
CORS(app)

# Phần Train này xây dụng kiến trúc LSTM