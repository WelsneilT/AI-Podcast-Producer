# tts_server.py (PHIÊN BẢN HOÀN CHỈNH - SỬA LỖI TẢI MODEL PYTORCH V4)

import os
import torch
import tempfile
import logging
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

# Cố gắng import TTS
try:
    from TTS.api import TTS
except ImportError:
    print("🛑 LỖI: Không tìm thấy thư viện Coqui TTS...")
    exit()

# === SỬA LỖI TẢI MODEL PYTORCH (PHIÊN BẢN ĐẦY ĐỦ NHẤT) ===
# Thêm TẤT CẢ các class tùy chỉnh mà XTTS cần vào "danh sách trắng".
try:
    from TTS.tts.configs import xtts_config
    from TTS.tts.models import xtts
    from TTS.config import shared_configs
    
    # Tạo một danh sách các class cần cho phép
    safe_globals_list = [
        xtts_config.XttsConfig,
        xtts.XttsAudioConfig,
        shared_configs.BaseDatasetConfig,
        xtts.XttsArgs # <-- THÊM CLASS THỨ TƯ VÀO ĐÂY
    ]
    
    torch.serialization.add_safe_globals(safe_globals_list)
    print("✅ Đã thêm các class Coqui TTS vào danh sách an toàn của PyTorch.")
except Exception as e:
    print(f"⚠️ Cảnh báo: Không thể thêm các class Coqui vào danh sách an toàn. Lỗi: {e}")
    pass

# Tắt log của Flask
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

print("--- Khởi động TTS Server ---")

# --- Tải Model một lần duy nhất ---
tts = None
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    print(f"Đang tải model Coqui TTS: {model_name} (sử dụng device: {device})...")
    
    tts = TTS(model_name).to(device)
    print("✅ Model TTS đã sẵn sàng.")
except Exception as e:
    print(f"🛑 LỖI NGHIÊM TRỌNG: Không thể tải model TTS. Lỗi: {e}")

# --- Khởi tạo Flask App ---
app = Flask(__name__)
CORS(app)

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    """Endpoint chính để nhận văn bản và trả về file audio."""
    if not tts:
        return jsonify({"error": "TTS model is not available due to a startup error."}), 503

    if not request.is_json:
        return jsonify({"error": "Invalid request: Content-Type must be application/json."}), 415

    data = request.json
    text = data.get('text')
    speaker_wav_path = data.get('speaker_wav')

    if not text or not speaker_wav_path:
        return jsonify({"error": "Missing 'text' or 'speaker_wav' in request body."}), 400

    if not os.path.exists(speaker_wav_path):
         return jsonify({"error": f"Speaker WAV file not found at the provided path: {speaker_wav_path}"}), 400

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            output_path = tmpfile.name
        
        print(f"Đang tạo audio cho: '{text[:40]}...'")
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav_path,
            language="en",
            file_path=output_path
        )
        print(f"Tạo audio thành công: {os.path.basename(output_path)}")

        return send_file(output_path, as_attachment=True, mimetype='audio/wav')

    except Exception as e:
        print(f"🛑 Lỗi trong quá trình tạo audio: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"TTS Server đang lắng nghe tại http://127.0.0.1:5001")
    app.run(host='0.0.0.0', port=5001)