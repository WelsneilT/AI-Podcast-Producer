# test_tts.py (PHIÊN BẢN TEST mina_voice.wav)
import os
import torch

# Đặt biến môi trường nếu cần
os.environ['TRANSFORMERS_ALLOW_TORCH_LOAD'] = '1'

from TTS.api import TTS

print("--- Bắt đầu kiểm tra Coqui TTS với giọng mẫu 'mina_voice.wav' ---")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

try:
    tts = TTS(model_name, progress_bar=True).to(device)
    print("\n--- Model đã được tải thành công từ cache ---")
    
    # === THAY ĐỔI Ở ĐÂY: Trỏ đến file giọng mẫu mới ===
    speaker_wav_path = os.path.join(ASSETS_DIR, 'mina_voice.wav')
    
    if not os.path.exists(speaker_wav_path):
        raise FileNotFoundError(f"Không tìm thấy file giọng mẫu tại: {speaker_wav_path}")

    print(f"--- Sử dụng giọng mẫu từ: {speaker_wav_path} ---")

    # Thử tạo một file audio ngắn
    print("--- Thử tạo file audio test_mina.wav ---")
    tts.tts_to_file(
        text="Hello, this is a test using Mina's voice. Does it sound natural and clear?", 
        file_path="test_mina.wav",
        speaker_wav=speaker_wav_path,
        language="en"
    )
    
    print("\n✅ THÀNH CÔNG! Model XTTS hoạt động với giọng mẫu mới.")
    print("Hãy nghe thử file 'test_mina.wav' để kiểm tra chất lượng.")
    
except Exception as e:
    print(f"\n🛑 LỖI: {e}")