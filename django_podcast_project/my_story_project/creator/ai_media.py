# creator/ai_media.py (PHIÊN BẢN TỐI GIẢN - CHỈ TẠO ẢNH GỐC)

import os
import torch
import gc
import requests
from django.conf import settings
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from pydub import AudioSegment
from PIL import Image

# --- Cấu hình và Biến toàn cục ---
print("⚙️ [ai_media.py] Khởi tạo module xử lý media (Ảnh + TTS Client)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
image_gen_pipe, refiner_pipe = None, None

# ===================================================================
# --- QUẢN LÝ MÔ HÌNH ẢNH ---
# ===================================================================
def load_image_gen_model():
    """Tải model nền (Base) và model tinh chỉnh (Refiner). Bỏ Inpainting."""
    global image_gen_pipe, refiner_pipe
    if image_gen_pipe and refiner_pipe: 
        return image_gen_pipe, refiner_pipe, None # Trả về None cho vị trí thứ 3 để code cũ không lỗi

    # --- Model Base ---
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0" 
    print(f"\n--- Tải model AI nền: {base_model_id} ---")
    base = DiffusionPipeline.from_pretrained(
        base_model_id, torch_dtype=torch.float16, use_safetensors=True
    )
    base.enable_model_cpu_offload()
    image_gen_pipe = base
    print(f"✅ Model nền {os.path.basename(base_model_id)} đã sẵn sàng.")

    # --- Model Refiner ---
    refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
    print(f"\n--- Tải model AI tinh chỉnh: {refiner_model_id} ---")
    refiner = DiffusionPipeline.from_pretrained(
        refiner_model_id, text_encoder_2=base.text_encoder_2, vae=base.vae, 
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    refiner.enable_model_cpu_offload()
    refiner_pipe = refiner
    print(f"✅ Model tinh chỉnh {os.path.basename(refiner_model_id)} đã sẵn sàng.")
    
    return image_gen_pipe, refiner_pipe, None # Thêm None để tương thích với lời gọi hàm

# Đổi tên hàm generate_anchor_image lại thành tên gốc và duy nhất
def generate_image_locally(prompt, negative_prompt, output_path, image_pipe, refiner_pipe):
    """Hàm tạo ảnh duy nhất, sử dụng chu trình Base + Refiner."""
    print(f"    -> 🎨 [Image Gen] Đang vẽ ảnh (Base + Refiner): '{prompt[:70]}...'")
    n_steps, high_noise_frac = 40, 0.8
    latent_image = image_pipe(
        prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=n_steps, 
        denoising_end=high_noise_frac, output_type="latent", guidance_scale=8.5, 
        height=720, width=1280
    ).images
    image = refiner_pipe(
        prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=n_steps, 
        denoising_start=high_noise_frac, image=latent_image
    ).images[0]
    
    image.save(output_path)
    print(f"    -> ✅ [Image Gen] Đã lưu ảnh: {os.path.basename(output_path)}")
    return output_path

# ===================================================================
# --- QUẢN LÝ GIỌNG NÓI (GIỮ NGUYÊN) ---
# ===================================================================
def text_to_speech(text, output_path):
    """Gửi yêu cầu đến TTS Server để tạo file âm thanh."""
    print(f"    -> 🗣️ [TTS Client] Gửi yêu cầu đến TTS Server cho: '{text[:50]}...'")
    tts_server_url = "http://127.0.0.1:5001/synthesize"
    speaker_wav_path = os.path.join(settings.ASSETS_DIR, 'mina_voice.wav')
    payload = {"text": text, "speaker_wav": speaker_wav_path}
    try:
        response = requests.post(tts_server_url, json=payload, timeout=300)
        if response.status_code == 200:
            with open(output_path, 'wb') as f: f.write(response.content)
            print(f"    -> ✅ [TTS Client] Đã nhận và lưu audio: {os.path.basename(output_path)}")
            return output_path
        else:
            print(f"    -> 🛑 [TTS Client] TTS Server báo lỗi: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"    -> 🛑 [TTS Client] Không thể kết nối đến TTS Server: {e}")
        return None

# ===================================================================
# --- XỬ LÝ NHẠC NỀN (GIỮ NGUYÊN) ---
# ===================================================================
def add_background_music(narration_path, final_output_path):
    """Trộn giọng nói với nhạc nền."""
    print(f"    -> 🎵 [Audio Mixer] Trộn nhạc nền cho file: {os.path.basename(narration_path)}...")
    try:
        music_path = os.path.join(settings.ASSETS_DIR, 'background.mp3')
        narration_audio = AudioSegment.from_wav(narration_path)
        background_music = AudioSegment.from_mp3(music_path) - 12
        if len(background_music) < len(narration_audio):
            background_music *= -(-len(narration_audio) // len(background_music))
        background_music = background_music[:len(narration_audio)]
        final_audio = narration_audio.overlay(background_music)
        final_audio.export(final_output_path, format="mp3")
        print(f"    -> ✅ [Audio Mixer] Đã lưu audio hoàn chỉnh: {os.path.basename(final_output_path)}")
        return final_output_path
    except Exception as e:
        print(f"    -> 🛑 [Audio Mixer] Lỗi khi trộn nhạc: {e}")
        narration_audio.export(final_output_path, format="mp3")
        return final_output_path

# ===================================================================
# --- QUẢN LÝ BỘ NHỚ (GIỮ NGUYÊN) ---
# ===================================================================
def unload_all_ai_models():
    """Gỡ TẤT CẢ các model AI tại chỗ."""
    global image_gen_pipe, refiner_pipe
    if image_gen_pipe or refiner_pipe:
        print("\n🧹 [Memory Optimizer] Unloading local AI models...")
    
    if image_gen_pipe is not None: del image_gen_pipe; image_gen_pipe = None
    if refiner_pipe is not None: del refiner_pipe; refiner_pipe = None

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("✅ [Memory Optimizer] Memory has been freed.")