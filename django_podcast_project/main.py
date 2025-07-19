# main.py - AI Podcast & Storyboard Producer (Local Version 6.0 - Content-Rich)

# ====================================================
# PHẦN 1: IMPORT, CẤU HÌNH
# ====================================================
print("⚙️ Khởi tạo hệ thống sản xuất video AI...")

import os, re, json, time, requests, shutil, torch, soundfile as sf, ffmpeg
from groq import Groq
from bs4 import BeautifulSoup
from pydub import AudioSegment
from unidecode import unidecode
from diffusers import DiffusionPipeline
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from dotenv import load_dotenv

load_dotenv()

try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key: raise ValueError("GROQ_API_KEY không được tìm thấy trong file .env")
    groq_client = Groq(api_key=groq_api_key)
    print("✅ Đã cấu hình Groq API Key.")
except Exception as e:
    print(f"🛑 LỖI cấu hình API: {e}"); exit()

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu": print("⚠️ CẢNH BÁO: Không tìm thấy GPU, chương trình sẽ chạy rất chậm.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ Thiết bị: {device.upper()}. Thư mục output: {OUTPUT_DIR}")

tts_model_pack = None
image_gen_pipe = None

# ====================================================
# PHẦN 2: THƯ VIỆN HÀM CHỨC NĂNG
# ====================================================
print("🛠️ Đang định nghĩa thư viện hàm sản xuất...")

def load_ai_models():
    global tts_model_pack, image_gen_pipe
    print("\n--- Tải các mô hình AI (chỉ lần đầu, sẽ mất nhiều thời gian) ---")
    if tts_model_pack is None:
        print("🎙️ [TTS] Đang tải mô hình giọng nói...")
        try:
            processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
            embeddings_file_path = os.path.join(MODELS_DIR, "speaker_embeddings.pt")
            if not os.path.exists(embeddings_file_path): raise FileNotFoundError(f"LỖI: Không tìm thấy file '{embeddings_file_path}'.")
            speaker_embeddings = torch.load(embeddings_file_path, map_location=device)
            voices = {"Alex": speaker_embeddings[7306].unsqueeze(0), "Ben": speaker_embeddings[1234].unsqueeze(0)}
            tts_model_pack = {"processor": processor, "model": model, "vocoder": vocoder, "voices": voices}
            print("✅ [TTS] Mô hình giọng nói đã sẵn sàng.")
        except Exception as e: print(f"🛑 Lỗi khi tải mô hình TTS: {e}"); raise e
    if image_gen_pipe is None:
        print("🎨 [Image Gen] Đang tải mô hình Dreamshaper 8...")
        try:
            model_id = "Lykon/dreamshaper-8"
            pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
            pipe.to(device)
            image_gen_pipe = pipe
            print("✅ [Image Gen] Mô hình tạo ảnh đã sẵn sàng.")
        except Exception as e: print(f"🛑 Lỗi khi tải mô hình Image Gen: {e}"); raise e
    print("--- Tất cả mô hình AI đã được tải ---")

def get_content_from_url(url):
    print(f"  -> 🔎 [Scraper] Lấy nội dung từ: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']): tag.decompose()
        text = " ".join(t for t in soup.get_text(separator=' ').split() if t)
        print("  -> ✅ [Scraper] Lấy nội dung thành công.")
        return text
    except Exception as e: print(f"  -> 🛑 [Scraper] Lỗi: {e}"); return None

def generate_podcast_script_from_content(content):
    """NÂNG CẤP: Tạo ra một kịch bản hội thoại thực sự."""
    print("  -> ✍️ [LLM-Scriptwriter] Tạo kịch bản hội thoại chi tiết...")
    script_prompt = f"You are a master podcast scriptwriter. Based on the following text, create an engaging and natural-sounding conversational script in ENGLISH for two hosts, Alex (the curious host) and Ben (the expert). The script MUST have at least 8 to 10 back-and-forth conversational turns to make it interesting. Each line MUST strictly start with 'Alex:' or 'Ben:'. Do not add any titles, episode names, or markdown formatting. Just the raw dialogue.\n\nText to base the script on:\n---\n{content[:5000]}\n---"
    try:
        completion = groq_client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": script_prompt}])
        raw_script_text = completion.choices[0].message.content
        cleaned_lines = [f"{m.group(1)}: {m.group(2)}" for line in raw_script_text.split('\n') if (m := re.match(r"^\*?\*?(Alex|Ben)\*?\*?:\s*(.*)", line.strip()))]
        if not cleaned_lines: print("  -> 🛑 [LLM-Scriptwriter] Không thể trích xuất được dòng thoại hợp lệ."); return None
        final_script = "\n".join(cleaned_lines)
        print("  -> ✅ [LLM-Scriptwriter] Kịch bản đã được tạo và làm sạch.")
        return final_script
    except Exception as e: print(f"  -> 🛑 [LLM-Scriptwriter] Lỗi khi tạo kịch bản: {e}"); return None

def create_storyboard_from_script(script_text, overall_style):
    """NÂNG CẤP V3: Ép AI phải xử lý từng dòng thoại, không được gộp."""
    print(f"  -> 🎬 [Director AI] Phân tích kịch bản TỪNG DÒNG một...")
    
    # --- PROMPT "ĐẠO DIỄN" SIÊU NGHIÊM KHẮC ---
    system_prompt = f"""
    You are a meticulous AI Film Director. Your task is to process a podcast script LINE BY LINE and create a visual scene for EACH line of dialogue.

    **CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:**

    1.  **PROCESS EACH LINE INDIVIDUALLY:** You will be given a script. For **EVERY SINGLE LINE** of dialogue (e.g., "Alex: ...", "Ben: ..."), you must create one corresponding scene object in the output array. DO NOT skip any lines. DO NOT group lines together. One line of dialogue equals one scene.
    2.  **VISUAL STORYTELLING:** When the dialogue **describes** a place, event, or concept, the image prompt **MUST** visualize that concept. DO NOT show the host's face.
    3.  **REACTION SHOTS:** Only generate an image of a host for short, emotional reactions or direct questions (e.g., "Wow, that's amazing!").
    4.  **STYLE CONSISTENCY:** The entire storyboard MUST be in the style of: **'{overall_style}'**. This style MUST be in every image prompt.
    5.  **JSON OUTPUT:** You MUST return a single, valid JSON object with one key: "storyboard". The value MUST be an array of objects. The number of objects in the array must be equal to the number of dialogue lines in the input script.
    6.  **MANDATORY FIELDS:** EACH object in the array MUST contain these THREE keys: "host", "dialogue", and "image_prompt". All keys must have non-empty string values.
    """
    
    try:
        # Lấy các dòng thoại để đưa vào prompt, giúp LLM tập trung
        dialogue_lines = "\n".join([line for line in script_text.split('\n') if re.match(r"^(Alex|Ben):", line.strip())])
        
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Here is the script. Process it line by line:\n\n{dialogue_lines}"}],
            response_format={"type": "json_object"}
        )
        
        raw_json = json.loads(completion.choices[0].message.content)
        storyboard_scenes = raw_json.get("storyboard", [])
        
        # Validation vẫn giữ nguyên để đảm bảo chất lượng
        validated_scenes = [s for s in storyboard_scenes if isinstance(s, dict) and all(k in s and s[k] for k in ['host', 'dialogue', 'image_prompt'])]
        
        if not validated_scenes:
            print("  -> 🛑 [Director AI] Không có cảnh nào hợp lệ được tạo ra.")
            return None
            
        print(f"  -> ✅ [Director AI] Tạo và xác thực {len(validated_scenes)} cảnh thành công!")
        return validated_scenes
    except Exception as e:
        print(f"  -> 🛑 [Director AI] Lỗi nghiêm trọng khi tạo storyboard: {e}")
        return None

def generate_image_locally(prompt, filename):
    print(f"    -> 🎨 [Image Gen] Đang vẽ ảnh: '{prompt[:60]}...'")
    try:
        negative_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, bad anatomy, watermark, signature, low quality, text, letters"
        image = image_gen_pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, height=768, width=768).images[0]
        image.save(filename)
        print(f"    -> ✅ [Image Gen] Đã lưu ảnh: {os.path.basename(filename)}"); return filename
    except Exception as e: print(f"    -> 🛑 [Image Gen] Lỗi khi tạo ảnh: {e}"); return None

def text_to_speech(text, host, output_path):
    print(f"    -> 🗣️ [TTS] Tạo audio cho '{host}': '{text[:50]}...'")
    processed_text = unidecode(text)
    inputs = tts_model_pack["processor"](text=processed_text, return_tensors="pt").to(device)
    speech = tts_model_pack["model"].generate_speech(inputs["input_ids"], tts_model_pack["voices"][host], vocoder=tts_model_pack["vocoder"])
    sf.write(output_path, speech.cpu().numpy(), 16000)
    print(f"    -> ✅ [TTS] Đã lưu audio: {os.path.basename(output_path)}"); return output_path

def assemble_video_with_ffmpeg(audio_clips_info, image_paths, assets_dir, output_path):
    print("\n🎬 Bắt đầu công đoạn dựng video với FFmpeg...")
    video_inputs, audio_inputs = [], []
    TARGET_SIZE = '768x768'
    for i, img_path in enumerate(image_paths):
        if i < len(audio_clips_info) and os.path.exists(img_path):
            duration = audio_clips_info[i]['duration']
            print(f"  -> Chuẩn bị cảnh {i+1}, thời lượng {duration:.2f} giây.")
            video_stream = (ffmpeg.input(img_path, loop=1, t=duration, framerate=24).filter('zoompan', z='min(zoom+0.001,1.15)', d=1, x='iw/2-(iw/zoom/2)', y='ih/2-(ih/zoom/2)', fps=24, s=TARGET_SIZE))
            video_inputs.append(video_stream)
            audio_inputs.append(ffmpeg.input(audio_clips_info[i]['path']))
    if not video_inputs: raise ValueError("Không có clip hình ảnh nào hợp lệ để dựng.")
    
    concatenated_video = ffmpeg.concat(*video_inputs, v=1, a=0).node
    concatenated_audio = ffmpeg.concat(*audio_inputs, v=0, a=1).node
    final_audio = concatenated_audio[0]
    try:
        bgm_path = os.path.join(assets_dir, "background.mp3")
        if os.path.exists(bgm_path):
            print("  -> Thêm nhạc nền...")
            bgm_stream = ffmpeg.input(bgm_path, stream_loop=-1).audio
            final_audio = ffmpeg.filter([concatenated_audio[0], bgm_stream], 'amix', inputs=2, duration='first', dropout_transition=1, weights="1 0.15")
    except Exception as e: print(f"  - Cảnh báo: Không thể thêm nhạc nền. Lỗi: {e}")

    print(f"  -> Đang xuất video ra file: {output_path}")
    (ffmpeg.output(concatenated_video[0], final_audio, output_path, vcodec='libx264', acodec='aac', shortest=None, pix_fmt='yuv420p').overwrite_output().run(quiet=True))
    print(f"✅ Đã dựng xong video!"); return output_path

# ====================================================
# PHẦN 3: DÂY CHUYỀN SẢN XUẤT CHÍNH
# ====================================================
def run_production_pipeline():
    print("\n" + "="*60 + "\n🚀 BẮT ĐẦU DÂY CHUYỀN SẢN XUẤT AUDIO-VISUAL...\n" + "="*60)
    
    URL_INPUT = input(">> Nhập URL bài viết để tạo video: ")
    print(">> Chọn phong cách nghệ thuật:")
    style_map = {'1': 'cinematic, dramatic, photorealistic', '2': 'Ghibli studio anime style, beautiful scenery', '3': 'impressionistic oil painting, vibrant colors'}
    for key, value in style_map.items(): print(f"   {key}: {value}")
    STYLE_CHOICE = input(">> Lựa chọn của bạn (1, 2, 3): ")
    OVERALL_STYLE = style_map.get(STYLE_CHOICE, 'cinematic, dramatic, photorealistic')
    EPISODE_NAME = input(">> Nhập tên cho sản phẩm (không dấu, không cách): ") or "MyFirstLocalVideo"
    
    print("\n" + "="*60); print(f"🚀 Bắt đầu tạo sản phẩm '{EPISODE_NAME}'..."); print("="*60 + "\n")

    episode_output_dir = os.path.join(OUTPUT_DIR, EPISODE_NAME)
    audio_temp_dir = os.path.join(episode_output_dir, 'audio')
    image_temp_dir = os.path.join(episode_output_dir, 'images')
    if os.path.exists(episode_output_dir): shutil.rmtree(episode_output_dir)
    os.makedirs(audio_temp_dir); os.makedirs(image_temp_dir)
    
    final_video_path = None
    try:
        print("--- BƯỚC 1: NGHIÊN CỨU & VIẾT KỊCH BẢN ---")
        content = get_content_from_url(URL_INPUT)
        if not content: raise ValueError("Không lấy được nội dung.")
        script_text = generate_podcast_script_from_content(content)
        if not script_text: raise ValueError("Không tạo được kịch bản.")
        print("   -> Kịch bản đã được tạo:\n---\n" + script_text + "\n---")

        print("\n--- BƯỚC 2: TẠO STORYBOARD ---")
        storyboard_scenes = create_storyboard_from_script(script_text, OVERALL_STYLE)
        if not storyboard_scenes: raise ValueError("Không tạo được storyboard hợp lệ.")
        
        print("\n--- BƯỚC 3: SẢN XUẤT ÂM THANH & HÌNH ẢNH ---")
        audio_clips_info, image_paths = [], []
        for i, scene in enumerate(storyboard_scenes):
            print(f"\n-> Đang xử lý Cảnh {i+1}/{len(storyboard_scenes)}...")
            try:
                dialogue, host, prompt = scene['dialogue'], scene['host'], scene['image_prompt']
                audio_path = os.path.join(audio_temp_dir, f"line_{i:03d}.wav")
                text_to_speech(dialogue, host, audio_path)
                image_path = os.path.join(image_temp_dir, f"img_{i:03d}.png")
                generated_img = generate_image_locally(prompt, image_path)
                if generated_img and os.path.exists(audio_path):
                    audio_clips_info.append({'path': audio_path, 'duration': AudioSegment.from_wav(audio_path).duration_seconds})
                    image_paths.append(generated_img)
                    print(f"   -> ✅ Hoàn thành xử lý Cảnh {i+1}.")
                else: raise ValueError("Không thể tạo audio hoặc image.")
            except Exception as scene_error: print(f"   -> ⚠️ Bỏ qua Cảnh {i+1} do lỗi: {scene_error}")
        
        print(f"\n--- BƯỚC 4: DỰNG VIDEO ({len(image_paths)} cảnh hợp lệ) ---")
        if not image_paths: raise ValueError("Không có cảnh nào hợp lệ để dựng video.")
        final_video_path = os.path.join(episode_output_dir, f"{EPISODE_NAME}.mp4")
        assemble_video_with_ffmpeg(audio_clips_info, image_paths, ASSETS_DIR, final_video_path)
        
        print("\n\n🎉🎉🎉 HOÀN TẤT! 🎉🎉🎉")
        print(f"Video đã được lưu tại: {os.path.abspath(final_video_path)}")

    except Exception as e:
        print(f"\n🛑🛑🛑 LỖI DÂY CHUYỀN: {e}")

# ====================================================
# PHẦN 4: ĐIỂM BẮT ĐẦU CHƯƠNG TRÌNH
# ====================================================
if __name__ == "__main__":
    try:
        load_ai_models()
        run_production_pipeline()
    except Exception as e:
        print(f"\nMột lỗi nghiêm trọng đã xảy ra khi khởi động: {e}")