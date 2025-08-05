# test/local_image_test.py (PHIÊN BẢN KIỂM TRA LORA CHO CONAN)

import torch
from diffusers import DiffusionPipeline
import os

print("--- BẮT ĐẦU BÀI KIỂM TRA LORA CHO CONAN EDOGAWA ---")

# --- 1. Kiểm tra GPU ---
if torch.cuda.is_available():
    print(f"✅ GPU được nhận diện: {torch.cuda.get_device_name(0)}")
else:
    print("🛑 Không tìm thấy GPU."); exit()

# --- 2. Tải mô hình nền Animagine XL ---
model_id = "cagliostrolab/animagine-xl-3.1"
print(f"\n⏳ Đang tải mô hình nền '{model_id}'...")
try:
    pipe = DiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.enable_model_cpu_offload()
    print("✅ Tải mô hình nền thành công.")
except Exception as e:
    print(f"🛑 Lỗi khi tải mô hình nền: {e}"); exit()

# --- 3. Tải file LoRA của Conan ---
lora_filename = "conan.safetensors"
lora_path = os.path.join("my_story_project", "loras", lora_filename)

print(f"\n⏳ Đang tải LoRA '{lora_filename}'...")
if not os.path.exists(lora_path):
    print(f"🛑 Lỗi: Không tìm thấy file LoRA tại '{os.path.abspath(lora_path)}'.")
    exit()

try:
    adapter_name = os.path.splitext(lora_filename)[0]
    pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
    pipe.set_adapters([adapter_name])
    # Không fuse LoRA để giữ lại khả năng kiểm soát tốt hơn
    print("✅ Tải LoRA thành công.")
except Exception as e:
    print(f"🛑 Lỗi khi tải LoRA: {e}"); exit()

trigger_words = "CONAN EDOGAWA, BROWN HAIR, BLUE EYES, GLASSES"

# Mô tả vật lý giờ đây chỉ là phần phụ để củng cố
physical_description = "a young Japanese boy detective"

# === PROMPT MỚI, NGẮN GỌN VÀ MẠNH MẼ ===
prompt = f"""
masterpiece, best quality, cel animation, anime style, clean lineart,
(full body shot of ({trigger_words}:1.4):1.2),
{physical_description},
wearing a blue schoolboy suit jacket and a red bow tie,
standing confidently in a detective's office
"""

# Negative prompt
negative_prompt = "lowres, bad anatomy, text, error, blurry, 3d, realistic, photorealistic, signature, watermark, ugly, deformed, extra limbs, missing fingers, bad hands, white shirt"

print(f"\n🎨 Bắt đầu tạo ảnh...")
print(f"   Prompt: '{prompt}'")

try:
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=35,
        guidance_scale=8,
        width=768,
        height=1024,
    ).images[0]
    
    # --- 5. Lưu ảnh ---
    output_filename = "conan_lora_test_final.png" # Lưu file mới để so sánh
    image.save(output_filename)

    print(f"\n🎉 THÀNH CÔNG! Đã lưu ảnh vào file '{output_filename}'")
    print(f"   Đường dẫn: {os.path.abspath(output_filename)}")

except Exception as e:
    print(f"\n🛑 Lỗi trong quá trình tạo ảnh: {e}")

finally:
    # Quan trọng: Gỡ LoRA để giải phóng bộ nhớ
    print("\n🧹 Dọn dẹp LoRA...")
    try:
        pipe.unload_lora_weights()
    except Exception as e:
        print(f"   -> Lỗi khi dọn dẹp: {e}")