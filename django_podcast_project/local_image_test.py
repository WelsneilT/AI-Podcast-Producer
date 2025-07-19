# local_image_test.py

import torch
from diffusers import DiffusionPipeline
import os

print("--- BẮT ĐẦU BÀI KIỂM TRA VỚI DREAMSHAPER 8 ---")

# --- 1. Kiểm tra GPU ---
if torch.cuda.is_available():
    print(f"✅ GPU được nhận diện: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("🛑 Không tìm thấy GPU."); exit()

# --- 2. Tải mô hình Dreamshaper 8 ---
model_id = "Lykon/dreamshaper-8"
print(f"\n⏳ Đang tải mô hình '{model_id}'...")
print("   (Lần đầu sẽ mất thời gian, các lần sau sẽ dùng cache.)")

try:
    pipe = DiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe = pipe.to(device)
    print("✅ Tải mô hình thành công.")
except Exception as e:
    print(f"🛑 Lỗi khi tải mô hình: {e}"); exit()


# --- 3. Định nghĩa Prompt và Negative Prompt ---
prompt = "PMasterpiece, high-detail 32-bit pixel art sprite of a small, sleek fighter spaceship, top-down view. The dark metallic fuselage has visible panel lines, rivets, and subtle weathering. Features a glowing blue cockpit canopy and two swept-back wings with mounted laser cannons. A pair of high-output engines emit a bright cyan thruster glow. Intricate shading and specular highlights give it a realistic metallic texture. Centered, dark background."
negative_prompt = "complex, detailed, large, ship, realistic, 3D, blurry, waves, shadows, intricate, 16-bit, 32-bit."

print(f"\n🎨 Bắt đầu tạo ảnh...")
print(f"   Prompt: '{prompt}'")

try:
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30
    ).images[0]
    
    # --- 4. Lưu ảnh ---
    output_filename = "dreamshaper_test.png"
    image.save(output_filename)

    print(f"\n🎉 THÀNH CÔNG! Đã lưu ảnh vào file '{output_filename}'")
    print(f"   Đường dẫn: {os.path.abspath(output_filename)}")

except Exception as e:
    print(f"\n🛑 Lỗi trong quá trình tạo ảnh: {e}")