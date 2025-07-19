# build_embedding_file.py
import torch
import numpy as np
import os
import glob

def build_speaker_embedding_file():
    """
    Đọc tất cả các file x-vector .npy và ghép chúng thành một tensor duy nhất.
    """
    xvector_dir = "xvectors"
    output_file = "models/speaker_embeddings.pt"
    
    os.makedirs("models", exist_ok=True)
    
    if os.path.exists(output_file):
        print(f"✅ File '{output_file}' đã tồn tại. Không cần tạo lại.")
        return

    npy_files = sorted(glob.glob(os.path.join(xvector_dir, "*.npy")))
    
    if not npy_files:
        print(f"🛑 LỖI: Không tìm thấy file .npy nào trong thư mục '{xvector_dir}'.")
        print("   Vui lòng tải và giải nén 'spkrec-xvect.zip' vào đó.")
        return
        
    print(f"🔍 Tìm thấy {len(npy_files)} file x-vector. Đang tổng hợp...")
    
    all_embeddings = []
    for f in npy_files:
        embedding = np.load(f)
        all_embeddings.append(torch.from_numpy(embedding))
        
    final_tensor = torch.stack(all_embeddings)
    
    torch.save(final_tensor, output_file)
    print("\n🎉 THÀNH CÔNG! Đã tạo và lưu file speaker embeddings tại:")
    print(f"   -> {os.path.abspath(output_file)}")
    print(f"   Tensor có kích thước: {final_tensor.shape}")

if __name__ == "__main__":
    build_speaker_embedding_file()