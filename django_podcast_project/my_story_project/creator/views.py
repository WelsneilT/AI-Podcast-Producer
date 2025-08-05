# creator/views.py (PHIÊN BẢN SỬA LỖI CUỐI CÙNG - LOGIC ĐÚNG)

from django.shortcuts import render
from django.http import JsonResponse
from celery import chain
from celery.result import AsyncResult
from django.templatetags.static import static
import os

from .tasks import create_full_story_task, generate_all_media_task
from .characters import CARTOON_CHARACTERS

def create_story_view(request):
    """
    View chính của ứng dụng.
    - POST: Nhận yêu cầu, khởi động một Celery chain ĐƠN GIẢN.
    - GET: Hiển thị trang.
    """
    if request.method == 'POST':
        try:
            plotline = request.POST.get('plotline')
            char1_name = request.POST.get('character1')
            char2_name = request.POST.get('character2')

            selected_chars_data = [
                char for char in CARTOON_CHARACTERS if char['name'] in [char1_name, char2_name]
            ]

            if len(selected_chars_data) < 2:
                return JsonResponse({'error': 'Please select two valid characters.'}, status=400)

            # === XÂY DỰNG DÂY CHUYỀN SẢN XUẤT ĐƠN GIẢN VÀ ĐÚNG ĐẮN ===
            # Bước 1: Tạo nội dung văn bản.
            # Bước 2: Dùng kết quả của bước 1 để tạo media.
            # Celery tự động xử lý việc chuyển kết quả.
            # Việc cập nhật trạng thái đã được xử lý BÊN TRONG `generate_all_media_task`
            # nhờ có `bind=True`.
            production_chain = chain(
                create_full_story_task.s(plotline, selected_chars_data),
                generate_all_media_task.s()
            )

            # Thực thi chain và lấy ID của nó
            result = production_chain.apply_async()

            # Trả về task_id của chain cho frontend để polling
            return JsonResponse({'task_id': result.id})

        except Exception as e:
            print(f"🛑 LỖI NGHIÊM TRỌNG TRONG create_story_view (POST): {e}")
            return JsonResponse({'error': f'An internal server error occurred: {e}'}, status=500)

    # --- Phần xử lý request GET (giữ nguyên không đổi) ---
    characters_with_static_urls = []
    for char in CARTOON_CHARACTERS:
        new_char = char.copy()
        try:
            # Đường dẫn tương đối bên trong thư mục static
            relative_path = os.path.join('character_portraits', new_char['image_url']).replace("\\", "/")
            new_char['static_image_url'] = static(relative_path)
        except (ValueError, TypeError):
            # Fallback nếu có lỗi
            new_char['static_image_url'] = ''
        characters_with_static_urls.append(new_char)

    context = {
        # Đổi tên file template về đúng như bạn đang dùng là index.html hoặc create_page.html
        'characters_preset': characters_with_static_urls
    }

    # Đảm bảo bạn render đúng tên template bạn đang dùng
    # Ví dụ: 'creator/index.html' hoặc 'creator/create_page.html'
    return render(request, 'creator/create_page.html', context)


def task_status_view(request, task_id):
    """
    View API để kiểm tra trạng thái của chain.
    Logic này đã đúng và sẽ hoạt động với task mới.
    """
    task_result = AsyncResult(task_id)

    status_message = 'Brewing up your story...' # Tin nhắn mặc định
    if task_result.info and isinstance(task_result.info, dict):
        status_message = task_result.info.get('status', status_message)

    response_data = {
        'state': task_result.state,
        'details': {'status': status_message},
    }

    if task_result.state == 'SUCCESS':
        # Lấy kết quả cuối cùng từ chain
        response_data['result'] = task_result.get()
    elif task_result.state == 'FAILURE':
        # Nếu thất bại, hiển thị lỗi
        response_data['details']['status'] = str(task_result.info)

    return JsonResponse(response_data)