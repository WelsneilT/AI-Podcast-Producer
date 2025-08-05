# my_story_project/celery.py (FILE MỚI)

import os
from celery import Celery

# Đặt biến môi trường mặc định của Django để Celery biết tìm settings ở đâu
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'my_story_project.settings')

# Tạo một instance của Celery app, đặt tên là 'my_story_project'
app = Celery('my_story_project')

# Tải cấu hình từ file settings.py của Django.
# namespace='CELERY' nghĩa là Celery sẽ tìm các biến bắt đầu bằng CELERY_ trong settings
app.config_from_object('django.conf:settings', namespace='CELERY')

# Tự động tìm tất cả các file tasks.py trong các app Django của bạn (ví dụ: creator/tasks.py)
app.autodiscover_tasks()