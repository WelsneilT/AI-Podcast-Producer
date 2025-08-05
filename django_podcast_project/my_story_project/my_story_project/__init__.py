# my_story_project/__init__.py (TOÀN BỘ FILE)

# Đảm bảo app Celery được import khi Django khởi động
from .celery import app as celery_app

# Dòng này là một quy ước của Python để khai báo những gì sẽ được export
# khi một module khác import package này.
__all__ = ('celery_app',)