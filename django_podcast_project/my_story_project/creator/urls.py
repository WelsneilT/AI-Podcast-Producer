# creator/urls.py (PHIÊN BẢN CUỐI CÙNG, AN TOÀN HƠN)

from django.urls import path
from . import views

app_name = 'creator'

urlpatterns = [
    # Path cho trang chính
    path('', views.create_story_view, name='create_story_view'),
    
    # Path cho API kiểm tra trạng thái
    path('api/task-status/<str:task_id>/', views.task_status_view, name='task_status_view'),
]