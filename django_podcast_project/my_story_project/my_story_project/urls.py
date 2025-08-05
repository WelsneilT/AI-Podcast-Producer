# my_story_project/urls.py (PHIÊN BẢN ĐÚNG)

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

# Chỉ định các URL ở cấp cao nhất của dự án.
# Dòng `include('creator.urls')` sẽ chuyển giao tất cả các request
# không phải '/admin/' cho file urls.py của app creator.
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('creator.urls')),
]

# Phần này chỉ để phục vụ file media và static trong môi trường development.
# Nó không gây ra lỗi đệ quy.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)