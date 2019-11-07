from django.urls import path

from image import views
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    path('image/', views.validate_image),
]

urlpatterns = format_suffix_patterns(urlpatterns)