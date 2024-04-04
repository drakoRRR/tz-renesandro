from django.urls import path

from . import views

app_name = 'validate_images'

urlpatterns = [
    path('api/validate_images', views.ValidateImage.as_view(), name='validate_images'),
]
