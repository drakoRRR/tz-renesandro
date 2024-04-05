from rest_framework import views, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from . import serializers
from .services import ValidateImageService


class ValidateImage(views.APIView):
    """Validating images and return the good ones."""

    def post(self, request):
        serializer = serializers.ValidateImageSerializer(data=request.data)
        if serializer.is_valid():
            images_list = serializer.validated_data.get('images_list')
            valid_images = ValidateImageService(images_list).validate_images()
            return Response(data=dict(images_list=valid_images), status=status.HTTP_200_OK)
        return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)

