from rest_framework import serializers
from .services import ValidateImageService


class ValidateImageSerializer(serializers.Serializer):
    """Validating images serializer."""
    images_list = serializers.ListField(child=serializers.CharField())

