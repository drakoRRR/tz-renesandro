from rest_framework import serializers


class ValidateImageSerializer(serializers.Serializer):
    """Validating images serializer."""
    images = serializers.ListField(child=serializers.CharField())

