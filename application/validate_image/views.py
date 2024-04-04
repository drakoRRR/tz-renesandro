from rest_framework import views, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from . import serializers


class ValidateImage(views.APIView):
    """Validating images and return the good ones."""

    def post(self, request):
        serializer = serializers.ValidateImageSerializer(data=request.data)
