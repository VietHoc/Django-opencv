import base64
import io
from imageio import imread
from PIL import Image
import numpy as np
from django.shortcuts import render

# Create your views here.
from rest_framework import status, serializers
from rest_framework.decorators import api_view
from rest_framework.response import Response
import cv2 as cv
import argparse

from image.models import Image


class ImageListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = '__all__'


@api_view(['POST'])
def validate_image(request):
    if request.method == 'POST':
        # serializer = ImageListSerializer(data=request.data)
        print(request.data.get('image'))
        img = imread(io.BytesIO(base64.b64decode(request.data.get('image'))))
        return Response(detect_face_and_eyes(img), status=status.HTTP_200_OK)


def detect_face_and_eyes(frame):
    face_cascade = cv.CascadeClassifier('image/data/haarcascades/haarcascade_frontalface_alt.xml')
    eyes_cascade = cv.CascadeClassifier('image/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    print('faces:', len(faces))

    eyes = 0;
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y + h, x:x + w]
        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        print('eyes:', len(eyes))
        eyes = len(eyes)
        message = 'Pass'
        if eyes < 2:
            message = 'No eyes in the photo'
    return {
        'faces': len(faces),
        'eyes': eyes,
        'message': message
    }

