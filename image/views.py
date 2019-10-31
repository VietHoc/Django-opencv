import base64
import io

from imageio import imread
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


@api_view(['POST'])
def validate_image(request):
    if request.method == 'POST':
        image_bade_64 = request.data.get('image')
        temp_img = stringToImage(image_bade_64)
        img = imread(io.BytesIO(base64.b64decode(request.data.get('image'))))

        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.show()
        return Response([
            {'image_removed_background': remove_background_image(toRGB(temp_img))},
            detect_face_and_eyes(img),
            {'image_contrast': check_contrast_by_histogram(hist)},
        ],
            status=status.HTTP_200_OK)


# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv.cvtColor(np.array(image), cv.COLOR_BGR2RGB)


def detect_face_and_eyes(frame):
    face_cascade = cv.CascadeClassifier('image/data/haarcascades/haarcascade_frontalface_alt.xml')
    eyes_cascade = cv.CascadeClassifier('image/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    smile_cascade = cv.CascadeClassifier('image/data/haarcascades/haarcascade_smile.xml')

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    print('faces:', len(faces))

    eyes_of_1_face = 0
    message = 'Pass'
    header_message = 'Normal'
    is_smile: bool
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y + h, x:x + w]

        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        print('eyes:', len(eyes))
        eyes_of_1_face = len(eyes)

        if eyes_of_1_face < 2:
            message = 'No eyes in the photo'
        else:
            if (abs(eyes[0][1] - eyes[1][1]) > 5) | (abs(eyes[0][3] - eyes[1][3]) > 8):
                header_message = "Header is no frontend"

        # -- In each face, detect smile
        is_smile = smile_detection(smile_cascade, faceROI)
    return {
        'faces': len(faces),
        'eyes': eyes_of_1_face,
        'message': message,
        'header_message': header_message,
        'is_smile': is_smile
    }


def check_contrast_by_histogram(hist):
    if (sum(hist[:5]) < 20) & (sum(hist[-5:]) < 20):
        return 'Normal'
    else:
        if (sum(hist[5]) > 1000) & (sum(hist[-5:]) < 20):
            return 'Low contrast'
        else:
            if (sum(hist[:5]) < 20) & (sum(hist[-5:]) > 1000):
                return 'High contrast'

    return 'Normal'


def smile_detection(smile_cascade, faceROI):
    smiles = smile_cascade.detectMultiScale(faceROI, 1.8, 20)
    return len(smiles) > 0


def remove_background_image(img):
    # == Parameters
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 100
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # -- Edge detection
    edges = cv.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv.dilate(edges, None)
    edges = cv.erode(edges, None)

    # -- Find contours in edges, sort by area
    contour_info = []
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv.isContourConvex(c),
            cv.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    for c in contour_info:
        cv.fillConvexPoly(mask, c[0], (255))

    # -- Smooth mask, then blur it
    mask = cv.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background
    mask_stack = mask_stack.astype('float32') / 255.0
    img = img.astype('float32') / 255.0

    print(type(img), type(mask_stack))

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
    masked = (masked * 255).astype('uint8')

    print(type(masked))
    pil_img = Image.fromarray(masked)
    buff = io.BytesIO()
    pil_img.save(buff, format="png")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return new_image_string
