import cv2
import numpy as np
import pytesseract
import keras_ocr
import re

def intersection(box1, box2):
    x_min1, y_min1 = box1[0][0], box1[0][1]
    x_max1, y_max1 = box1[2][0], box1[2][1]
    x_min2, y_min2 = box2[0][0], box2[0][1]
    x_max2, y_max2 = box2[2][0], box2[2][1]
    if x_max1 < x_min2 or x_min1 > x_max2:
        return False
    if y_max1 < y_min2 or y_min1 > y_max2:
        return False
    return True

def group_box(boxes, function):
    groupes = [0] * len(boxes)
    group = 1
    for i in range(len(boxes)):
        if groupes[i] == 0:
            groupes[i] = group
        for j in range(i, len(boxes)):
            if function(boxes[i], boxes[j]):
                groupes[j] = groupes[i]
        group += 1
    return groupes

def create_new_boxes(boxes, groupes):
    new_boxes = []
    if (groupes == []):
        return []
    for j in range(np.max(groupes) + 1):
        x_min = x_max = y_min = y_max = 0
        first = True
        for i in range(len(boxes)):
            if (groupes[i] == j):
                if (first):
                    x_min = np.min(boxes[i, :, 0])
                    x_max = np.max(boxes[i, :, 0])
                    y_min = np.min(boxes[i, :, 1])
                    y_max = np.max(boxes[i, :, 1])
                    first = False
                else:
                    x_min = min(np.min(boxes[i, :, 0]), x_min)
                    x_max = max(np.max(boxes[i, :, 0]), x_max)
                    y_min = min(np.min(boxes[i, :, 1]), y_min)
                    y_max = max(np.max(boxes[i, :, 1]), y_max)
        if (not first):
            small_box = []
            small_box.append([x_min, y_min])
            small_box.append([x_max, y_min])
            small_box.append([x_max, y_max])
            small_box.append([x_min, y_max])
            new_boxes.append(small_box)
    return np.array(new_boxes)

def isOnSameLine(boxOne, boxTwo):
    boxOneStartY = boxOne[0,1]
    boxOneEndY = boxOne[2,1]
    boxTwoStartY = boxTwo[0,1]
    boxTwoEndY = boxTwo[2,1]
    if((boxTwoStartY <= boxOneEndY and boxTwoStartY >= boxOneStartY)
    or(boxTwoEndY <= boxOneEndY and boxTwoEndY >= boxOneStartY)
    or(boxTwoEndY >= boxOneEndY and boxTwoStartY <= boxOneStartY)):
        return True
    else:
        return False

def detecter_piece_et_taille(texte):
    texte = texte.strip("\n\t ")
    lignes = texte.split('\n')

    piece_maison = None
    taille = None

    for ligne in lignes:
        nombre_m = re.search(r'\d+(\.\d+)?\s*x\s*\d+(\.\d+)?\s*m$|^\d+(\.\d+)?\s*m+$', ligne)
        if nombre_m:
            taille = re.sub(r'\s*m+$', '', nombre_m.group())
            if taille.count('x') >= 1:
                dimensions = taille.split('x')
                taille = 1
                for dimension in dimensions:
                    taille *= float(dimension.strip())
                
        else:
            piece_maison = ligne.strip()

    return piece_maison, taille


def OCR(path):
    detector = keras_ocr.detection.Detector()
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    image = keras_ocr.tools.read(path)
    boxes = detector.detect(images=[image], text_threshold=0.2)[0]
    groupes = group_box(boxes, intersection)
    new_boxes = create_new_boxes(boxes, groupes)
    image_OCR = []

    cv2image = cv2.imread(path)
    for i, box in enumerate(new_boxes):
        x_min, y_min = map(int, box[0])
        x_max, y_max = map(int, box[2])
        
        cropped_box = cv2image[y_min:y_max, x_min:x_max]
        current_height = cropped_box.shape[0]
        scale_percent = (160 / current_height) * 100
        width = int(cropped_box.shape[1] * scale_percent / 100)
        height = int(cropped_box.shape[0] * scale_percent / 100)
        resized_box = cv2.resize(cropped_box, (width, height), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(resized_box, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
        img = clahe.apply(img)
        img = cv2.fastNlMeansDenoising(img, h=20)
        kernel = np.array([[-1,-1,-1],
                        [-1,9,-1],
                        [-1,-1,-1]])

        img = cv2.filter2D(img, -1, kernel)
        
        text = pytesseract.image_to_string(img)
        text = re.sub('\xad(\x0c)*',  '', text)
        text = re.sub('[\x0c]', ' ', text)
        char_to_remove = "?!@!#$%^&*()[]'\\/><~`:;_"
        for char in char_to_remove:
            text = text.replace(char, '')
        piece, taille = detecter_piece_et_taille(text)
        if piece is not None:
            image_OCR.append([piece, taille])
    
    return image_OCR
