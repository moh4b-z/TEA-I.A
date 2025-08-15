import cv2
import numpy as np
from src.image_analysis import calculate_brightness, calculate_saturation

def test_calculate_brightness():
    # Criando uma imagem branca (brilho máximo)
    white_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    brightness = calculate_brightness(white_img)
    assert brightness > 250  # muito alto

def test_calculate_saturation():
    # Criando uma imagem totalmente cinza (saturação mínima)
    gray_img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    saturation = calculate_saturation(gray_img)
    assert saturation == 0
