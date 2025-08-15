import cv2
import numpy as np


def calcular_brilho(frame: np.ndarray) -> float:
    """Calcula o brilho médio (0-255)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def calcular_saturacao(frame: np.ndarray) -> float:
    """Calcula a saturação média (0-255)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    return float(np.mean(s_channel))
