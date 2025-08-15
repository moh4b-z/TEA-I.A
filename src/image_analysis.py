import cv2
import numpy as np
from utils import calcular_brilho, calcular_saturacao


# Configurações
LIMIAR_BRILHO = 180
LIMIAR_SATURACAO = 150
AREA_MINIMA = 500

CORES_ALERTA = {
    "Brilho Alto": (0, 255, 255),
    "Saturacao Alta": (255, 0, 255),
    "Vermelho Puro": (0, 0, 255),
    "Amarelo Puro": (0, 255, 255),
    "Laranja": (0, 165, 255),
}


class ImageAnalyzer:
    def __init__(self):
        pass

    def _gerar_mascaras(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        s_channel = hsv[:, :, 1]

        masks = {
            "Brilho Alto": cv2.inRange(gray, LIMIAR_BRILHO, 255),
            "Saturacao Alta": cv2.inRange(s_channel, LIMIAR_SATURACAO, 255),
            "Vermelho Puro": cv2.inRange(hsv, (0, 120, 120), (10, 255, 255)),
            "Amarelo Puro": cv2.inRange(hsv, (20, 120, 120), (30, 255, 255)),
            "Laranja": cv2.inRange(hsv, (10, 120, 120), (20, 255, 255)),
        }
        return masks

    def _desenhar_bounding_boxes(self, frame: np.ndarray, mask: np.ndarray, cor: tuple, label: str):
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contornos:
            if cv2.contourArea(cnt) < AREA_MINIMA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    def process_frame(self, frame: np.ndarray) -> dict:
        brilho = calcular_brilho(frame)
        saturacao = calcular_saturacao(frame)
        dados_para_ia = np.array([brilho, saturacao])

        masks = self._gerar_mascaras(frame)
        for label, mask in masks.items():
            self._desenhar_bounding_boxes(frame, mask, CORES_ALERTA[label], label)

        return {"dados_para_ia": dados_para_ia, "frame_processado": frame}
