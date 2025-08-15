import cv2


class CameraStream:
    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Erro: Não foi possível acessar a câmera.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Erro ao capturar imagem da câmera.")
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
