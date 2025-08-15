import cv2
import numpy as np
from camera_stream import CameraStream
from image_analysis import ImageAnalyzer
from ai_model import AIModel


def main():
    camera = CameraStream(camera_id=0)
    analyzer = ImageAnalyzer()
    model = AIModel()

    print("Pressione 'q' para sair.")

    while True:
        frame = camera.get_frame()
        resultado = analyzer.process_frame(frame)

        dados_para_ia = resultado["dados_para_ia"]
        frame_processado = resultado["frame_processado"]

        # Envia dados para a IA (placeholder)
        predicao = model.predict(dados_para_ia)
        print(f"Predição IA: {predicao}")

        cv2.imshow("Analise de Ambiente - TEA", frame_processado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()


if __name__ == "__main__":
    main()
