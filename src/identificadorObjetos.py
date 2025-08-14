import cv2
import numpy as np

# Limiares para destacar (pode ajustar conforme o teste)
LIMIAR_BRILHO = 180   # 0-255
LIMIAR_SATURACAO = 150  # 0-255

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar imagem.")
            break

        # Converte para escalas necessárias
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Extrai canal de saturação
        s_channel = hsv[:, :, 1]

        # Máscaras para locais com alto brilho/saturação
        mask_brilho = gray > LIMIAR_BRILHO
        mask_saturacao = s_channel > LIMIAR_SATURACAO

        # Cria uma cópia para sobreposição visual
        overlay = frame.copy()

        # Destaca brilho alto em amarelo
        overlay[mask_brilho] = [0, 255, 255]  # BGR → Amarelo

        # Destaca saturação alta em magenta
        overlay[mask_saturacao] = [255, 0, 255]  # BGR → Magenta

        # Combina original + overlay com transparência
        resultado = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Exibe
        cv2.imshow("Mapa de Brilho/Saturacao", resultado)

        # Sai com 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
