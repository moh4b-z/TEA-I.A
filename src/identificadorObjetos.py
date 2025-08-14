import cv2
import numpy as np

# Limiares ajustáveis
LIMIAR_BRILHO = 180       # Quanto maior, mais exigente
LIMIAR_SATURACAO = 150

def desenhar_bounding_boxes(frame, mask, cor, label):
    """
    Desenha bounding boxes em regiões definidas pela máscara.
    """
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        # Ignora áreas muito pequenas
        if cv2.contourArea(cnt) < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

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

        # Conversões
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]

        # Máscaras
        mask_brilho = cv2.inRange(gray, LIMIAR_BRILHO, 255)
        mask_saturacao = cv2.inRange(s_channel, LIMIAR_SATURACAO, 255)

        # Desenha bounding boxes
        desenhar_bounding_boxes(frame, mask_brilho, (0, 255, 255), "Brilho Alto")
        desenhar_bounding_boxes(frame, mask_saturacao, (255, 0, 255), "Saturacao Alta")

        # Mostra resultado
        cv2.imshow("Deteccao de Regioes - Brilho/Saturacao", frame)

        # Sai com 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
