import cv2
import numpy as np

def calcular_brilho(frame):
    """
    Calcula o brilho médio da imagem (0 a 255)
    Convertendo para escala de cinza e tirando a média
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def calcular_saturacao(frame):
    """
    Calcula a saturação média da imagem (0 a 255)
    Convertendo para HSV e pegando o canal S
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    return np.mean(s_channel)

def main():
    # Abre a webcam padrão (0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar imagem da câmera.")
            break

        # Calcula brilho e saturação
        brilho_medio = calcular_brilho(frame)
        saturacao_media = calcular_saturacao(frame)

        # Prepara dados para IA (exemplo de vetor de entrada)
        dados_para_ia = np.array([brilho_medio, saturacao_media])

        # Exibe valores no terminal
        print(f"Brilho: {brilho_medio:.2f} | Saturação: {saturacao_media:.2f}")

        # Mostra vídeo com texto na tela
        texto = f"B:{brilho_medio:.1f} | S:{saturacao_media:.1f}"
        cv2.putText(frame, texto, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Analise de Ambiente - TEA", frame)

        # Sai se apertar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
