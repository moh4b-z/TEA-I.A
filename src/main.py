import cv2
import numpy as np

# =====================
# Configurações (Open-Closed Principle)
# =====================
LIMIAR_BRILHO = 180
LIMIAR_SATURACAO = 150
AREA_MINIMA = 500

CORES_ALERTA = {
    "Brilho Alto": (0, 255, 255),     # Amarelo
    "Saturacao Alta": (255, 0, 255),  # Magenta
    "Vermelho Puro": (0, 0, 255),     # Vermelho
    "Amarelo Puro": (0, 255, 255),    # Amarelo
    "Laranja": (0, 165, 255)          # Laranja
}


# =====================
# Single Responsibility: Funções de Métricas
# =====================
def calcular_brilho(frame: np.ndarray) -> float:
    """Calcula o brilho médio da imagem (0-255)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def calcular_saturacao(frame: np.ndarray) -> float:
    """Calcula a saturação média da imagem (0-255)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    return float(np.mean(s_channel))


# =====================
# Responsabilidade: Geração de Máscaras
# =====================
def gerar_mascaras(frame: np.ndarray) -> dict:
    """Cria máscaras para brilho, saturação e cores puras."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    s_channel = hsv[:, :, 1]
    h_channel = hsv[:, :, 0]

    # Máscaras principais
    mask_brilho = cv2.inRange(gray, LIMIAR_BRILHO, 255)
    mask_saturacao = cv2.inRange(s_channel, LIMIAR_SATURACAO, 255)

    # Máscaras de cores puras (intervalos HSV aproximados)
    mask_vermelho = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
    mask_amarelo = cv2.inRange(hsv, (20, 120, 120), (30, 255, 255))
    mask_laranja = cv2.inRange(hsv, (10, 120, 120), (20, 255, 255))

    return {
        "Brilho Alto": mask_brilho,
        "Saturacao Alta": mask_saturacao,
        "Vermelho Puro": mask_vermelho,
        "Amarelo Puro": mask_amarelo,
        "Laranja": mask_laranja
    }


# =====================
# Responsabilidade: Visualização
# =====================
def desenhar_bounding_boxes(frame: np.ndarray, mask: np.ndarray, cor: tuple, label: str):
    """Desenha bounding boxes em regiões definidas pela máscara."""
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        if cv2.contourArea(cnt) < AREA_MINIMA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)


# =====================
# Dependency Inversion: Classe principal que orquestra
# =====================
class AnalisadorAmbiente:
    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Erro: Não foi possível acessar a câmera.")

    def capturar_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Erro ao capturar imagem da câmera.")
        return frame

    def processar_frame(self, frame: np.ndarray) -> dict:
        """Processa o frame, calcula métricas e desenha bounding boxes."""
        # Calcula métricas numéricas
        brilho_medio = calcular_brilho(frame)
        saturacao_media = calcular_saturacao(frame)

        # Cria vetor pronto para IA
        dados_para_ia = np.array([brilho_medio, saturacao_media])

        # Gera máscaras e desenha bounding boxes
        mascaras = gerar_mascaras(frame)
        for label, mask in mascaras.items():
            desenhar_bounding_boxes(frame, mask, CORES_ALERTA[label], label)

        return {
            "dados_para_ia": dados_para_ia,
            "frame_processado": frame
        }

    def liberar(self):
        self.cap.release()
        cv2.destroyAllWindows()


# =====================
# Execução
# =====================
def main():
    analisador = AnalisadorAmbiente()

    print("Pressione 'q' para sair.")

    while True:
        frame = analisador.capturar_frame()
        resultado = analisador.processar_frame(frame)

        # Aqui já temos dados prontos para IA
        dados_para_ia = resultado["dados_para_ia"]

        # Mostra frame processado com bounding boxes
        cv2.imshow("Analise de Ambiente - TEA", resultado["frame_processado"])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    analisador.liberar()


if __name__ == "__main__":
    main()
