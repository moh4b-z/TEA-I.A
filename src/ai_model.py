import numpy as np


class AIModel:
    def __init__(self):
        # Aqui futuramente carregaria um modelo real
        self.modelo = None

    def predict(self, dados: np.ndarray):
        """
        Simulação de predição.
        Substituir pelo modelo real (ex: TensorFlow, PyTorch).
        """
        brilho, saturacao = dados
        if brilho > 180 or saturacao > 150:
            return "Ambiente Potencialmente Desconfortável"
        return "Ambiente Normal"
