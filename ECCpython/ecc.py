import numpy as np
import matplotlib.pyplot as plt


class ECC:
    def __init__(self, code):
        self.code = code
        self.info_length = self.code.k * 100

    def simulate(self, trials=100, SNR_min=0, SNR_max=10, SNR_step=1, algorithm="default"):
        pass
