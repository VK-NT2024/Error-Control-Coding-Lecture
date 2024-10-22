import numpy as np
import matplotlib.pyplot as plt
from ecc import ECC
from TurboCodes.bicm import BICM
from Mapping.modulation import Modulation
from BlockCodes.spc import SPC

M = Modulation(4, 'antigray', 'QAM')
S = SPC(4)

BICM = BICM(S, M)
BICM.set_interleaver('random', 5)

n_dB = 10 * np.log10(BICM.spectral_efficiency)
u_len = 4
depth = 1000
SNR_range = np.arange(-10, 10, 1)
error = np.zeros((depth, len(SNR_range)))

for loop in range(depth):
    for i, SNR_dB in enumerate(SNR_range):
        u = np.random.randint(2, size=u_len)
        x = BICM.transmit(u)

        # channel
        SNR_lin = 10 ** (SNR_dB / 10)
        mu, sigma = 0, np.sqrt(1 / SNR_lin)
        w = np.random.normal(mu, sigma, size=len(x))

        y = x + w
        u_ = BICM.receive(y, SNR_dB, iterations=5)

        error[loop, i] = np.count_nonzero(u_ - u)
    error_rate = np.mean(error, axis=0) / u_len

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.semilogy(SNR_range, error_rate)
ax1.set_xlabel('$E_s/N_0(dB)$')
ax1.set_ylabel('Symbol Error Rate ($P_s$)')
ax1.set_title('Probability of Symbol Error over AWGN channel')
ax1.set_xlim(SNR_range[0] - 1, SNR_range[-1] + 1)
ax1.grid(True)
ax2.semilogy(SNR_range - n_dB, error_rate)
ax2.set_xlabel('$E_b/N_0(dB)$')
ax2.set_ylabel('Bit Error Rate ($P_b$)')
ax2.set_title('Probability of Bit Error over AWGN channel')
ax2.set_xlim(SNR_range[0] - 1 - n_dB, SNR_range[-1] + 1 - n_dB)
ax2.grid(True)
plt.show()