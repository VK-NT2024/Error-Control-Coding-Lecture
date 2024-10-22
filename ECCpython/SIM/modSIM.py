import numpy as np
import matplotlib.pyplot as plt

from Modulation import modulation
from SPC import SPC
from Interleavers import interleavers

def SIM(depth = 1000, iterations = 5, SNR = [-5, 5], SNR_step = 0.5):
    M = modulation(2, 'antigray', 'QAM')
    S = SPC(3)
    T = interleavers()

    spectral_efficiency = M.m*S.k/S.n
    SNR_range = np.arange(start=SNR[0], stop=SNR[1], step=SNR_step)
    error = np.zeros((depth, len(SNR_range)))

    for loop in range(depth):
        for s, SNR in enumerate(SNR_range):
            # raw data
            u = M.generateData(S.k)
            # encoding data
            c = S.encode(u)
            # interleaving
            T.pInterleaver(flag=0, size=S.n)
            c0 = T.Interleaver(0, c)
            # modulation
            x, _ = M.modulate(c0)
            # channel
            w = M.generateNoise(SNR, len(x))
            y = x + w
            # soft demodulation + decoding
            Le2 = np.zeros(S.n)
            for _ in range(iterations):
                La1 = T.Interleaver(0, Le2)
                Lc1 = M.soft_demodulation(y, SNR, La1)
                y2 = T.deInterleaver(0, La1-Lc1)
                Lc2, Le2 = S.SPCsoftDecode(y2, 2*SNR)

            c_hat = Lc2<=0
            u_hat = c_hat.flatten()[:S.k]

            error[loop, s] = np.count_nonzero(u_hat - u)
    
    error_rate = np.mean(error, axis = 0)/S.k

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.semilogy(SNR_range, error_rate)
    ax1.set_xlabel('$E_s/N_0(dB)$'); ax1.set_ylabel('Symbol Error Rate ($P_s$)')
    ax1.set_title('Probability of Symbol Error over AWGN channel')
    ax1.set_xlim(SNR_range[0]-1, SNR_range[-1]+1); ax1.grid(True);
    nn = 10*np.log10(spectral_efficiency)
    ax2.semilogy(SNR_range - nn, error_rate)
    ax2.set_xlabel('$E_b/N_0(dB)$'); ax2.set_ylabel('Bit Error Rate ($P_b$)')
    ax2.set_title('Probability of Bit Error over AWGN channel')
    ax2.set_xlim(SNR_range[0]-1-nn, SNR_range[-1]+1-nn); ax2.grid(True);
    plt.show()

########################################################################################
SIM(depth = 10000, iterations = 5, SNR = [0, 10], SNR_step = 0.5)