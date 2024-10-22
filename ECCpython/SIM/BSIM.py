import numpy as np
from Hamming import Hamming
from SPC import SPC

def BSIM(k, depth=1000, SNR=[-5, 10], SNR_step=1):
    import matplotlib.pyplot as plt
    BC = SPC(k)
    
    Rc = BC.k/BC.n
    nn = 10*np.log10(Rc)

    SNR_range = np.arange(start=SNR[0], stop=SNR[1], step=SNR_step)
    error = np.zeros((depth, len(SNR_range)))

    ########################################################################################

    for trial in range(depth):
        for i, SNR_dB in enumerate(SNR_range):
            #----------transmitter-------------
            u = np.random.randint(2, size=BC.k)
            c = BC.encode(u)
            x = 1 - 2*c
            #----------channel-----------------
            SNR = 2*10**(SNR_dB/10); Lch = 2*SNR
            mu, sigma = 0, np.sqrt(1/SNR)
            w = np.random.normal(mu, sigma, size=BC.n)
            y = x + w
            #----------receiver----------------
            Lc, Le = BC.SPCsoftDecode(y, Lch)
            u_hat = Lc[0:BC.k]<=0
            #----------error-------------------
            error[trial, i] = np.count_nonzero(u_hat - u)

    error_rate = np.mean(error, axis = 0)/BC.k

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.semilogy(SNR_range, error_rate)
    ax1.set_xlabel('$E_s/N_0(dB)$'); ax1.set_ylabel('Symbol Error Rate ($P_s$)')
    ax1.set_title('Probability of Symbol Error over AWGN channel')
    ax1.set_xlim(SNR_range[0]-1, SNR_range[-1]+1); ax1.grid(True);
    ax2.semilogy(SNR_range - nn, error_rate)
    ax2.set_xlabel('$E_b/N_0(dB)$'); ax2.set_ylabel('Bit Error Rate ($P_b$)')
    ax2.set_title('Probability of Bit Error over AWGN channel')
    ax2.set_xlim(SNR_range[0]-1-nn, SNR_range[-1]+1-nn); ax2.grid(True);
    plt.show()

######################################################################################
BSIM(3, 1, [-5,10], 0.5)