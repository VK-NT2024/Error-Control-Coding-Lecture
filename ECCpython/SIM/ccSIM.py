import numpy as np
from ConvCode import ConvCode


def ccSIM(generator_poly, rsc_poly, decode_type='BCJR', depth=10, SNR=[-10, 10], SNR_step=1, terminate=True,
          info_length=1000):
    import matplotlib.pyplot as plt
    CC = ConvCode(generator_poly, rsc_poly)

    Dx = {'BCJR': lambda y, Lch, terminate: CC.BCJR_decode(y, Lch, terminate)[0] <= 0,
          'viterbi': lambda y, Lch, terminate: CC.viterbi_decode(y, terminate)[0],
          'MLM': lambda y, Lch, terminate: CC.MaxLogMap_decode(y, Lch, terminate)[0] <= 0}

    L = info_length

    Rc = 1 / CC.n
    if terminate:
        Rc = L / (CC.n * L + CC.M)
    nn = 10 * np.log10(Rc)

    SNR_range = np.arange(start=SNR[0], stop=SNR[1], step=SNR_step)
    error = np.zeros((depth, len(SNR_range)))

    ########################################################################################

    for trial in range(depth):
        for i, SNR_dB in enumerate(SNR_range):
            # ----------transmitter-------------
            u = np.random.randint(2, size=L)
            c = CC.encode(u, terminate)[0]
            x = 1 - 2 * c
            # ----------channel-----------------
            SNR = 2 * 10 ** (SNR_dB / 10)
            Lch = 2 * SNR
            mu, sigma = 0, np.sqrt(1 / SNR)
            w = np.random.normal(mu, sigma, size=len(c))
            y = x + w
            # ----------receiver----------------
            u_hat = Dx[decode_type](y, Lch, terminate)
            # ----------error-------------------
            error[trial, i] = np.count_nonzero(u_hat - u)

    error_rate = np.mean(error, axis=0) / L

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.semilogy(SNR_range, error_rate)
    ax1.set_xlabel('$E_s/N_0(dB)$');
    ax1.set_ylabel('Symbol Error Rate ($P_s$)')
    ax1.set_title('Probability of Symbol Error over AWGN channel')
    ax1.set_xlim(SNR_range[0] - 1, SNR_range[-1] + 1);
    ax1.grid(True);
    ax2.semilogy(SNR_range - nn, error_rate)
    ax2.set_xlabel('$E_b/N_0(dB)$');
    ax2.set_ylabel('Bit Error Rate ($P_b$)')
    ax2.set_title('Probability of Bit Error over AWGN channel')
    ax2.set_xlim(SNR_range[0] - 1 - nn, SNR_range[-1] + 1 - nn);
    ax2.grid(True);
    plt.show()


##########################################################################################
# ---------------------------------------SIMULATION---------------------------------------#
##########################################################################################

G = np.array([[1, 0, 1], [1, 1, 1]])
ccSIM(G, 1, decode_type='MLM', depth=1000, SNR=[-10, 10], SNR_step=1, terminate=False, info_length=100)
