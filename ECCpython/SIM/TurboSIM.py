import numpy as np
import matplotlib.pyplot as plt

from TurboCode import TurboCode

def parallel_spc(k, flag, depth = 10, iterations = 5, SNR = [-10, 10], SNR_step = 1):
    from SPC import SPC
    
    spc = SPC(k); TT = TurboCode()

    # initialisation of interleavers
    TT.bInterleaver(flag=0, size=k**2)              # block interleaver
    TT.pInterleaver(flag=1, size=k**2)              # random
    TT.sInterleaver(flag=2, S_value=k, size=k**2)   # s random

    Rc = k**2/((k+2)*k)

    SNR_range = np.arange(start=SNR[0], stop=SNR[1], step=SNR_step)
    error = np.zeros((depth, len(SNR_range)))

    #########################################################################

    for loop in range(depth):
        for s, SNR in enumerate(SNR_range):

            u = np.random.randint(2, size=k**2)
            u_in = TT.Interleaver(flag, u)

            #------------------------encoding-------------------------#

            c1 = []; c2 = []
            for i in range(0, len(u), k):
                c1 = np.append(c1, spc.encode(u[i:i+k])[k]) # column
                #transposing u and interleaving
                c2 = np.append(c2, spc.encode(u_in[i:i+k])[k]) # row
            c12 = np.append(c1, c2)
            c = np.append(u, c12)
            x = 1-2*c

            #--------------------------channel------------------------#

            SNR = 10**(SNR/10)
            mu, sigma = 0, np.sqrt(1/(2*SNR))
            w = np.random.normal(mu, sigma, size=(k+2)*k)
            y = x + w

            #------------------------decoding-------------------------#
            
            y = np.reshape(y, (-1,k))
            yc1 = y[-2]; yc2 = y[-1]
            yu = np.delete(y, [-2,-1], axis=0)
            y_in = TT.Interleaver(flag, yu.flatten())
            y_in = np.reshape(y_in, (-1,k))

            Lu1 = np.zeros((k, k)); Lu2 = np.zeros((k, k))
            Le1 = np.zeros((k, k)); Le2 = np.zeros((k, k))
            
            for _ in range(iterations):
                
                La1 = np.reshape(TT.deInterleaver(flag, Le2.flatten()), (-1,k))
                for i in range(k):
                    y1 = np.append(y[i,:], yc1[i])
                    Lu1[i,:], Le1[i,:] = spc.SPCsoftDecode(y1, SNR, La1[i,:])

                La2 = np.reshape(TT.Interleaver(flag, Le1.flatten()), (-1,k))
                for j in range(k):
                    y2 = np.append(y_in[j,:], yc2[j])
                    Lu2[j,:], Le2[j,:] = spc.SPCsoftDecode(y2, SNR, La2[j,:])
            u_hat = Lu1<=0
            u_hat = u_hat.flatten()

            error[loop, s] = np.count_nonzero(u_hat - u)
    
    error_rate = np.mean(error, axis = 0)/(k**2)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.semilogy(SNR_range, error_rate)
    ax1.set_xlabel('$E_s/N_0(dB)$'); ax1.set_ylabel('Symbol Error Rate ($P_s$)')
    ax1.set_title('Probability of Symbol Error over AWGN channel')
    ax1.set_xlim(SNR_range[0]-1, SNR_range[-1]+1); ax1.grid(True);
    nn = 10*np.log10(Rc)
    ax2.semilogy(SNR_range - nn, error_rate)
    ax2.set_xlabel('$E_b/N_0(dB)$'); ax2.set_ylabel('Bit Error Rate ($P_b$)')
    ax2.set_title('Probability of Bit Error over AWGN channel')
    ax2.set_xlim(SNR_range[0]-1-nn, SNR_range[-1]+1-nn); ax2.grid(True);
    plt.show()
 
######################################################################################################
######################################################################################################
 
def parallel_rsc(N, flag, puncture_pattern, depth = 10, iterations = 5, SNR = [-10, 10], SNR_step = 1):
    from ConvCode import ConvCode

    TT = TurboCode(); CC = ConvCode([[1,0,1], [1,1,1]], 1)

    # initialisation of Interleavers
    TT.nInterleaver(flag=0, size=N)                                     # no interleaver
    TT.pInterleaver(flag=1, size=N)                                     # random
    TT.sInterleaver(flag=2, S_value=int(np.sqrt(2*N)), size=N)          # s random

    PP = np.array(puncture_pattern)
    p, q = np.shape(PP)

    SNR_range = np.arange(start=SNR[0], stop=SNR[1], step=SNR_step)
    error = np.zeros((depth, len(SNR_range)))

    ###########################################################################

    for loop in range(depth):
        for s, SNR in enumerate(SNR_range):

            #------------------------processing------------------------#

            u = np.random.randint(2, size=N)
            ui = TT.Interleaver(flag, u)

            #------------------------encoding--------------------------#
            
            c0 = u
            c1 = CC.encode(u, False)[0][::2]
            c2 = CC.encode(ui, False)[0][::2]

            #----------------------puncturing--------------------------#
            
            # c = []
            # for i in range(N):
            #     c = np.append(c, np.multiply([u[i], c1[i], c2[i]], PP[:,i%q]))

            #----------------------modulation--------------------------#            

            x0 = 1-2*c0
            x1 = 1-2*c1
            x2 = 1-2*c2

            #------------------------channel---------------------------#

            S = 2*10**(SNR/10); Lch = 2*SNR
            mu, sigma = 0, np.sqrt(1/S)
            w0 = np.random.normal(mu, sigma, size=len(c0))
            w1 = np.random.normal(mu, sigma, size=len(c1))
            w2 = np.random.normal(mu, sigma, size=len(c2))
            z0 = x0 + w0
            z1 = x1 + w1
            z2 = x2 + w2

            y1 = np.array(zip(z0, z1)).flatten()
            y2 = np.array(zip(TT.Interleaver(flag, z0), z2)).flatten()

            #-----------------------unpuncturing------------------------#

            # y_p = []
            # for j, i in enumerate(range(0, len(y), 3)):
            #     y_p = np.append(y_p, np.multiply([y[i], y[i+1], y[i+2]], PP[:,j%q]))

            # y_u = []; y1 = []; yc2 = []; y2 = []
            # for i in range(0, len(y), 3):
            #     y_u = np.append(y_u, y_p[i])
            #     y1 = np.append(y1, [y_p[i], y_p[i+1]])
            #     yc2 = np.append(yc2, y_p[i+2])

            # y_in = TT.Interleaver(flag, y_u)

            # for i, u_in in enumerate(y_in):
            #     y2 = np.append(y2, [u_in, yc2[i]])

            #--------------------------decoding------------------------#

            Le2 = np.zeros(N)
            for _ in range(iterations):
                # decoder 1
                La1 = TT.deInterleaver(flag, Le2)
                # Ia1 = calc_I(u1,La1); % mutual info between data and La
                Lu1, Le1 = CC.MaxLogMap_decode(y1, Lch, La1, False)
                # Ie1 = calc_I(u,Le1); % mutual info between data and Le

                # decoder 2
                La2 = TT.Interleaver(flag, Le1)
                # Ia2 = calc_I(u2, La2); % mutual info between data and La
                Lu2, Le2 = CC.MaxLogMap_decode(y2, Lch, La2, False)
                # Ie2 = calc_I(u2, Le2); % mutual info between data and Le
            u_hat = Lu1<=0

            error[loop, s] = np.count_nonzero(u_hat - u)
    
    ###########################################################################

    error_rate = np.mean(error, axis = 0)/N
    Rc = len(u)/(len(x0)+len(x1)+len(x2)); nn = 10*np.log10(Rc)

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
 
##########################################################################################
#---------------------------------------SIMULATION---------------------------------------#
##########################################################################################

# parallel_spc(k=4, flag=2, depth=1000, iterations=10, SNR=[-5, 5], SNR_step=0.1)

# PP = [[1, 1, 1 ,1],
#       [1, 0, 0, 0],
#       [0, 0, 1, 0]]

PP = [[1],
      [1],
      [1]] # no puncturing

parallel_rsc(N=200, flag=1, puncture_pattern=PP, depth=100, iterations=10, SNR=[-5, -3], SNR_step=0.25)