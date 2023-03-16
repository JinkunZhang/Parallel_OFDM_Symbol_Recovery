# -*- coding: utf-8 -*-
import numpy as np
import random
from time import time
import sys,argparse
from pyspark import SparkContext
from operator import add
from pyspark.mllib.random import RandomRDDs


def QAM_Constellation(M_Mod):
    """ Generate the QAM Constellation. Output is an np.ndarray with M_Mod elements, each is a complex number"""
    d = np.sqrt(6. / (M_Mod-1)) # distance between constellation
    L = int(np.sqrt(float(M_Mod))) # Layers of constellation
    #print('L = ',L)
    Const = np.array([(1.*p + 1.j*q - (1.+ 1.j)/ 2 *L + (0.5+0.5j))*d for p in range(L) for q in range(L)])
    return Const

def Clip(x,A,z):
    """ Generate clipped signal given z"""
    if z == 0:
        return x
    elif z == 1:
        return x * (A/np.absolute(x))
    else:
        print(' Input z not binary! ',z)
        return 0.+0.j

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def Update_z(x,y_bar,Var,A):
    """ Update z(n) , return a 0/1 integer z_new(n)"""
    square_clipped = np.absolute(y_bar - Clip(x,A,1))**2
    square_unclipped = np.absolute(y_bar - x)**2
    p = sigmoid(1. / Var**2 * (square_clipped - square_unclipped))
    z_new = np.random.binomial(1,p,size=1)
    #print('p = ',p,' z_new = ',z_new) 
    return z_new

def Compute_xnGivenX(n,l,xn,Xl,N,Constellation):
    """Compute_xnGivenX used in updating x, return a ndarray of M_Mod length, with u-th of it being x^(old)_{n,Xl = Su}"""
    return xn + (Constellation - Xl) / np.sqrt(N) * np.exp( 2.j * np.pi * n * l / N)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Parallele Bayesian Detection.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
    parser.add_argument('--N',default=2048,type=int,help ="Number of carriers (length of seqence)")
    parser.add_argument('--IsNewData',default=False,type=bool,help ="If generate new simulation data set")
    parser.add_argument('--Data',default='ParallelDataSet.npy',type=str,help ="File saving/loading the generated data, saving when arg.NewData == True, loading o.w.")
    parser.add_argument('--SNRdB',default=30.0,type=float,help ="SNR in dB")
    parser.add_argument('--Kmax',default=4,type=int,help ="Number of iteration")
    parser.add_argument('--Npartition',default=20,type=int, help='Parallelization Level')
    parser.add_argument('--a0',default=10.0,type=float, help='Hyperparameter for Inverse Gamma Prior')
    parser.add_argument('--b0',default=2.0,type=float, help='Hyperparameter for Inverse Gamma Prior')
 
    args = parser.parse_args()

    sc = SparkContext("local["+str(args.Npartition)+"]",appName='Parallel Bayesian Detection')

# Generate / load data
    N = args.N
    M_Mod = 16 # Modulation number of QAM
    if args.IsNewData:
        X_init = np.random.randint(low = 0, high = M_Mod, size = N, dtype = int)
        np.save(args.Data,X_init)
        print('Data saved at '+args.Data)
    else:
        X_init = np.load(args.Data)
        print('Loading from '+args.Data)

# Basic Settings
    A = 4.42
    M_channel = 5
    eps = 1e-4

# QAM
    Constellation = QAM_Constellation(M_Mod)
    #print(Constellation)
    X_freq = np.array([Constellation[i] for i in X_init]) # X_freq has normalized average power

# Transmitted Signal
    x_time = np.sqrt(N) * np.fft.ifft(X_freq)
    x_time_clipped = np.array([Clip(x_time[n],A, np.absolute(x_time[n]) > A ) for n in range(N)])
    X_freq_clipped = 1./np.sqrt(N) * np.fft.fft(x_time_clipped)

# AWGN channel
    h = np.array([np.exp(-1.*m) for m in range(M_channel)] + [0 for i in range(N-M_channel)])
    H = np.fft.fft(h)
    SNR = 10. **(args.SNRdB / 10.)
    Var_W = 1. / SNR
    W = np.random.normal(0, np.sqrt(Var_W), N) + 1j * np.random.normal(0, np.sqrt(Var_W), N) # Complex AWG noise

# Received Signal
    Y_freq = X_freq_clipped.dot(H) + W

# Perfect channel Equalization
    #Y_bar_freq = Y_freq.dot(1. / H)
    Y_bar_freq = np.array([ Y_freq[l] / H[l] for l in range(N) ])
    #print('type(Y_freq) = ',type(Y_freq))
    #print('type(1. / H) = ',type(1. / H))
    #print('type(Y_bar_freq) = ',type(Y_bar_freq))
    y_bar_time = np.sqrt(N) * np.fft.ifft(Y_bar_freq)

# Random initial point
    print('Generating Initail Point')
    X_freq_old_idx = np.random.randint(low = 0, high = M_Mod, size = N, dtype = int)
    X_freq_old = np.array([Constellation[i] for i in X_freq_old_idx])
    x_time_old = np.sqrt(N) * np.fft.ifft(X_freq_old)
    z_old = np.random.randint(low = 0, high = 2, size = N, dtype = int)
    a0 = args.a0
    b0 = args.b0
    Var_old = random.gammavariate(a0,b0)

    #print(Clip(1+1j,1))
    
# Parallelize
    time_tuple = [(n,(x_time_old[n], y_bar_time[n], z_old[n])) for n in range(N)] # list [(n, (xold(n),ybar(n),zold(n)) ) ]
    time_tuple_rdd = sc.parallelize(time_tuple,numSlices=args.Npartition)
    freq_tuple = [(l, X_freq_old[l]) for l in range(N) ] #list[(l, Xold(l))]
    freq_tuple_rdd = sc.parallelize(freq_tuple,numSlices=args.Npartition)

# Start iteration
    start = time()
    for k in range(args.Kmax):
        print('Iteration No. ',k)

        # Update {z}
        time_tuple_rdd = time_tuple_rdd.mapValues(lambda tu: (tu[0],tu[1],Update_z(tu[0],tu[1],Var_old,A))).cache() # list [(n, (xold(n),ybar(n),znew(n)) )]

        # Generate cartesian product, will be used in updating {x}
        joined_rdd = time_tuple_rdd.cartesian(freq_tuple_rdd).map(lambda tu: ( (tu[0][0],tu[1][0]),(tu[0][1][0],tu[0][1][1],tu[0][1][2],tu[1][1]) )) # list[ ( (n,l),(xoldn,ybarn,znewn,Xoldl) )]

        # Generate {xnGivenX}
        xnGivenX_rdd = joined_rdd.map(lambda tu: ( tu[0][1], ( tu[0][0], tu[1][1], tu[1][2], Compute_xnGivenX(tu[0][0],tu[0][1],tu[1][0],tu[1][3],N,Constellation) ))) # list[ ( l, (n, znewn, ybarn, array(xnGivenX for u in range(M_Mod))) ]

        # Generate {Pr(Xl = Su)}
        PrXlSu_n_rdd = xnGivenX_rdd.mapValues(lambda tu: (tu[0], np.ndarray([ np.absolute( tu[2] - Clip(tu[3][int(u)],A,tu[1]) )**2 for u in range(M_Mod) ]) )) # list [ (l, (n, array( |yn - xn|^2 )) ) ]
        PrXlSu_rdd = PrXlSu_n_rdd.mapValues(lambda tu: tu[1]).reduceByKey(lambda array1,array2: array1 + array2) # list [(l , \sum_n{array( |yn - xn|^2 )})]

        # Update {Xl}
        freq_tuple_rdd = PrXlSu_rdd.mapValues( lambda array: Constellation[np.argmin(array)] ).cache()

        # Update {xn} i.e. fft
        #fft_rdd = time_tuple_rdd.cartesian(freq_tuple_rdd) # list[ (  (n,(xoldn,ybarn,znewn)) , (l, Xl)  )  ]
        #fft_rdd = fft_rdd.map(lambda tu: (tu[0][0], (tu[1][1] / np.sqrt(N) * np.exp(2.j * np.pi * tu[0][0] * tu[1][0] / N), tu[0][1][1], tu[0][1][2] ) ) ) # list[(n, (Xl*exp(2j \pi nl/N),ybarn,znewn))]
        #print('fft_rdd.count() = ',fft_rdd.count())
        #print(fft_rdd.take(1))
        #fft_rdd = fft_rdd.reduceByKey(lambda tu1,tu2: ( tu1[0] + tu2[0] ,tu1[1],tu1[2])) # list [(n, (\sum ..., ybarn, znewn))]
        #time_tuple_rdd = fft_rdd.cache() #  list [(n, (xnew(n),ybar(n),znew(n)) )]

        # Update Var_\epsilon
        aN = a0 + 1.* N / 2.
        squareSum = time_tuple_rdd.map(lambda tu: np.absolute(tu[1][1] - Clip(tu[1][0],A,tu[1][2]))**2).reduce(add)
        bN = b0 + 0.5 * squareSum
        Var_new = random.gammavariate(aN,bN)

        Var_old = Var_new
        
    RunTime = time() - start
    print('Run Time : ',RunTime)

