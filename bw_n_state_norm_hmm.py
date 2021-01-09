'''
    MAT 515 Paper by Oguzcan Adabuk
    This program is an improved version of the matlab program i submitted on 5/22/2020.
    Main improved features are support for 4 state Normal HMM, graphig returns of FTSE 100.
    I also made additions to connect to stock broker API and fetch real time data for stock
    evaluation but because that would require my personal brokerage account I omitted that part of the code.
'''
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import json
import sys
import os
import alpaca_connector as ap


def update_mu(row):
    tot_xt = 0
    for t in range(0, n):
        tot_xt = tot_xt + uhat[row][t] * xvect[t]
    return tot_xt / uhat[row].sum()


def update_sigma(row):
    tot = 0
    for z in range(0, n):
        tot = tot + uhat[row][z] * (xvect[z] - mu[row]) ** 2
    return math.sqrt(tot / uhat[row].sum())


# Number of states for the HMM
state_num = 2

# Find data and HMM config files
relative_path = os.getcwd()
config_path = "config_" + str(state_num) + "_state.json"
full_config_path = relative_path + "\\" + config_path
full_data_path = relative_path + "\\" + "ftse100.txt"

'''
ticker = 'MRNA'
date = '2020-06-04'
'''

# Start the program
if __name__ == "__main__":
    '''
    xvect = np.array(ap.get_stock_returns(ticker, date))
    xvect = xvect[400:]
    n = len(xvect)
    '''

    xvect = []
    #data_url = "C:\\Users\\adabu\\Documents\\Applied Math - DePaul\\MAT 515 - Financial Modeling\\Paper_05_20\\spy500.txt"
    # fetch ftse100 data
    with open(full_data_path) as f:
        for line in f:
            line = line.strip('\n')
            xvect.append(float(line))

    # crop the data tiny bit to avoid underflow
    xvect = xvect[420:]
    n = len(xvect)

    # Load HMM state configuration
    with open(full_config_path) as f:
        config = json.load(f)

    # Initialize HMM parameters
    states = config["states"]
    mu = config["mu"]
    sigma = config["sigma"]
    dvect = np.array(config["dvect"])
    Gmat = np.array(config["Gmat"])

    alpha_mat = np.zeros([n, states])
    beta_mat = np.zeros([n, states])
    ptensor = np.zeros([n + 1, states, states])
    vhat = np.zeros([n, states, states])
    uhat = np.zeros([states, n])

    for iter in range(0, 15):

        for t in range(0, n):
            m = xvect[t]
            p = []
            for l in range(0, states):
                p.append(norm.pdf(m, loc=mu[l], scale=sigma[l]))

            Pxt_mat = np.diag(p)
            ptensor[t] = Pxt_mat

        # Calculate alpha
        alpha_mat[0] = np.matmul(dvect, ptensor[0])
        for j in range(1, n):
            a = np.matmul(alpha_mat[j - 1], Gmat)
            b = np.matmul(a, ptensor[j])
            alpha_mat[j] = b

        # Calculate beta
        beta_mat[n - 1] = np.ones([1, states])
        for j in range(n - 1, 0, -1):
            a = np.matmul(Gmat, ptensor[j])
            b = np.matmul(a, beta_mat[j].transpose())
            beta_mat[j - 1] = b

        likelihood = np.matmul(alpha_mat[2], beta_mat[2].transpose())
        if abs(likelihood - np.matmul(alpha_mat[states], beta_mat[states].transpose()) > 0.1):
            print("Warning: likelihood is wrong!")

        # E Step
        # Update values of vhat
        for j in range(0, states):
            for t in range(0,n):
                uhat[j][t] = alpha_mat[t][j] * beta_mat[t][j] / likelihood

        for j in range(0, states):
            for k in range(0, states):
                for t in range(1, n):
                    a = alpha_mat[t - 1][j]
                    g = Gmat[j][k]
                    p = ptensor[t][k][k]
                    b = beta_mat[t][k]
                    expr1 = a * g * p * b
                    # print(str(a) + " * " + str(g) + " * " + str(p) + " * " + str(b) + " = " + str(expr1))
                    vhat[t][j][k] = expr1 / likelihood

        # M Step
        # Update dvect
        for y in range(0, states):
            dvect[y] = uhat[y][0]

        # Calculate new Gmat
        freq = np.zeros([states, states])
        sum = 0
        for j in range(0, states):
            for k in range(0, states):
                for m in range(0, n):
                    sum = sum + vhat[m][k][j]
                freq[j][k] = freq[j][k] + sum
                sum = 0

        for j in range(0, states):
            for k in range(0, states):
                Gmat[j][k] = freq[j][k] / freq[j].sum()

        # Update MUs and Sigmas for Normal-HMM
        for i in range(0, states):
            mu[i] = update_mu(i)
            sigma[i] = update_sigma(i)

        print("***New Params***")
        print("mus:" + str(mu))
        print(" sigmas:" + str(sigma))
        print("****************")

    print('output of 3 State Normal HMM')
    for i in range(0, len(mu)):
        print('mu' + str(i) + ' = ', mu[i])
        print('sigma' + str(i) + ' = ', sigma[i])

    print(Gmat)

    ticker = ""
    date = ""
    ax = sns.distplot(xvect, bins=len(xvect), kde=True, color='red',
                      hist_kws={'linewidth': 15, 'alpha': 1, 'color': "skyblue"})
    ax.set(xlabel=ticker + " FTSE 100 returns", ylabel='return')
    plt.show()