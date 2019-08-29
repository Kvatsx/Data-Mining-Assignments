'''
------------------------------------------------------
@Author: Kaustav Vats (kaustav16048@iiitd.ac.in)
@Roll-No: 2016048
@Date: Thursday August 29th 2019 2:14:25 pm
------------------------------------------------------
'''

from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

ArrayPath = "./Weights/"

def generate_data(n_samples=1000, Classes=10, n_features=2):
    X, Y = make_blobs(n_samples=n_samples, centers=Classes, n_features=n_features)
    np.save(ArrayPath + "X.npy", X)
    np.save(ArrayPath + "Y.npy", Y)
    print("[+] Data Generated & Saved to Data/")
    return X, Y

def load_data():
    X = np.load(ArrayPath + "X.npy")
    Y = np.load(ArrayPath + "Y.npy")
    print("[+] Data Loaded")
    return X, Y

def run_simulation_with_repetition(Labels, K=[1, 2, 3, 4]):
    Result = np.zeros((1000, len(K)), dtype=np.float64)
    Count = np.zeros((1000, len(K)), dtype=np.int64)
    for k in range(len(K)):
        for t in range(1000):
            counts = np.zeros(10)
            counts.fill(K[k])
            total_count = K[k]*10
            trials = 0
            while(total_count > 0):
                index = random.randint(0, 999)
                if (counts[Labels[index]] > 0):
                    counts[Labels[index]] -= 1
                    total_count -= 1
                trials += 1
            Result[trials:, k] += 1
            Count[trials, k] += 1
            #print(trials)
    Result /= 1000.0
    # print(sum(Result[:, 0]))
    np.save(ArrayPath+"WithRep.npy", Result)
    np.save(ArrayPath+"CountWithRep.npy", Count)
    plot_graphs(Result)
    # plot_histogram(Count)

def run_simulation_without_repetition(Labels, K=[1, 2, 3, 4]):
    Result = np.zeros((1000, len(K)), dtype=np.float64)
    Count = np.zeros((1000, len(K)), dtype=np.int64)
    for k in range(len(K)):
        for t in range(1000):
            counts = np.zeros(10)
            visited = np.zeros(1000)
            counts.fill(K[k])
            total_count = K[k]*10
            trials = 0
            while(total_count > 0):
                index = random.randint(0, 999)
                if (counts[Labels[index]] > 0 and visited[index] == 0):
                    counts[Labels[index]] -= 1
                    visited[index] = 1
                    total_count -= 1
                trials += 1
            Result[trials:, k] += 1
            Count[trials, k] += 1
            #print(trials)
    Result /= 1000.0
    # print(sum(Result[:, 0]))
    np.save(ArrayPath+"WithoutRep.npy", Result)
    np.save(ArrayPath+"CountWithoutRep.npy", Count)
    plot_graphs_without_rep(Result)
    # plot_histogram_without_rep(Count)


def plot_graphs(Result, K=[1, 2, 3, 4]):
    colors = ['b', 'g', 'r', 'c', 'm']
    for k in range(len(K)):
        plt.plot(np.arange(150), Result[:150, k], colors[k], label = "K={}".format(K[k]))
    plt.xlabel('Sample Size')
    plt.ylabel('Probability')
    plt.title('Probability v/s Sample Size [With-Repetition]')
    plt.legend() 
    plt.savefig("WithRep.png")

def plot_graphs_without_rep(Result, K=[1, 2, 3, 4]):
    colors = ['b', 'g', 'r', 'c', 'm']
    for k in range(len(K)):
        plt.plot(np.arange(150), Result[:150, k], colors[k], label = "K={}".format(K[k]))
    plt.xlabel('Sample Size')
    plt.ylabel('Probability')
    plt.title('Probability v/s Sample Size [Without-Repetition]')
    plt.legend() 
    plt.savefig("WithoutRep.png")

def plot_histogram(Count, K=[1, 2, 3, 4]):
    colors = mcolors.TABLEAU_COLORS
    names = list(colors)
    for k in range(len(K)):
        plt.bar(np.arange(140), Count[:140, k], color=colors[names[k]], label = "K={}".format(K[k]))
    plt.xlabel('Sample Size')
    plt.ylabel('Frequency')
    plt.title('Frequency v/s Sample Size [With-Repetition]')
    plt.legend() 
    plt.savefig("HistoWithRep.png")
            

def plot_histogram_without_rep(Count, K=[1, 2, 3, 4]):
    colors = mcolors.TABLEAU_COLORS
    names = list(colors)
    for k in range(len(K)):
        plt.bar(np.arange(140), Count[:140, k], color=colors[names[k]], label = "K={}".format(K[k]))
    plt.xlabel('Sample Size')
    plt.ylabel('Frequency')
    plt.title('Frequency v/s Sample Size [Without-Repetition]')
    plt.legend() 
    plt.savefig("HistoWithoutRep.png")

if __name__ == "__main__":
    # X, Y = generate_data()
    X, Y = load_data()
    run_simulation_with_repetition(Y)
    run_simulation_without_repetition(Y)

