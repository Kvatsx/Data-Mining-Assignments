'''
------------------------------------------------------
@Author: Kaustav Vats (kaustav16048@iiitd.ac.in)
@Roll-No: 2016048
@Date: Saturday August 24th 2019 4:45:26 pm
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


def get_histogram(Labels):
    Histogram = np.zeros(np.unique(Labels).shape[0])
    for i in range(Labels.shape[0]):
        Histogram[Labels[i]-1] += 1
    print(Histogram)

def get_probability(Labels, sample_size, min_data_points, WithReplacement, Trials=1000, AvgCount=1):
    if (10*min_data_points > sample_size):
        return 0
    else:
        Prob = [0 for i in range(AvgCount)]
        for avg in range(AvgCount):
            true = 0
            for t in range(Trials):
                ElementCount = [min_data_points for i in range(10)]
                random_data = get_random_data(sample_size, WithReplacement)
                for rd in random_data:
                    ElementCount[Labels[rd]] -= 1
                flag = True
                for ec in ElementCount:
                    if (ec > 0):
                        flag = False
                        break
                if (flag):
                    true += 1
            Prob[avg] = true/Trials
        return sum(Prob)/AvgCount


def run_simulation(Labels, ArrayName, K=[1, 2, 3, 4], WithReplacement=True):
    MaxSampleSize = 160
    Result = np.zeros((MaxSampleSize, len(K)), dtype=np.float64)
    for k in tqdm(range(len(K))):
        for i in tqdm(range(MaxSampleSize)):
            Result[i, k] = get_probability(Labels, i, K[k], WithReplacement)
        # for i in range(1, MaxSampleSize):
        #     Result[i, k] += Result[i-1, k]
    np.save(ArrayPath + ArrayName + ".npy", Result)
    plot_graphs(MaxSampleSize, Result, WithReplacement, ArrayName)


def plot_graphs(MaxSampleSize, Result, WithReplacement, Name, K=[1, 2, 3, 4]):
    colors = ['b', 'g', 'r', 'c', 'm']
    for k in range(len(K)):
        plt.plot(np.arange(MaxSampleSize), Result[:, k], colors[k], label = "K={}".format(K[k]))
    plt.xlabel('Sample Size')
    plt.ylabel('Probability')
    if (WithReplacement):
        plt.title('Probability v/s Sample Size [With-Repetition]')
    else:
        plt.title('Probability v/s Sample Size [Without-Repetition]')
    plt.legend() 
    plt.savefig(Name + ".png")

def plot_graphs2(MaxSampleSize, Result, WithReplacement, Name, K=[1, 2, 3, 4]):
    colors = mcolors.TABLEAU_COLORS
    names = list(colors)
    for k in range(len(K)):
        plt.plot(np.arange(MaxSampleSize), Result[:, k], colors[names[k]], label = "K={}".format(K[k]))
    plt.xlabel('Sample Size')
    plt.ylabel('Probability')
    if (WithReplacement):
        plt.title('Probability v/s Sample Size [With-Repetition]')
    else:
        plt.title('Probability v/s Sample Size [Combined]')
    plt.legend() 
    plt.savefig(Name + ".png")

#ref- https://pynative.com/python-random-sample/
def get_random_data(sample_size, WithReplacement, SampleCount=1000):
    if (WithReplacement):
        random_data = np.random.choice(SampleCount, size=sample_size, replace=True)        
        return random_data
    else:
        random_data = np.random.choice(SampleCount, size=sample_size, replace=False)
        return random_data


if __name__ == "__main__":
    # X, Y = generate_data()
    X, Y = load_data()
    # get_histogram(Y)
    # for i in range(3):
    #     print(get_random_data(10))
    # run_simulation(Y, "Result1", WithReplacement=True)
    # run_simulation(Y, "Result2", WithReplacement=False)
    plot_graphs(160, np.load(ArrayPath+"Result1.npy"), True,  "Result1")
    plot_graphs2(160, np.load(ArrayPath+"Result2.npy"), False, "Result12")
    