# Vadim Litvinov
# 314552365
import numpy as np
import scipy, sys
import scipy.io
from scipy.io import wavfile
import random
MAX_ITER = 30

sample, centroids = sys.argv[1], sys.argv[2]
# fs is the sampling rate, y is the samples
fs, y = scipy.io.wavfile.read(sample)
x = np.array(y.copy())
centroids = np.loadtxt(centroids)

# initializes random centroids in the boundaries
def init_centroids():
    k = 16
    centroids_tmp = [[] for i in range(k)]
    for i in range(k):
        centroids_tmp[i] = [random.randint(-10000, 10000),
                           random.randint(-10000, 10000)]
    centroids_tmp = np.array(list(centroids_tmp))
    return centroids_tmp

# dictionary from point index to centroid
def create_file():
    return open("output.txt", "w")

def create_dictionary():
    return [[] for i in range(len(centroids))]

def update_centroids(centroids, cntr_to_samp, conv_flag):
    i = 0
    for c in centroids:
        if len(cntr_to_samp[i]) != 0:
            if not np.array_equal(c, np.around(np.sum(cntr_to_samp[i], axis=0) / len(cntr_to_samp[i]))):
                centroids[i] = np.around(np.sum(cntr_to_samp[i], axis=0) / len(cntr_to_samp[i]))
                conv_flag = False
        i = i + 1
    return conv_flag

if __name__ == '__main__':
    #centroids = init_centroids()
    # the number of centroids given in input file
    k = len(centroids)
    file = create_file()
    cntr_to_tamp = create_dictionary()
    #avgLossList = []
    # when none of the centroids updated, it converged
    for itr in range(MAX_ITER):
        convFlag = True
        avg_loss = 0
        # sample index
        i = 0
        # the samples loop
        for sample in x:
            min_dist_cntrd = 0
            # centroid index
            j = 0
            for c in centroids:
                if (np.linalg.norm(sample - c) <
                        (np.linalg.norm(sample - centroids[min_dist_cntrd]))):
                    min_dist_cntrd = j
                j = j + 1
            cntr_to_tamp[min_dist_cntrd].append(sample)
            avg_loss += np.linalg.norm(sample - centroids[min_dist_cntrd]) ** 2
            i = i + 1
        # now we shall update the centroids
        avg_loss = avg_loss / len(x)
        #avgLossList.append(avgLoss)
        convFlag = update_centroids(centroids, cntr_to_tamp, convFlag)
        file.write(f"[iter {itr}]:{','.join([str(i) for i in centroids])}\n")
        if convFlag:
            break
        cntr_to_tamp = create_dictionary()
    # plotting for the report
    '''
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], avgLossList)
    plt.ylabel('average loss')
    plt.xlabel('iteration')
    plt.title("K = 16")
    plt.show()
    '''
