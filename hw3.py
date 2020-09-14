#!/usr/bin/env python3

import random
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


#TODO: Read the input file and store it in the data structure
def read_data(path):
    """
    Read the input file and store it in data_set.

    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        path: path to the dataset

    Returns:
        data_set: a list of data points, each data point is itself a list of features:
            [
                [x_1, ..., x_n],
                ...
                [x_1, ..., x_n]
            ]
    """
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():             
            line = line.strip()                

            x = line.split(",")
            temp = []
            for item in x:
                temp.append(float(item))
            
            data.append(temp)
            # print(data)
    # print(data)
    return data

# TODO: Select k points randomly from your data set as starting centers.
def init_centers_random(data_set, k):
    """
    Initialize centers by selecting k random data points in the data_set.
    
    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        data_set: a list of data points, where each data point is a list of features.
        k: the number of mean/clusters.

    Returns:
        centers: a list of k elements: centers initialized using random k data points in your data_set.
                 Each center is a list of numerical values. i.e., 'vals' of a data point.
    """
    identical = True
    while identical == True:
        temp = random.sample(data_set, k)
        identical = False
        for i in range(len(temp)): 
            for i1 in range(len(temp)): 
                if i != i1: 
                    if temp[i] == temp[i1]: 
                        identical = True
    return temp


# TODO: compute the euclidean distance from a data point to the center of a cluster
def dist(vals, center):
    """
    Helper function: compute the euclidean distance from a data point to the center of a cluster

    Args:
        vals: a list of numbers (i.e. 'vals' of a data_point)
        center: a list of numbers, the center of a cluster.

    Returns:
         d: the euclidean distance from a data point to the center of a cluster
    """
    # if len(vals) > 0 and len(vals) == len(center):

    #     diff = 0
    #     for i in range(len(vals)):
    #         diff = diff + ( (vals[i] - center[i])**2 )
    #     return math.sqrt(diff)
    #     #return diff
    # else:
    #     print("vals: ", vals)
    #     print("center: ", center)
    #     exit(1)
    
    d = np.linalg.norm(np.asarray(vals)-np.asarray(center))
    return d

# TODO: return the index of the nearest cluster
def get_nearest_center(vals, centers):
    """
    Assign a data point to the cluster associated with the nearest of the k center points.
    Return the index of the assigned cluster.

    Args:
        vals: a list of numbers (i.e. 'vals' of a data point)
        centers: a list of center points.

    Returns:
        c_idx: a number, the index of the center of the nearest cluster, to which the given data point is assigned to.
    """

    min_val = dist(vals, centers[0])
    min_index = 0
    for i in range(1, len(centers)):
        temp = dist(vals, centers[i])
        if temp < min_val:
            min_val = temp
            min_index = i
    return min_index

    


# TODO: compute element-wise addition of two vectors.
def vect_add(x, y):
    """
    Helper function for recalculate_centers: compute the element-wise addition of two lists.
    Args:
        x: a list of numerical values
        y: a list of numerical values

    Returns:
        s: a list: result of element-wise addition of x and y.
    """
    if len(x) == len(y):
        s = []
        for i in range(len(x)):
            s.append(x[i]+y[i])
        return s
    else:
        print("error, vect_add two vector size not match")
        exit(1)

# TODO: averaging n vectors.
def vect_avg(s, n):
    """
    Helper function for recalculate_centers: Averaging n lists.
    Args:
        s: a list of numerical values: the element-wise addition over n lists.
        n: a number, number of lists

    Returns:
        s: a list of numerical values: the averaging result of n lists.
    """
    s_new = []
    for i in range(len(s)):
        s_new.append(s[i]/n)
    return s_new


# TODO: return the updated centers.
def recalculate_centers(clusters):
    """
    Re-calculate the centers as the mean vector of each cluster.
    Args:
         clusters: a list of clusters. Each cluster is a list of data_points assigned to that cluster.

    Returns:
        centers: a list of new centers as the mean vector of each cluster.
    """
    centers = []
    for cluster in clusters:
        if len(cluster) == 0:
            #print("empty cluster")
            temp_vector = []
        else:
            temp_vector = [0] * len(cluster[0])
            for item in cluster:
                temp_vector = vect_add(temp_vector, item)
            temp_vector = vect_avg(temp_vector, len(cluster))
        centers.append(temp_vector)
    return centers


# TODO: run kmean algorithm on data set until convergence or iteration limit.
def train_kmean(data_set, centers, iter_limit):
    """
    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        data_set: a list of data points, where each data point is a list of features.
        centers: a list of initial centers.
        iter_limit: a number, iteration limit

    Returns:
        centers: a list of updates centers/mean vectors.
        clusters: a list of clusters. Each cluster is a list of data points.
        num_iterations: a number, num of iteration when converged.
    """
    
    temp_label = [0] * len(data_set) # initiallize everything to class 0
    num_iterations = 0
    convergence = False
    while (num_iterations < iter_limit) and (convergence == False):
        convergence = True

        for data_index in range(len(data_set)): # assign label to every points
            temp_label_of_item = get_nearest_center(data_set[data_index], centers)
            if (temp_label[data_index] != temp_label_of_item):
                convergence = False
                temp_label[data_index] = temp_label_of_item

        clusters = [ [] for _ in range(len(centers)) ]
        for j in range(len(data_set)): # recalculate centers
            clusters[temp_label[j]].append(data_set[j])
        centers = recalculate_centers(clusters)
        
        for center_index in range(len(centers)): # check if there is empty centers
            if centers[center_index] == []:
                centers[center_index] = init_centers_random(data_set, 1)[0] #assign a new data point
        num_iterations = num_iterations + 1
    return centers, clusters, num_iterations
        


# TODO: helper function: compute within group sum of squares
def within_cluster_ss(cluster, center):
    """
    For each cluster, compute the sum of squares of euclidean distance
    from each data point in the cluster to the empirical mean of this cluster.
    Please note that the euclidean distance is squared in this function.

    Args:
        cluster: a list of data points.
        center: the center for the given cluster.

    Returns:
        ss: a number, the within cluster sum of squares.
    """
    ss = 0
    
    for item in cluster:
        ss = ss + (dist(item, center))**2
    # print("within_cluster_ss: ", ss)
    return ss


# TODO: compute sum of within group sum of squares
def sum_of_within_cluster_ss(clusters, centers):
    """
    For total of k clusters, compute the sum of all k within_group_ss(cluster).

    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        clusters: a list of clusters.
        centers: a list of centers of the given clusters.

    Returns:
        sss: a number, the sum of within cluster sum of squares for all clusters.
    """
    sss = 0
    for i in range(len(centers)):
        sss = sss + within_cluster_ss(clusters[i], centers[i])
    return sss


if __name__ == "__main__":
    path = "wine.txt"
    data_set = read_data(path)

    #[2] test init_centers_random
    #[a, b] = init_centers_random(data_set, 2)
    #print("center: ", [a, b])
    #print("dist: ", dist(a, b))


    #[3] test get_nearest_center
    #print(get_nearest_center([10, 10, 10], [[100, 100, 100], [11, 11, 12]]))

    #[4] test vect_add
    #print(vect_add([1,2,3,4], [4,5,6,7]))

    #[5] test vect_avg
    #print(vect_avg([100.0,100.0,100.0], 15))

    # test every
    centers = init_centers_random(data_set, 3)
    iter_limit = 100


    y_value = []
    y_value_2 = []
    for i in range(2,11):
        centers = init_centers_random(data_set, i)
        centers, clusters, iter_number = train_kmean(data_set, centers, iter_limit)
        y_value.append(sum_of_within_cluster_ss(clusters, centers))
        y_value_2.append(iter_number)

    print("value of ss from 2 to 10: ", y_value)

    fig, ax = plt.subplots()
    fig.suptitle('Changes with different k_value')
    ax.plot([2,3,4,5,6,7,8,9,10], y_value)
    ax.set(xlabel='k_value', ylabel='sum of all k within_group_ss(cluster)')
    ax.grid()
    plt.tight_layout()
    # fig, ax = plt.subplots(2)
    # fig.suptitle('Changes with different k_value')
    # ax[0].plot([2,3,4,5,6,7,8,9,10], y_value)
    # ax[0].set(xlabel='k_value', ylabel='sum of all k within_group_ss(cluster)')
    # ax[0].grid()

    # ax[1].plot([2,3,4,5,6,7,8,9,10], y_value_2)
    # ax[1].set(xlabel='k_value', ylabel='iter_number')
    # ax[1].grid()


    fig.savefig("result.png")
    # plt.show() # csil does not apply show
