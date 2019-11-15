import pandas as pd
import numpy as np
import random
from copy import deepcopy
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

def get_array(customer_data):
    '''
    (pandas.dataframe) -> numpy.ndarray
    Return a numpy ndarray with relevant features from given customer_data
    dataframe. Converts categorical data into new columns taking values 0 or 1.
    '''
    # drop unnecessary columns of dataframe
    customer_data = customer_data.drop('CustomerID',axis=1)
    customer_data = customer_data.drop('Spending Score (1-100)', axis=1)
    
    # convert categorical data into binary numerical values
    customer_data = pd.get_dummies(customer_data)

    # get an array representation of dataframe and return it
    customer_features = np.array(customer_data)
    return customer_features

def assign_to_cluster(features, clusterAssignment, centers):
    '''
    (numpy.ndarray, numpy.ndarray, numpy.ndarray) -> NoneType
    Assign each given datapoint from features to a cluster in clusterAssignment
    given centers, the center of each cluster.
    '''
    # loop through every datapoint
    for i in range(0,len(clusterAssignment)):
        # modify clusterAssignment by assigning a new cluster for each
        # datapoint from customer_features
        distances = np.linalg.norm(features[i]-centers,axis=1)
        # each index of distances represents distance to a cluster for a
        # datapoint; choose the cluster with the smallest distance
        clusterAssignment[i] = np.argmin(distances)

def update_center(centers, features, clusterAssignment, k):
    '''
    (numpy.ndarray, numpy.ndarray, numpy.ndarray, integer) -> NoneType
    Update the center of each cluster given current clusters, center,
    feature array features, current labels, clusterAssignment, and number
    of clusters, k.
    '''
    # go through every cluster
    for j in range(0, k):
        # declare variables to help calculate new centers for each cluster
        total_vector = np.zeros(features.shape[1])
        count = 0
        # go through every datapoints current cluster assignment
        for i in range(0, len(clusterAssignment)):
            # if the datapoint is in cluster j, add to the total sum
            if (j == clusterAssignment[i]):
                total_vector += features[i]
                count += 1
        # new center is the average of the total of each datapoint
        centers[j] = total_vector/count

def calculate_dist(features, centers):
    '''
    (np.ndarray, np.ndarray) -> np.ndarray
    For each datapoint given in features, find the distance to the closest
    center given in centers, and return it.
    '''
    # define array to store closest distance for each datapoint
    lowest = np.zeros(len(features))
    # go through every observation and find lowest distance to some cluster
    for i in range(0,len(features)):
        distances = np.linalg.norm(features[i]-centers,axis=1)
        lowest[i] = np.amin(distances)
    return lowest

def run_kmeans_pp(features, k):
    '''
    (numpy.ndarray, integer) -> numpy.ndarray
    Run the kmeans++ algorithm given feature matrix, features, and number
    of cluseters, k. Different from kmeans algorithm because of the
    initialization of clusters.
    '''
    # declare first center at random
    rand = random.randint(0,len(features-1))
    centers = features[rand]
    centers = centers.reshape(1,-1)
    # remove the point
    featuresCopy = np.delete(features,rand,0)
    # initialize k-1 more centers from points
    for i in range(1,k):
        # calculate shortest distance for all points to a cluster
        lowest = calculate_dist(featuresCopy,centers)
        # choose center as point with the highest distance from a center
        centers = np.vstack((centers, featuresCopy[np.argmax(lowest)]))
        featuresCopy = np.delete(featuresCopy, np.argmax(lowest),0)
    
    # algorithm becomes same as kmeans at after intialization
    oldCenters = np.zeros(centers.shape)
    # create an array holding which cluster each data point belongs to
    clusterAssignment = np.zeros(features.shape[0])
    # find the initial error, being the distance between current and old centers
    error = np.linalg.norm(centers - oldCenters)
    # run algorithm until error change in error becomes small
    count = 0
    # run algorithm while the centers are still changing
    while(error > 0):
        # go through each datapoint and assign it to the closest center
        assign_to_cluster(features,clusterAssignment,centers)
        # create a copy of centers and assign it to old centers
        # we need a deep copy because the arrays are mutable, and oldCenters
        # would change along with centers being changed when using '='
        oldCenters = deepcopy(centers)
        # find the new center of each cluster
        update_center(centers,features,clusterAssignment,k)
        # calculate the new error
        error = np.linalg.norm(centers - oldCenters)
        count += 1

    print("K-Means++ number of iterations:", count)
    # plot the clusters and then return them
    my_kmeans_plot(features,clusterAssignment,k, True)
    return clusterAssignment

def run_kmeans(features, k):
    '''
    (numpy.ndarray, integer) -> numpy.ndarray
    Run the kmeans algorithm given feature matrix, features, and number
    of clusters, k.
    '''
    # intitialize k amount of centers by choosing k random data instances
    centers = features[np.random.choice(features.shape[0],k,replace=False)]
    # create an array containing location of previous centers
    oldCenters = np.zeros(centers.shape)
    # create an array holding which cluster each data point belongs to
    clusterAssignment = np.zeros(features.shape[0])
    # find the initial error, being the distance between current and old centers
    error = np.linalg.norm(centers - oldCenters)
    # run algorithm until error change in error becomes small
    count = 0
    # run algorithm while the centers are still changing
    while(error > 0):
        # go through each datapoint and assign it to the closest center
        assign_to_cluster(features,clusterAssignment,centers)
        # create a copy of centers and assign it to old centers
        # we need a deep copy because the arrays are mutable, and oldCenters
        # would change along with centers being changed when using '='
        oldCenters = deepcopy(centers)
        # find the new center of each cluster
        update_center(centers,features,clusterAssignment,k)
        # calculate the new error
        error = np.linalg.norm(centers - oldCenters)
        count += 1

    print("K-Means number of iterations:", count)
    # plot the clusters and then return them
    my_kmeans_plot(features,clusterAssignment,k, False)
    return clusterAssignment

def my_kmeans_plot(features, clusters, k, kPP):
    '''
    (numpy.ndarray, numpy.ndarray, integer, boolean) -> NoneType
    Plot the given clusters after k means algorithm finishes.
    '''
    # declare list of potential colours for cluster k (up to 5)
    colours = ['red', 'blue', 'green', 'yellow', 'purple']
    figure = plt.figure()
    ax = figure.add_subplot('111')
    # go through each cluster and plot its' datapoints
    for j in range(0, k):
        # get all the points which belong to the current cluster and seperate them by gender
        male_points = np.array([features[i] for i in range(len(features)) if clusters[i] == j and features[i][3]==1])
        female_points = np.array([features[i] for i in range(len(features)) if clusters[i] == j and features[i][2]==1])
        # scatter all the points using different symbols for male and female
        ax.scatter(male_points[:,0], male_points[:,1], marker = '.', c=colours[j])
        ax.scatter(female_points[:,0], female_points[:,1], marker = '*', c=colours[j])
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income')
    # create a legend for the data
    star = mlines.Line2D([], [], color = 'k', marker='*', markersize=10, label='Female')
    dot = mlines.Line2D([], [], color = 'k', marker='.', markersize=10, label='Male')
    ax.legend(handles= [star,dot])

    # title the plot based on initialization type
    if (kPP == True):
        ax.set_title("Customer Data Kmeans++ Cluster")
    else:
        ax.set_title("Customer Data Kmeans Cluster")
    
def my_kmeans(features, k, kPP):
    '''
    (numpy.ndarray, integer, boolean) -> numpy.ndarray
    Given feature matrix, features, number of clusters, k, run the kmeans or
    kmeans++ algorithm depending on boolean kPP.
    REQ: 2 <= k <= 5
    '''
    # run kmeans if kPP is false
    if (kPP == False):
        return run_kmeans(features, k)
    # otherwise run kmeans++ algorithm if kPP is true
    else:
        return run_kmeans_pp(features,k)

if __name__ == "__main__":
    # read the customer.csv data and store it in a dataframe
    customer_data = pd.read_csv("customer.csv", header=1)
    # get array representation of data
    customer_features = get_array(customer_data)
    
    # set the parameters for kmeans or kmeans++ (k must be 2 to 5 inclusive)
    k = 5

    # run the algorithm
    kmeansCluster = my_kmeans(customer_features, k, False)
    kmeansClusterplus = my_kmeans(customer_features,k,True)