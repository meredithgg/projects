'''kmeans.py
Performs K-Means clustering
Meredith Green
CS 252 Mathematical Data Analysis Visualization, Spring 2022
'''
from dis import dis
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors


class KMeans():
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''

        data = np.copy(self.data)

        return data

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)

        referecned day 22 notebook
        '''
        sub = pt_1-pt_2
        return np.sqrt(np.dot(sub.T, sub))

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)

        referenced day 22 code
        '''

        distances = [self.dist_pt_to_pt(pt, centroid) for centroid in centroids]
        return distances

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops

        referenced day 22 notebook
        '''

        #randomly choose data point indices
        centroidinds = np.random.choice(np.arange(len(self.data)), size=k, replace=False)
        self.centroids = self.data[centroidinds]


        return self.centroids

    def initialize_plusplus(self, k):
        '''Initializes K-means by setting the initial centroids (means) according to the K-means++
        algorithm

        (LA section only)

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        TODO:
        - Set initial centroid (i = 0) to a random data sample.
        - To pick the i-th centroid (i > 0)
            - Compute the distance between all data samples and i-1 centroids already initialized.
            - Create the distance-based probability distribution (see notebook for equation).
            - Select the i-th centroid by randomly choosing a data sample according to the probability
            distribution.

        referenced day 23 notebook
        '''
        #for the first, intialize randomly
        centroidinds = np.random.choice(np.arange(len(self.data)), size=1, replace=False)
        labels = self.assign_labels(self.data[centroidinds])

        #for the rest of the centroids
        for i in range(k - 1):
            #calculate the probability

            num = np.power([self.dist_pt_to_pt(self.data[i], self.data[centroidinds[labels[i]], :])for i in range(len(self.data))],2)
            denom = np.sum(np.power([self.dist_pt_to_pt(self.data[i], self.data[centroidinds[labels[i]], :])for i in range(len(self.data))], 2))


            probs = num/denom

            #pick a new one based on probability
            index = np.random.choice(np.arange(len(self.data)), size = 1, replace = False, p=probs)
            centroidinds = np.append(centroidinds, index, axis = 0)
            #assign labels so we can get the closest centroid for each 
            labels = self.assign_labels(self.data[centroidinds])
        
        #get the centroids based on ids
        self.centroids = self.data[centroidinds]
        return self.centroids


    def cluster(self, k=2, tol=1e-5, max_iter=1000, verbose=False, init_method = "random"):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all 
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for

        reference day 23 notebook
        '''
        

        # intialize centroids
        self.k = k

        if init_method == "random":
            self.centroids = self.initialize(k)
        elif init_method == "plusplus":
            self.centroids = self.initialize_plusplus(k)
        else:
            print("Please enter valid intializaiton method (random or plusplus).")
            return
        if verbose:
            print("INITIAL CENTROIDS:")
            print(self.centroids)

        self.data_centroid_labels = self.assign_labels(self.centroids)
        new_inertia = self.compute_inertia()
        last_inertia = new_inertia + 1
        count = 0
        while abs(last_inertia - new_inertia) > tol and count < max_iter:
            if verbose:
                print("iteration(s): ", count)
            last_inertia = new_inertia
            self.centroids, diff = self.update_centroids(self.k, self.data_centroid_labels, self.centroids)
            self.data_centroid_labels = self.assign_labels(self.centroids)
            new_inertia = self.compute_inertia()
            if verbose:
                print("Inertia: ", new_inertia)
            count +=1
        self.inertia = new_inertia
        return self.inertia, count

    def cluster_batch(self, k=2, n_iter=1, verbose=False, init_method = "random"):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''

        countcount = 0
        countcountval = 0

        for i in range(n_iter):

            if verbose:
                print("Have ran kmeans " , i + 1, " times.")

            #run kmeans

            new_inertia,count = self.cluster(k, verbose, init_method=init_method)

            countcount+=1
            countcountval+=count

            if verbose:
                print("Iterated " , count, " times and has inertia of ", new_inertia, ".")

            #if first time, set the best to the inertia
            if i == 0:
                best_inertia = new_inertia
                best_centroid_labels = self.data_centroid_labels
                best_centroids = self.centroids

            #if the best so far
            if new_inertia < best_inertia:
                #updating based on fields
                best_centroids = self.centroids
                best_centroid_labels = self.data_centroid_labels
                best_inertia = new_inertia

        #update fields based on best
        self.centroids = best_centroids
        self.data_centroid_labels = best_centroid_labels
        self.inertia = best_inertia

        if verbose:
            print("Best inertia: ", best_inertia)

        return self.inertia, countcountval/countcount

    def assign_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray. shape=(self.num_samps,). Holds index of the assigned cluster of each data sample

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]

        referenced day 23 notebook
        '''

        assignments = [np.argmin(self.dist_pt_to_centroids(datum, centroids)) for datum in self.data]

        return np.array(assignments)

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster.
        
        The basic algorithm is to loop through each cluster and assign the mean value of all 
        the points in the cluster. If you find a cluster that has 0 points in it, then you should
        choose a random point from the data set and use that as the new centroid.

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values
        '''
        self.k = k
        new_centroids = np.zeros([prev_centroids.shape[0],prev_centroids.shape[1]])
        for i in range(k):

            indices = np.array(np.where(np.array(data_centroid_labels) == i))

            #no stuff in cluster
            if not indices.any():
                new_centroids[i,:] = (np.random.choice(np.arange(len(self.data)), size=1, replace=False))

            #if there is stuff in the cluster
            else:
                centroid = np.sum(self.data[indices,:],axis=1)/int(indices.size)
                new_centroids[i,:] = centroid 

        #calculate differences
        differences = new_centroids - prev_centroids

        return new_centroids, differences


    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Parameters:
        -----------
        None

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        sum = 0
        points = len(self.data_centroid_labels)
        for i in range(points):
            sum += self.dist_pt_to_pt(self.data[i], self.centroids[self.data_centroid_labels[i]])**2

        return sum / points

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).

        Based on lab code
        '''

        fig, ax = plt.subplots(1,1, sharex=False, sharey=False, figsize = (10,10))

        scatter = ax.scatter(self.data[:,0], self.data[:,1], edgecolors="black", cmap= "viridis", c=self.data_centroid_labels)
        ax.legend(handles=scatter.legend_elements()[0], 
                labels=[0,1,2],
                title="Centroid")

        ax.scatter(self.centroids[:,0], self.centroids[:,1], marker = "+", s = 100)
        ax.legend(handles=scatter.legend_elements()[0], 
                labels=range(1, len(self.centroids) + 1),
                title="Centroid")
        
        ax.set_title("First vs Second Feature of Data by Cluster")
        ax.set_ylabel("Second Feature")
        ax.set_xlabel("First Feature")

        return self.inertia

    def elbow_plot(self, max_k, n_iter = 1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        n_iter: in. Run k-means with k=i for n_iter iterations

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''

        k_inertias = []
        for k in range(1, max_k + 1, 1):
            inertia = self.cluster_batch(k, n_iter)
            k_inertias.append(inertia)

        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.plot(k_inertias)
        ax.set_xlabel('k')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Plot')
        ax.set_xticks(range(max_k), range(1, max_k + 1, 1))
        plt.show()

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        for centroid in range(self.k):
            indices = np.array(np.where(np.array(self.data_centroid_labels) == centroid))
            self.data[indices] = self.centroids[centroid]

        return (self.data)

    def replace_color_with_centroid_thermo(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None

        reference day 18 notebook for ordering based on order of another array (like we did in PCA)
        '''
        if self.k != 5:
            print("There should be 5 centroids for thermo")

        thermal_centroids = np.array([[0.56471,0.11765,0.06667,1.],[0.94118,0.55294,0.20392,1.],[0.4549,0.97255,0.52157,1.],[0.21569,0.55294,0.96863,1.],[0.,0.09804,0.56863,1.]])
        sums = [centroid[0] + centroid[1] + centroid[2] for centroid in self.centroids]
        
        sums_order = np.argsort(sums)[::-1]

        self.centroids = thermal_centroids[sums_order,:]

        for centroid in range(self.k):
            indices = np.array(np.argwhere(np.array(self.data_centroid_labels) == centroid))
            self.data[indices] = self.centroids[centroid]

        return (self.data)

    def replace_color_with_centroid_warhol(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''

        if self.k != 3:
            print("There should be three centroids for warhol")

        first_centroids = np.array([[0.56471,0.11765,0.06667,1.],[0.94118,0.55294,0.20392,1.],[0.4549,0.97255,0.52157,1.]])
        second_centroids = np.array([[0.21569,0.55294,0.96863,1.],[0.,0.09804,0.56863,1.],[0.29019,0.68235,0.52157,1.]])
        third_centroids = np.array([[0.79569,0.25294,0.96863,1.],[0.8,0.09804,0.66863,1.],[0.59019,0.68235,0.92157,1.]])
        fourth_centroids = np.array([[0.71569,0.55294,0.56863,1.],[0.2,0.29804,0.26863,1.],[0.79019,0.38235,0.62157,1.]])

        data_one = np.zeros((self.data.shape[0], self.data.shape[1]))
        data_two = np.zeros((self.data.shape[0], self.data.shape[1]))
        data_three = np.zeros((self.data.shape[0], self.data.shape[1]))
        data_four = np.zeros((self.data.shape[0], self.data.shape[1]))

        for centroid in range(self.k):
            indices = np.array(np.argwhere(np.array(self.data_centroid_labels) == centroid))
            data_one[indices] = first_centroids[centroid]
            data_two[indices] = second_centroids[centroid]
            data_three[indices] = third_centroids[centroid]
            data_four[indices] = fourth_centroids[centroid]

        self.data = np.concatenate((data_one, data_two, data_three, data_four), axis = 0)

        return (self.data)

    def replace_color_with_centroid_b_a_w(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        if self.k != 2:
            print("There should be two centroids for black and white")

        #centroid 0 is black
        if self.centroids[0, 0] > self.centroids[1,0]:
            self.centroids = np.array([[1,1,1,1],[0,0,0,1]])
        #centroid 0 is white
        else:
            self.centroids = np.array([[0,0,0,1],[1,1,1,1]])

        for centroid in range(self.k):
            indices = np.array(np.argwhere(np.array(self.data_centroid_labels) == centroid))
            self.data[indices] = self.centroids[centroid]

        return (self.data)

        
