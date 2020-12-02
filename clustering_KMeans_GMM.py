'''
Executable from terminal with first argument being the dataset filename (csv, delimited by commas)
It prints 
'''

import numpy as np
import scipy.stats as stats
import pandas as pd
import sys

# PARAMETERS

clusters_nr = 5

till_convergence = False

max_iterations = 10

def normalize_data(data):
    # get the max and min value for each dimension
    max_list = []
    min_list = []
    for i in range(data.shape[1]):
        dimension_vector = data[:,i]
        dimension_vector_t = dimension_vector.transpose()
        max_nr = np.amax(dimension_vector_t)
        max_list += [max_nr]
        min_nr = np.amin(dimension_vector_t)
        min_list += [min_nr]
    
    new_matrix = np.zeros(data.shape)
    
    # recalculate each value of the data matrix and fill new matrix with it
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            new_val = ( data[j,i] - min_list[i] ) / ( max_list[i] - min_list[i] )
            new_matrix[j,i] = new_val
    
    return new_matrix

def standardize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

X = normalize_data(np.genfromtxt(sys.argv[1], delimiter = ","))

X = np.genfromtxt(sys.argv[1], delimiter = ",")

print(X[0:10,:])

def KMeans(data):
    # initialize 5 random centroids
    def initialize_mu(clusters_nr,X):
        '''
        clusters_nr = k
        X_d = nr of dimensions of X
        ---
        Output: a ndarray of k rows and d dimensions
        '''
        min_val = float(X[:,0].mean() * 0.5)
        max_val = float(X[:,0].mean() * 1.5)
        values_set = np.linspace(min_val, max_val, num=100)
        init_vector = np.random.choice(values_set,(clusters_nr,1),replace=False)
        for i in range(1,X.shape[1]):
            min_val = float(X[:,i].mean() * 0.5)
            max_val = float(X[:,i].mean() * 1.5)
            values_set = np.linspace(min_val, max_val, num=100)
            new_vector = np.random.choice(values_set,(clusters_nr,1),replace=False)
            temp = init_vector
            init_vector = np.concatenate((temp,new_vector),axis=1)
        return init_vector
        
    mu = initialize_mu(clusters_nr,data)
    
    cluster_assignment = np.random.choice([i + 1 for i in range(clusters_nr)],(data.shape[0],1))
    
    def c_update(data,mu,cluster_assignment):
        '''
        data : ndarray of d dimensions and n rows
        mu : ndarray vector of d dimensions and k rows
        cluster_assignment : ndarray vector of n rows and 1 dimension
        ---
        Output: an updated cluster_assignment vector, with the best k value for each datapoint considering euclidean distance
        '''
        distances = []
        for i in range(data.shape[0]):
            for j in range(clusters_nr):
                xi = data[i,:]
                di = np.linalg.norm(xi - mu[j])
                distances += [di]
                if len(distances) == clusters_nr:
                    best = distances.index(min(distances)) + 1
                    cluster_assignment[i] = best
                    distances = []
        return cluster_assignment
    
    def mean_update(data,mu,cluster_assignment):
        '''
        data : ndarray of d dimensions and n rows
        mu : ndarray vector of d dimensions and k rows
        cluster_assignment : ndarray vector of n rows and 1 dimension
        ---
        Output: an updated mu vector, with the mean in d dimensions for each k assigned datapoints
        '''
        mu_new = np.zeros((clusters_nr,data.shape[1]))
        total_count = 0
        total_values = np.zeros((data.shape[0],1))
        for i in range(clusters_nr):
            for j in range(data.shape[0]):
                if cluster_assignment[j] == (i + 1):
                    total_values =+ data[j,:]
                    total_count += 1
            average = (1 / total_count) * total_values
            mu_new[i,:] = average
            total_count = 0
            total_values = np.zeros((data.shape[0],1))
        return mu_new
    
    def mean_update2(data,mu,cluster_assignment):
        '''
        data : ndarray of d dimensions and n rows
        mu : ndarray vector of d dimensions and k rows
        cluster_assignment : ndarray vector of n rows and 1 dimension
        ---
        Output: an updated mu vector, with the mean in d dimensions for each k assigned datapoints
        '''
        X_dataframe = pd.DataFrame(data)
        X_dataframe.insert(data.shape[1],'cluster',cluster_assignment)
        
        # create different dataframes based on the cluster assignment
        new_mu = np.zeros((clusters_nr,data.shape[1]))
        for i in range(clusters_nr):
            df_k = X_dataframe[X_dataframe['cluster']==i+1]
            df_k = df_k.mean()
            for j in range(data.shape[1]):
                new_mu[i,j] = df_k.iloc[j]
        return new_mu
            
        
        

    iteration_nr = 0
    
    
    if not till_convergence:
        while True:
            cluster_assignment = c_update(data,mu,cluster_assignment)
            mu = mean_update2(data,mu,cluster_assignment)
            iteration_nr += 1
        
            if iteration_nr == max_iterations:
                print("centroids-" + str(iteration_nr) + ".csv saved")
                filename = "centroids-" + str(iteration_nr+1) + ".csv"
                np.savetxt(filename, mu, delimiter=",")
                return mu


# GMM
  
pi_raw = np.empty((1,clusters_nr))
pi_raw.fill(1/clusters_nr)


def initialize_mu(clusters_nr,X):
    '''
    clusters_nr = k
    X_d = nr of dimensions of X
    ---
    Output: a ndarray of k rows and d dimensions
    '''
    min_val = float(X[:,0].min())
    max_val = float(X[:,0].max())
    values_set = np.linspace(min_val, max_val, num=100)
    init_vector = np.random.choice(values_set,(clusters_nr,1),replace=False)
    for i in range(1,X.shape[1]):
        min_val = float(X[:,i].min())
        max_val = float(X[:,i].max())
        values_set = np.linspace(min_val, max_val, num=100)
        new_vector = np.random.choice(values_set,(clusters_nr,1),replace=False)
        temp = init_vector.copy()
        init_vector = np.concatenate((temp,new_vector),axis=1)
    return init_vector

mu_plus = initialize_mu(clusters_nr,X)

sigma_plus = []
for i in range(clusters_nr):
    new_cov = np.identity(X.shape[1])
    sigma_plus += [new_cov]
    
fi_matrix = np.zeros((X.shape[0],clusters_nr))

def update_fi(X,mu_plus,pi_raw,sigma_plus,fi_matrix):
    '''
    Output: an ndarray with an updated version of fi_matrix
    '''
    # update the fi_matrix
    for row in range(X.shape[0]):
        for k in range(clusters_nr):
            distribution = stats.multivariate_normal.pdf(X[row,:],mean=mu_plus[k,:], cov=sigma_plus[k],allow_singular=True)
            fi_matrix[row,k] = pi_raw[0,k] * distribution
    sum_ = fi_matrix.sum(axis=1)
    for row in range(X.shape[0]):
        for k in range(clusters_nr):
            if np.isnan(fi_matrix[row,k] / sum_[row]):
                fi_matrix[row,k] = 0
            else:
                fi_matrix[row,k] = fi_matrix[row,k] / sum_[row]
    return fi_matrix

def update_pi_raw(fi_matrix):
    total = fi_matrix.shape[0]
    sum_ = fi_matrix.sum(axis=0)
    new_pi_raw = np.zeros((1,clusters_nr))
    for i in range(clusters_nr):
        new_pi_raw[0,i] = sum_[i] / total
    return new_pi_raw, sum_

def update_mu_plus(mu_plus,sum_,fi_matrix,X):
    for k in range(clusters_nr):
        k_sum = np.zeros((1,X.shape[1]))
        for i in range(X.shape[0]):
            k_sum+= (fi_matrix[i,k] * X[i,:])
        k_mu = k_sum / sum_[k]
        mu_plus[k,:] = k_mu
    return mu_plus

def update_sigma_plus(mu_plus,sum_,fi_matrix,sigma_plus,X):
    for k in range(clusters_nr):
        k_sigma = np.zeros((X.shape[1],X.shape[1]))
        for i in range(X.shape[0]):
            new_vector = X[i,:] - mu_plus[k,:]
            new_matrix = new_vector * new_vector.reshape((-1, 1))
            new_matrix_s = fi_matrix[i,k] * new_matrix
            k_sigma += new_matrix_s
        k_sigma = k_sigma / sum_[k]
        sigma_plus[k] = k_sigma
    return sigma_plus
            

def e_m_step(X,mu_plus,pi_raw,sigma_plus,fi_matrix):

    new_fi = update_fi(X,mu_plus,pi_raw,sigma_plus,fi_matrix)
    
    new_pi_raw, sum_ = update_pi_raw(new_fi)
    
    new_mu_plus = update_mu_plus(mu_plus, sum_, new_fi, X)
    
    new_sigma_plus = update_sigma_plus(new_mu_plus,sum_,new_fi,sigma_plus,X)
    
    return new_fi, new_pi_raw, new_mu_plus, new_sigma_plus

if __name__ == '__main__':
    
    centroids = KMeans(X)
    
    for it in range(max_iterations):
        print("ITERATION",it + 1,":")
        fi_matrix, pi_raw, mu_plus, sigma_plus = e_m_step(X,mu_plus,pi_raw,sigma_plus,fi_matrix)
        print("Done.\n")
        
    filename = "pi-matrix.csv" 
    np.savetxt(filename, pi_raw.transpose(), delimiter=",")
    filename = "mu-matrix.csv"
    np.savetxt(filename, mu_plus, delimiter=",")

    for j in range(clusters_nr):
        filename = "Sigma-" + str(j+1) + "-matrix.csv"
        np.savetxt(filename, sigma_plus[j], delimiter=",")
