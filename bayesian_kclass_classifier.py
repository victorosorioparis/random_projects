import numpy as np
import math

class bayesian_classifier(object):
    
    def __init__(self,X_train,y_train,X_test):
        '''

        Parameters
        ----------
        X_train : TYPE
            DESCRIPTION.
        y_train : TYPE
            DESCRIPTION.
        X_test : TYPE
            DESCRIPTION.

        Returns
        -------
        Trains a model on X_train, y_train and uses X_test to save a CSV with name probs_test:
            each row correspond to each observation in X_test
            each column correspond to the probabilities of belonging to each class (cardinal order)
        
        New predictions can be made later on by using the function new_prediction
        '''
        # Setting data
        self.X_train = np.genfromtxt(X_train, delimiter=",")
        self.y_train = np.genfromtxt(y_train)
        X_test = np.genfromtxt(X_test, delimiter=",")
        
        # Classifying
        final_outputs = self.pluginClassifier(self.X_train, self.y_train, X_test)

        vector_total = np.sum(final_outputs,axis=1)

        normalized_output = np.zeros(final_outputs.shape)
        
        for i in range(final_outputs.shape[0]):
            for j in range(final_outputs.shape[1]):
                normalized_output[i,j] = final_outputs[i,j] / vector_total[i]

        # Writing output file
        np.savetxt("probs_test.csv", normalized_output, delimiter=",")
        
    def new_prediction(self,X_not):
        '''
        Parameters
        ----------
        X_not : a ndarray
            a matrix with problem observation rows and same dimensions as unlabeled train set.

        Returns
        -------
        final_outputs : a ndarray of probabilities that each observation belongs to the different labels

        '''
        final_outputs = self.pluginClassifier(self.X_train, self.y_train, X_not)
        return final_outputs

    def calculate_mu_label(X_train,y_train,cat,indices):
        '''
        Parameters
        ----------
        X_train : Features training numpy array with nth rows
        y_train : Categories training numpy array with nth rows
        cat : int, category name
        indices : list of integers, index of X_train corresponding to the cat
    
        Returns
        -------
        the class_mu, a vector with the average value (float) for each feature for that class.
    
        '''
        # I create a new array with only the rows that contain the indices of that category
        
        cat_X = np.take(X_train,indices,0)
        
        cat_count = len(indices)
        
        single_vector_sum_all = np.sum(cat_X,0)
        
        class_mu = np.true_divide(single_vector_sum_all,cat_count)
        
        return class_mu
        
    
    def calculate_sigma_label(X_train,y_train,cat,indices,class_mu):
        '''
    
        Parameters
        ----------
        X_train : Features training numpy array with nth rows
        y_train : Categories training numpy array with nth rows
        cat : int, category name
        indices : list of integers, index of X_train corresponding to the cat
        class_mu : single row np array with the class_mu
    
        Returns
        -------
        A numpy n x n array with the covariance matrix for that cat.
    
        '''
        cat_X = np.take(X_train,indices,0)
        
        cat_count = len(indices)
        
        X_dim = X_train.shape[1]
        
        normalized_cat_X = np.zeros((X_dim,X_dim))
        
        for i in range(cat_count):
            new_x = np.take(cat_X,i,0)
            new_i = np.subtract(new_x,class_mu)
            new_i2 = np.outer(new_i,new_i.transpose())
            normalized_cat_X = np.add(normalized_cat_X,new_i2)
            
        normalized_matrix = normalized_cat_X / cat_count
        
        print("NORMALIZED CAT:",normalized_cat_X.shape)
        print("NORMALIZED MAT:",normalized_matrix.shape)
        
        return normalized_matrix
        
        
    
    def scan_labels(self,X_train,y_train):
        '''
        Parameters
        ----------
        X_train : Features training numpy array with nth rows
        y_train : Categories training numpy array with nth rows
    
        Returns
        -------
        A dictionary: keys correspond to y categories. Values are tuples with: empirical fraction of category (float),
                                                                               a list of index numbers of rows with that category,
                                                                               class mu for that label
                                                                               class sigma for that label
    
        '''
        labels_dict = {}
        
        # keep the number of rows
        print("CHECK Y TRAIN SHAPE",y_train.shape)
        n_rows = y_train.shape[0]
        
        # counting appearances of each cat and leaving counting as first pos of tuple value in dict
        # also adding index numbers for each category to second pos of tuple value (list)
        for i in range(n_rows):
            cat = y_train[i]
            if cat in labels_dict.keys():
                labels_dict[cat][0] += 1
                labels_dict[cat][1] += [i]
            else:
                labels_dict[cat] = [1,[i],None,None]
        
        # transforming the count into an empirical fraction and calculating the class mu
        for i in labels_dict.keys():
            temp = labels_dict[i][0]
            labels_dict[i][0] = temp / n_rows
            
            # adding the class mu single-dimension array to the third pos of tuple
            labels_dict[i][2] = self.calculate_mu_label(X_train,y_train,i,labels_dict[i][1])
            
            # adding the class sigma array to the fourth pos of tuple
            labels_dict[i][3] = self.calculate_sigma_label(X_train,y_train,i,labels_dict[i][1],labels_dict[i][2])
            
        return labels_dict
            
    
    def density_function_prediction(par_tuple, x_test):
        '''
        Parameters
        ----------
        par_tuple : a tuple with the following parameters relative to the class
                                                                               empirical fraction of category (float),
                                                                               a list of index numbers of rows with that category,
                                                                               class mu for that label
                                                                               class sigma for that label
        x_test : a single row np array with the x_not vector to predict
    
        Returns
        -------
        A probability (float) for that class to be true given x_test.
    
        '''
        
        class_prior = par_tuple[0]
    
        # x_not - class_mu
        density_function_1 = np.subtract(x_test, par_tuple[2])
        
        density_function_1t = density_function_1.transpose()
        
        # inverse of class sigma
        density_function_2 = np.linalg.pinv(par_tuple[3])
        
        # x_not - class_mu
        density_function_3 = np.subtract(x_test, par_tuple[2])
        
        # exponential terms multiplication
        
        density_function_mult1 = np.matmul(density_function_2,density_function_3)
        
        density_function_mult2 = np.matmul(density_function_1t,density_function_mult1)
        
        density_function_mult = np.dot(-0.5,density_function_mult2)
        
        # exponentiation of the terms
        
        density_function_exp = math.exp(density_function_mult)
        
        # final operations
        
        determinant_matrix = np.linalg.det(par_tuple[3])
        density_covariance_factor = 1 / np.sqrt(determinant_matrix)
        
        np.savetxt("matrix.csv", par_tuple[3], delimiter=",") # write output to file
        
        density_function = class_prior * density_covariance_factor * density_function_exp
        
        return density_function
        
    
    def pluginClassifier(self,X_train, y_train, X_test):   
        '''
        Parameters
        ----------
        X_train : Features training numpy array with nth rows
        y_train : Categories training numpy array with nth rows
        X_test : Features csv to make predictions with
    
        Returns
        -------
        A np array with: predicted probabilities for each class (a dimension for each label) for the nr of rows of X_test.
    
        '''
    
      # getting the parameters for each class to plug into the classifier
        labels_dict = self.scan_labels(X_train, y_train)
      
      # creating an empty array matrix to fill up with the probabilities
        n_rows = X_test.shape[0]
        cats = int(max(labels_dict.keys()))
        
        output_matrix = np.zeros((n_rows,cats+1),float)
        
        for row in range(n_rows):
            for cat in range(cats+1):
                  x_not = X_test[row,:]
                  if cat not in labels_dict.keys():
                      continue
                  probability = self.density_function_prediction(labels_dict[cat],x_not)
                  print(probability)
                  output_matrix[row,cat] = probability
        
        return output_matrix