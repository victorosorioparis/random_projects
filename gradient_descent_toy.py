
# Almost zero dependencies, only StandardScaler easily substitutable

import numpy as np
from sklearn.preprocessing import StandardScaler


class linear_regression(object):
    '''
    fits a linear regression mapping function based on labeled data
    '''
    
    def __init__(self,dataset,filename='output.csv',iterations=100):
        '''
        Parameters
        ----------
        dataset : str, name of a CSV file delimited by commas.
            Last column must contain the labels. All datatypes must correspond to real-val. numbers.
            
        filename : str, name for the output CSV (optional)
        
        iterations : max iterations for each learning rate

        Returns
        -------
        Saves a CSV several theta vectors for a linear regression.
        First column corresponds to the learning rate used for the gradient descent.
        Second column corresponds to the iterations done.
        Third column corresponds to the intercept.
        Subsequent columns correspond to the actual dataset variables slopes.
        
        '''
        
        # PRE-PROCESSING
        
        self.xinput = np.genfromtxt(dataset, delimiter = ",")
        
        labels_index = self.xinput.shape[1] - 1
        
        self.y_train = self.xinput[:,labels_index]
        
        self.X_train = np.delete(self.xinput, labels_index, 1)
        
        my_scaler = StandardScaler()
        
        my_scaler.fit(self.X_train)
        
        self.X_train = my_scaler.transform(self.X_train)
        
        self.X_train = np.insert(self.X_train, 0, 1, axis=1)

        # PARAMETERS

        self.theta = np.zeros((1,self.X_train.shape[1]))
        
        # Different learning rates to try
        self.learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.04]
        alpha_variants = len(self.learning_rates)
        
        # Max iterations for each learning rate
        self.iterations_list = [iterations for i in alpha_variants]
        
        self.output_matrix = np.zeros((alpha_variants,self.theta.shape[1]+2))
        
        # TESTING LEARNING RATES

        for alpha, it in zip(self.learning_rates,self.iterations_list):
            for i in range(it+1):
                # Uncomment to see convergence process: 
                # print(f"RISK at iteration {i}: {self.obj_func(self.X_train, self.y_train, self.theta}")
                theta = self.update_theta(self.X_train,self.y_train,self.theta,self.alpha)
            self.output_matrix[self.learning_rates.index(alpha),0] = alpha
            self.output_matrix[self.learning_rates.index(alpha),1] = it
            self.output_matrix[self.learning_rates.index(alpha),2] = theta[0,0]
            self.output_matrix[self.learning_rates.index(alpha),3] = theta[0,1]
            self.output_matrix[self.learning_rates.index(alpha),4] = theta[0,2]
            theta = np.zeros((1,self.X_train.shape[1]))
            
        # OUTPUT

        np.savetxt(filename, self.output_matrix, delimiter=",")

    def prediction(x_o,theta):
        total_value = 0
        for i in range(x_o.shape[0]):
            total_value += x_o[i] * theta[0,i]
        return total_value
    
    def obj_func(self):
        total_value = 0
        for i in range(self.X_train.shape[0]):
            sqr_error = ( self.prediction(self.X_train[i,:],self.theta) - self.y_train[i] ) ** 2
            total_value += sqr_error
        scaled_observations = self.X_train.shape[0] * 2
        risk = (1 / scaled_observations) * total_value
        return risk
    
    def update_theta(self,alpha):
        gradient_vector = np.zeros(self.theta.shape)
        for i in range(self.X_train.shape[0]):
            error = self.prediction(self.X_train[i,:],self.theta) - self.y_train[i]
            error_vector = error * self.X_train[i,:]
            temp = gradient_vector.copy()
            gradient_vector = temp + error_vector
        factor = 1 / self.X_train.shape[0]
        scaled_gradient = factor * gradient_vector
        alpha_gradient = alpha * scaled_gradient
        updated_theta = self.theta - alpha_gradient
        return updated_theta

    
        
        
        