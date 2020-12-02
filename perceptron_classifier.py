import numpy as np
import sys

class perceptron(object):
    '''
    Fits a boundary in the input space for classification problems based on perceptron alg.
    '''
    def __init__(self,X,filename='output.csv'):
        '''
        Parameters
        ----------
        X : string of the filename relative to a CSV delimited by commas
            The input data, with labels as dimension -1.
        
        filename : str with name for the CSV for the output (optional)
            Default name is 'output.csv'

        Returns
        -------
        Outputs a CSV file with a matrix of shape iterations * dimensions.
        Final matrix dimension will depend of iterations needed until convergence.
        Each row corresponds to best fit parameters vector, where last column is the intercept.

        '''
        self.xinput = np.genfromtxt(X, delimiter = ",")
        self.perceptron_matrix = np.zeros((1,self.xinput.shape[1]))
        self.filename = sys.argv[2]
        np.savetxt(self.filename, self.perceptron_output(self.xinput,self.perceptron_matrix), delimiter=",")

    def perceptron_prediction(x_0,perceptron):
        total_value = 0
        for i in range(2):
            total_value += (x_0[i] * perceptron[0,i])
        total_value += perceptron[0,2]
        if total_value > 0:
            total_value = 1
        else:
            total_value = -1
        return total_value
        
    def perceptron_fit(self,xinput,perceptron,iteration):
        new_perceptron = perceptron.copy()
        errors = 0
        for i in range(xinput.shape[0]):
            new_vector = xinput[i,:].copy()
            new_vector[2] = 1
            if (xinput[i,2] * self.perceptron_prediction(xinput[i,:],perceptron)) <= 0:
                errors += 1
                new_v = xinput[i,2] * new_vector
                temp = new_perceptron.__add__(new_v)
                new_perceptron = temp
            if errors == 0:
                no_errors = True
            else:
                no_errors = False
        print(new_perceptron)
        return new_perceptron, no_errors
    
    def perceptron_output(self,xinput,perceptron):
        iteration = 1
        output_matrix = np.zeros((1,3))
        first_vector, no_errors = self.perceptron_fit(xinput,perceptron,iteration)
        perceptron = first_vector.copy()
        output_matrix = output_matrix.__add__(first_vector)
        while not no_errors:
            iteration += 1
            new_vector, no_errors = self.perceptron_fit(xinput,perceptron,iteration)
            perceptron = new_vector.copy()
            output_matrix = np.concatenate((output_matrix,perceptron),axis=0)
        return output_matrix





