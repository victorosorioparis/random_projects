import numpy as np

class recommendation_system(object):
    
    def __init__(self,ratings_dataset,lam=2,sigma2=0.1,d=5,filename='output.csv'):
        '''
        Parameters
        ----------
        ratings_dataset : str, name of a CSV delimited with commas
            Contains a dataset of rows with first column being user id, second column object id
            and third column any type of numerical rating.
            
        lam : int, optional
            Bias penalty. The default is 2
            
        sigma2 : 0 > float < 1, optional
            Variance. The default is 0.1
            
        d : int, optional
            Embedding dimensions for the users/object vectors. The default is 5.

        Returns
        -------
        Saves three CSV file with comma separated values:
            V-matrix: contains the users vectors 
            U-matrix: contains the objects vectors
            objectives: contains progression of objective function, to evaluate quality of solution
        '''

        train_data = np.genfromtxt(ratings_dataset, delimiter = ",")

        # WRANGLING

        # removes rows with nan
        self.clean_data = train_data[~np.isnan(train_data).any(axis=1)]
        
        self.set_users, self.set_objects = set(self.clean_data[:,0].astype(int)), set(self.clean_data[:,1].astype(int))
        
        self.M_matrix = np.zeros((int(max(self.set_users)),int(max(self.set_objects))))
        
        self.observed_data = []

        for i in range(self.clean_data.shape[0]):
            user = self.clean_data[i,0].astype(int)
            movie = self.clean_data[i,1].astype(int)
            rating = self.clean_data[i,2]
            self.M_matrix[user-1,movie-1] = rating
            self.observed_data += [(user,str(movie))]

        self.u_matrix = np.zeros((self.M_matrix.shape[0],d))
        
        self.v_matrix = np.random.choice(2,(self.M_matrix.shape[1],d),replace=True)
        
        self.users_dict = {a:self.those_that_include(a) for a in self.set_users}

        self.objects_dict = {a:self.those_that_include(str(a)) for a in self.set_objects}
        
        # OUTPUT
        
        L = self.PMF(self.M_matrix,self.u_matrix,self.v_matrix,lam,sigma2,d)

        np.savetxt("objective.csv", L, delimiter=",")

    def those_that_include(self,what):
        where = self.observed_data
        return [pair for pair in where if what in pair]

    
    def update_location(self,of,M_matrix,u_matrix,v_matrix,lam,sigma,d):
        '''
        Parameters
        ----------
        of : 'users' to update users location or 'objects', likewise
        
        Returns
        -------
        an updated u_matrix or v_matrix accordingly.
    
        '''
        if of == 'users':
            for user in self.set_users:
                ratings_list = self.users_dict[user]
                items_sum = np.zeros((1,d))
                second_term = 0
                for item in ratings_list:
                    item_vector = v_matrix[int(item[1])-1,:]
                    product_vector = item_vector * item_vector.reshape(1,-1)
                    items_sum += product_vector
                    rating_product = M_matrix[user-1,int(item[1])-1] * item_vector
                    second_term += rating_product
                first_term = np.linalg.pinv(lam * np.identity(d) * items_sum)
                new_user_vector = np.dot(first_term, second_term)
                u_matrix[user-1,:] = new_user_vector
        if of == 'objects':
            for obj in self.set_objects:
                ratings_list = self.objects_dict[obj]
                users_sum = np.zeros((1,d))
                second_term = 0
                for user in ratings_list:
                    user_vector = u_matrix[int(user[0])-1,:]
                    product_vector = user_vector * user_vector.reshape(1,-1)
                    users_sum += product_vector
                    rating_product = M_matrix[user[0]-1,obj-1] * user_vector
                    second_term += rating_product
                first_term = np.linalg.pinv(lam * np.identity(d) * users_sum)
                new_obj_vector = np.dot(first_term, second_term)
                v_matrix[obj-1,:] = new_obj_vector
            return ratings_list
            
    def calculate_loss(self,M_matrix,u_matrix,v_matrix,lam,sigma,d):
        u_errors_sum = 0
        users_norm_sum = 0
        objects_norm_sum = 0
        for user in self.set_users:
            # calculating users errors
            ratings_list = self.users_dict[user]
            for item in ratings_list:
                rating = M_matrix[user-1,int(item[1])-1]
                user_vector_t = u_matrix[user-1,:].reshape(1,-1)
                movie_vector = v_matrix[int(item[1])-1]
                error = (rating - np.dot(user_vector_t,movie_vector)) ** 2
                u_errors_sum += (1 / (2 * sigma)) * error
                
            # calculating users variance
            users_norm = np.linalg.norm(u_matrix[user-1,:])
            users_norm_sum += (lam / 2) * (users_norm ** 2)
    
        for obj in self.set_objects:
            ratings_list = self.objects_dict[obj]
            # calculating objects variance
            object_norm = np.linalg.norm(v_matrix[obj-1,:])
            objects_norm_sum += (lam / 2) * (object_norm ** 2)
            
        return - u_errors_sum - users_norm_sum - objects_norm_sum
    
    def PMF(self,M_matrix,u_matrix,v_matrix,lam,sigma,d):
        iterations = 0
        L = np.zeros((50,1))
        while iterations != 50:
            iterations += 1
            for step in ('users','objects'):
                self.update_location(step, M_matrix, u_matrix, v_matrix, lam, sigma, d)
                L[iterations-1] = self.calculate_loss(M_matrix, u_matrix, v_matrix, lam, sigma, d)
        np.savetxt("U-matrix.csv", u_matrix, delimiter=",")
        np.savetxt("V-matrix.csv", v_matrix, delimiter=",")
        return L
    