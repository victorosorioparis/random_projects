'''
Takes an input.csv file with a labeled dataset for classification

Test most frequently used classifiers with an array of parameters

It chooses the best parameter for each and prints the accuracy of each classifier to terminal

It saves an output file with the results to ease model selection process

'''

import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors, tree

# INITIALIZE

data = pd.read_csv('input.csv')

X = data[['A','B']]

y = data[['label']]

k_fold = 5

RED = '\n\033[1;31;48m'
YELLOW = '\n\033[1;33;48m'
END = '\033[1;37;0m\n'

output_dict = {'model':[],'best_score':[],'test_score':[]}


print(RED + 'SVM LINEAR MODEL' + END)

# MODELLING

SVM_lin_model = svm.SVC(kernel='linear')

C_params = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]}

SVM_lin_model_grid = GridSearchCV(SVM_lin_model,C_params)

SVM_lin_model_grid.fit(X,y.values.ravel())

print("BEST ESTIMATOR C:",SVM_lin_model_grid.best_estimator_)

choosen_SVM_lin_model = SVM_lin_model_grid.best_estimator_

# TESTING

cv_results = cross_validate(choosen_SVM_lin_model, X, y.values.ravel(), cv = k_fold, return_train_score = True)
best_score_1 = max(cv_results['train_score'])
test_score_1 = cv_results['test_score'].mean()

print(YELLOW + "BEST SCORE:" + END,best_score_1)
print(YELLOW + "TEST SCORE:" + END,test_score_1)

output_dict['model'] += ['svm_linear']
output_dict['best_score'] += [best_score_1]
output_dict['test_score'] += [test_score_1]
 
print(RED + 'SVM POLYNOMIAL MODEL' + END)

# MODELLING

SVM_poly_model = svm.SVC(kernel='poly')

params2 = {'C':[0.1, 1, 3],'degree':[4, 5, 6],'gamma':[0.1, 0.5]}

SVM_poly_model_grid = GridSearchCV(SVM_poly_model,params2)

SVM_poly_model_grid.fit(X,y.values.ravel())

print("BEST ESTIMATOR POLY:",SVM_poly_model_grid.best_estimator_)

choosen_SVM_poly_model = SVM_poly_model_grid.best_estimator_

# TESTING

cv_results = cross_validate(choosen_SVM_poly_model, X, y.values.ravel(), cv = k_fold, return_train_score = True)
best_score_2 = max(cv_results['train_score'])
test_score_2 = cv_results['test_score'].mean()

print(YELLOW + "BEST SCORE:" + END,best_score_2)
print(YELLOW + "TEST SCORE:" + END,test_score_2)

output_dict['model'] += ['svm_polynomial']
output_dict['best_score'] += [best_score_2]
output_dict['test_score'] += [test_score_2]

print(RED + 'SVM RBF MODEL' + END)

# MODELLING

SVM_rbf_model = svm.SVC(kernel='rbf')

params3 = {'C':[0.1, 0.5, 1, 5, 10, 50, 100],'gamma':[0.1, 0.5, 1, 3, 6, 10]}

SVM_rbf_model_grid = GridSearchCV(SVM_rbf_model,params3)

SVM_rbf_model_grid.fit(X,y.values.ravel())

choosen_SVM_rbf_model = SVM_rbf_model_grid.best_estimator_

# TESTING

cv_results = cross_validate(choosen_SVM_rbf_model, X, y.values.ravel(), cv = k_fold, return_train_score = True)
best_score_3 = max(cv_results['train_score'])
test_score_3 = cv_results['test_score'].mean()

print(YELLOW + "BEST SCORE:" + END,best_score_3)
print(YELLOW + "TEST SCORE:" + END,test_score_3)

output_dict['model'] += ['svm_rbf']
output_dict['best_score'] += [best_score_3]
output_dict['test_score'] += [test_score_3]


print(RED + 'LOGISTIC MODEL' + END)

# MODELLING

log_model = LogisticRegression()

params4 = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]}

log_model_grid = GridSearchCV(log_model,params4)

log_model_grid.fit(X,y.values.ravel())

choosen_log_model = log_model_grid.best_estimator_

# TESTING

cv_results = cross_validate(choosen_log_model, X, y.values.ravel(), cv = k_fold, return_train_score = True)
best_score_4 = max(cv_results['train_score'])
test_score_4 = cv_results['test_score'].mean()

print(YELLOW + "BEST SCORE:" + END,best_score_4)
print(YELLOW + "TEST SCORE:" + END,test_score_4)

output_dict['model'] += ['logistic']
output_dict['best_score'] += [best_score_4]
output_dict['test_score'] += [test_score_4]

print(RED + 'KNN' + END)

# MODELLING

knn = neighbors.KNeighborsClassifier()

params5 = {'n_neighbors':[i + 1 for i in range(50)],'leaf_size':[i * 5 for i in range(1,13)]}

knn_grid = GridSearchCV(knn,params5)

knn_grid.fit(X,y.values.ravel())

choosen_knn_model = knn_grid.best_estimator_

# TESTING

cv_results = cross_validate(choosen_knn_model, X, y.values.ravel(), cv = k_fold, return_train_score = True)
best_score_5 = max(cv_results['train_score'])
test_score_5 = cv_results['test_score'].mean()

print(YELLOW + "BEST SCORE:" + END,best_score_5)
print(YELLOW + "TEST SCORE:" + END,test_score_5)

output_dict['model'] += ['knn']
output_dict['best_score'] += [best_score_5]
output_dict['test_score'] += [test_score_5]


print(RED + 'DECISION TREE' + END)

# MODELLING

dtree = tree.DecisionTreeClassifier()

params6 = {'max_depth':[i + 1 for i in range(50)],'min_samples_split':[i for i in range(2,11)]}

dtree_grid = GridSearchCV(dtree,params6)

dtree_grid.fit(X,y.values.ravel())

choosen_dtree_model = dtree_grid.best_estimator_

# TESTING

cv_results = cross_validate(choosen_dtree_model, X, y.values.ravel(), cv = k_fold, return_train_score = True)
best_score_6 = max(cv_results['train_score'])
test_score_6 = cv_results['test_score'].mean()

print(YELLOW + "BEST SCORE:" + END,best_score_6)
print(YELLOW + "TEST SCORE:" + END,test_score_6)

output_dict['model'] += ['decision_tree']
output_dict['best_score'] += [best_score_6]
output_dict['test_score'] += [test_score_6]


print(RED + 'RANDOM FOREST' + END)

# MODELLING

forest = RandomForestClassifier()

params7 = {'max_depth':[i + 1 for i in range(50)],'min_samples_split':[i for i in range(2,11)]}

forest_grid = GridSearchCV(forest,params7)

forest_grid.fit(X,y.values.ravel())

choosen_forest_model = forest_grid.best_estimator_

# TESTING

cv_results = cross_validate(choosen_forest_model, X, y.values.ravel(), cv = k_fold, return_train_score = True)
best_score_7 = max(cv_results['train_score'])
test_score_7 = cv_results['test_score'].mean()

print(YELLOW + "BEST SCORE:" + END,best_score_7)
print(YELLOW + "TEST SCORE:" + END,test_score_7)

output_dict['model'] += ['random_forest']
output_dict['best_score'] += [best_score_7]
output_dict['test_score'] += [test_score_7]


# SAVING OUTPUT

filename = 'output.csv'

output_df = pd.DataFrame(output_dict)

output_df.to_csv('output.csv')

        
        
        