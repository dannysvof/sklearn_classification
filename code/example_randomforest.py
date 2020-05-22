#fontes:
#https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#https://www.kaggle.com/tcvieira/simple-random-forest-iris-dataset
import argparse
import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import pickle

#load data 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv('../data/pima-indians-diabetes.data.csv', names= names)
array = dataframe.values

#split data in features(X) and classes(Y)
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33 # test data proportion
seed = 7

#split data in train and test subsets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

#create an instance of the machine learning algorithm and train(fit) over the training data
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)

#save the model to disk
pickle.dump(model, open('../model/rfmodel', 'wb'))

#load model from disk
loaded_model = pickle.load(open('../model/rfmodel', 'rb'))
result = loaded_model.score(X_test, Y_test)
#average result for the test data
print(result)

#Interactive 
import sys
print('-------------------------------------------------------------')
print('binary classification example by using randomforest algorithm')
print('-------------------------------------------------------------')
print('digit a number greater than 0 and lower than %i' %(len(X_test)))
for line in sys.stdin:
    try:
        z = int(line)
        dato = X_test[z]
        print('input data : X')
        print(dato)
        val = loaded_model.predict([dato])
        print('predicted y-value : %i  -- expected y-value : %i \n'%(val[0], Y_test[z]))
        print('digit a number greater than 0 and lower than %i' %(len(X_test)))
    except Exception as e:
        print('invalid input\n')
        print('digit a number greater than 0 and lower than %i'%(len(X_test)))
