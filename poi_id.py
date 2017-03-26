import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#convert to pandas df for transformation/exploration
df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))
df.set_index(employees, inplace=True)

#format df to compute summary stats
df.replace("NaN", np.nan, inplace = True)
pd.to_numeric(df['bonus'], errors = 'coerce')

### Task 2: Remove outliers
#2 rows dropped as they skew the data
df.drop('TOTAL', inplace=True)
#a business, not a POI
df.drop('THE TRAVEL AGENCY IN THE PARK', inplace=True)

### Task 3: Create new feature(s)
#create and compute values for a new feature
df['percent_of_total_messages_to_poi'] = df['from_this_person_to_poi']/df['from_messages']

### Store to my_dataset for easy export below.
#features_list to include the new feature
features_list = ['poi',
                'salary', 
                'total_payments', 
                'loan_advances', 
                'bonus', 
                'expenses', 
                'exercised_stock_options', 
                'other', 
                'long_term_incentive', 
                'director_fees', 
                'from_poi_to_this_person', 
                'from_this_person_to_poi', 
                'shared_receipt_with_poi', 
                'percent_of_total_messages_to_poi',
                'deferred_income', 
                'deferral_payments', 
                'restricted_stock', 
                'restricted_stock_deferred', 
                'total_stock_value']

#convert the df to dict, and replace nan with 0
df.replace('NaN', 0, inplace=True)
df_dict = df.to_dict('index')

#set my_dataset to reflect the changes made during pandas exploration
my_dataset = df_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#use stratified shuffle split instead of train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)

for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

### Task 4: Create a Classifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

#create pipeline elements
pca = PCA()
tree = tree.DecisionTreeClassifier()
scaler = MinMaxScaler()
skbest = SelectKBest()

#create pipeline
pipe = Pipeline(steps=[("scaler", scaler), 
                        ("pca", pca), 
                        ("skbest", skbest), 
                        ('tree', tree)])

#parameter dict to pass to gridsearch
parameters = {'pca__n_components':[3,4,5], 
              'pca__whiten':[True, False], 
              'skbest__k':[1,2,3], 
              'tree__min_samples_split':[10,20,30,40], 
              'tree__criterion':['entropy']}

grid_search = GridSearchCV(pipe, parameters, cv=cv, scoring='f1')

grid_search.fit(features, labels)

#create classifier from the best gridsearch estimator
clf = grid_search.best_estimator_

#make prediction
pred = clf.predict(features_test)

#run tests using tester.py
from tester import test_classifier
print "Tester Classification report"
test_classifier(clf, my_dataset, features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 

#completed using gridsearchcv, pca, and selectkbest above

        

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)