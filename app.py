import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.set_option('float_format', '{:.3f}'.format)

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import add_dummy_feature


# %matplotlib inline
warnings.simplefilter(action='ignore', category=FutureWarning)

RNDM = 1990

print('Jump2Digital 2022 Data Science')
print('Working...')

"""# Dataset"""

# Train dataset
try:
    train = pd.read_csv('https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Jump2digital+2022/train.csv', sep=';')
except:
    train = pd.read_csv('/train.csv', sep=';')
train.head()

# Test dataset
try:
    test = pd.read_csv('https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Jump2digital+2022/test.csv', sep=';')
except:
    test = pd.read_csv('test.csv', sep=';')
test.head()

# Function for checking NaN, Shape and Target-values in train and test
def na0(df):
    return [df.isin([0]).sum().sum(),df.isin([1]).sum().sum(),df.isin([2]).sum().sum(), df.isna().sum().sum(), df.shape]

displ=[]
for df in train, test:
    displ.append(na0(df)) 
pd.DataFrame(displ, columns=['zeros','ones','twos', 'NaN', 'shape'])

df = train.copy()


target = df.groupby('target')['feature1'].count().reset_index()

# Correlation

df_corr = df.drop('target', axis=1)
corr = df_corr.corr()


# Data Preprocessing

# copying train dataset

train_pre = train.copy()

# Train Test Split

X = train_pre.drop(['target'], axis=1)
y = train_pre['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RNDM, stratify=y)

features = X.columns


"""**Model: Random Forest**"""

rf = RandomForestClassifier(random_state = RNDM)
rf.fit(X_train, y_train)

rf.score(X_train, y_train)

rf.score(X_test, y_test)

"""The model is overfitted."""

y_pred = rf.predict(X_test)

cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, linewidths=1, annot=True, fmt='g',cmap="BuPu")

print(classification_report(y_test, y_pred))


"""Most important features: 3 & 6.

# Features
"""

# copying train dataset
train_ftr = train.copy()

train_ftr['dummy'] = np.random.rand(2100)

train_ftr.head()

X = train_ftr.drop(['target'], axis=1)
y = train_ftr['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RNDM, stratify=y)

features = X.columns
features

rf = RandomForestClassifier(random_state = RNDM)
rf.fit(X_train, y_train)

rf.score(X_train, y_train)

rf.score(X_test, y_test)

y_pred = rf.predict(X_test)

cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, linewidths=1, annot=True, fmt='g',cmap="BuPu")

print(classification_report(y_test, y_pred))

fi = pd.DataFrame([rf.feature_importances_],columns=features)


# copying train dataset
train_ftr2 = train.copy()

X = train_ftr2.drop(['target','feature7', 'feature8'], axis=1)
y = train_ftr2['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RNDM, stratify=y)

features = X.columns


"""**Model: Random Forest**"""

rf = RandomForestClassifier(random_state = RNDM)
rf.fit(X_train, y_train)

rf.score(X_train, y_train)

rf.score(X_test, y_test)

y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))

fi = pd.DataFrame([rf.feature_importances_],columns=features)

"""# Model Optimization"""

def scoring_f1(y_test,X_test,method): # generation of f1 macro
    f1score=f1_score(y_test,
                     method.predict(X_test),
                     average= 'macro'
)        
    return f1score

# Commented out IPython magic to ensure Python compatibility.
# %timeit
param_test1 = {'n_estimators':[2, 8, 32, 128, 256, 512],
               'max_features': ["sqrt", "log2", 0.2, 1, 3, 5],
               'max_leaf_nodes': [2**i for i in range(1, 8)]
              }
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(random_state = RNDM),
                        param_grid = param_test1, 
                        n_jobs = 4, 
                        cv = 5,
                        verbose = 4,
)

gsearch1.fit(X_train,y_train)

gsearch1.best_params_, gsearch1.best_score_

scoring_f1(y_test,X_test,gsearch1)

param_test2 = {'n_estimators':[256], 
               'max_features': [3],
               'max_leaf_nodes': [128],
               'criterion': ['gini', 'entropy'],
               'max_depth': [2, 4, 8, 16, 32, None]
               }
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(random_state = RNDM),
                        param_grid = param_test2, 
                        n_jobs = 4, 
                        cv = 5,
                        verbose = 4,
)

gsearch2.fit(X_train,y_train)

gsearch2.best_params_, gsearch2.best_score_

scoring_f1(y_test,X_test,gsearch2)

param_test3 = {'n_estimators':[256], 
               'max_features': [3],
               'max_leaf_nodes': [128],
               'criterion': ['gini'],
               'max_depth': [16],
               'bootstrap': [True, False],
               'max_features': ["sqrt", "log2", 1, 2, 3, 4, 5, 6]
               }
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(random_state = RNDM),
                        param_grid = param_test3, 
                        n_jobs = 4, 
                        cv = 5,
                        verbose = 4,
)

gsearch3.fit(X_train,y_train)

gsearch3.best_params_, gsearch3.best_score_

scoring_f1(y_test,X_test,gsearch3)

y_pred = gsearch3.predict(X_test)

cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, linewidths=1, annot=True, fmt='g',cmap="BuPu")

print(classification_report(y_test, y_pred))

"""# Prediction"""

test = test.drop(['feature7', 'feature8'], axis=1)

test.head()

prediction = gsearch3.predict(test)

prediction

df_prediction = pd.DataFrame(prediction, columns=["final_status"])
df_prediction.head()

#export csv
df_prediction.to_csv(r'predictions.csv', index = False)

#export json
df_prediction = pd.DataFrame(prediction, columns=['target'])
json_prediction = df_prediction.to_json()
with open('predictions.json', 'w') as outfile:
    outfile.write(json_prediction)

print('done!')
