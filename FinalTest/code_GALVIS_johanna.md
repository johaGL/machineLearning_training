EXAM johanna GALVIS RODRIGUEZ


```python
import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn import model_selection
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
import time
```


```python
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import OneHotEncoder
```


```python
datatrain = pd.read_csv("Train.csv", sep=';', header=0, index_col=0, decimal=",")
datatrain.head(2) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LB</th>
      <th>AC</th>
      <th>FM</th>
      <th>UC</th>
      <th>ASTV</th>
      <th>MSTV</th>
      <th>ALTV</th>
      <th>MLTV</th>
      <th>DL</th>
      <th>DS</th>
      <th>...</th>
      <th>Min</th>
      <th>Max</th>
      <th>Nmax</th>
      <th>Nzeros</th>
      <th>Mode</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Variance</th>
      <th>Tendency</th>
      <th>Classe</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>73</td>
      <td>0.5</td>
      <td>43</td>
      <td>2.4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>62</td>
      <td>126</td>
      <td>2</td>
      <td>0</td>
      <td>120</td>
      <td>137</td>
      <td>121</td>
      <td>73</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>132</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>17</td>
      <td>2.1</td>
      <td>0</td>
      <td>10.4</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>68</td>
      <td>198</td>
      <td>6</td>
      <td>1</td>
      <td>141</td>
      <td>136</td>
      <td>140</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 23 columns</p>
</div>




```python
datatest = pd.read_csv("Test.csv", sep=';', header=0, index_col=0, decimal=',')
datatest.head(2) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LB</th>
      <th>AC</th>
      <th>FM</th>
      <th>UC</th>
      <th>ASTV</th>
      <th>MSTV</th>
      <th>ALTV</th>
      <th>MLTV</th>
      <th>DL</th>
      <th>DS</th>
      <th>...</th>
      <th>Width</th>
      <th>Min</th>
      <th>Max</th>
      <th>Nmax</th>
      <th>Nzeros</th>
      <th>Mode</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Variance</th>
      <th>Tendency</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1639</th>
      <td>130</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>25</td>
      <td>1.7</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>94</td>
      <td>76</td>
      <td>170</td>
      <td>6</td>
      <td>0</td>
      <td>133</td>
      <td>119</td>
      <td>121</td>
      <td>73</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1640</th>
      <td>133</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>25</td>
      <td>1.7</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>94</td>
      <td>76</td>
      <td>170</td>
      <td>6</td>
      <td>0</td>
      <td>133</td>
      <td>117</td>
      <td>117</td>
      <td>77</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 22 columns</p>
</div>




```python
datatrain.shape
```




    (1638, 23)




```python
preX = datatrain.values
preX = preX.astype(float)
X_train = np.copy(preX[:,0:22]) 

```


```python
Y_train = np.copy(preX[:,22:23]) 
```


```python
# format target variable:
Y_train[Y_train == 1] = 0
Y_train[Y_train == 2] = 1
```


```python
X_test = datatest.values
X_test = X_test.astype(float)
X_test.shape, X_train.shape
```




    ((488, 22), (1638, 22))




```python
scaler = StandardScaler()
Xnorm_train = scaler.fit_transform(X_train)
```


```python
Xnorm_test = scaler.transform(X_test)
```


```python
# EFFICIENT PARAMETER TUNING :
parRF ={
    'n_estimators' : [50,70]
}
grid_RF = model_selection.GridSearchCV(RandomForestClassifier(), parRF, n_jobs=4, scoring='accuracy')

parknn = {  
   'n_neighbors': [5,7,11,13,15],
    'weights' : ['uniform', 'distance']
} 

grid_knn = model_selection.GridSearchCV(KNeighborsClassifier(), parknn, cv=5,
                                      scoring='accuracy')

parmlp = {
    'hidden_layer_sizes': [(40,20), (45,23) , (50,30)],
    'activation' : ['tanh', 'relu'],
    'alpha' : [1e-3, 1e-4, 1e-5],
    'solver' : ['lbfgs', 'sgd', 'adam'],
    'max_iter' : [100,200]
}
grid_mlp = model_selection.GridSearchCV(MLPClassifier(), parmlp, cv=5, n_jobs=4,
                             scoring='accuracy') 

parbag ={
    'n_estimators' : [50,70]
}
grid_bag = model_selection.GridSearchCV(BaggingClassifier(), parbag, n_jobs=4, scoring='accuracy')
```


```python
print("searching best parameters(gridSearcCV)")

print("RandomForestClassifier")
grid_RF.fit(Xnorm_train, Y_train)
print(grid_RF.best_params_)
print(grid_RF.best_score_)

print("KNN")
grid_knn.fit(Xnorm_train, Y_train)
print(grid_knn.best_params_)
print(grid_knn.best_score_)

print("MLP")
grid_mlp.fit(Xnorm_train, Y_train)
print(grid_mlp.best_params_)
print(grid_mlp.best_score_)


```

    searching best parameters(gridSearcCV)
    RandomForestClassifier
    {'n_estimators': 50}
    0.9316252703811443
    KNN
    {'n_neighbors': 7, 'weights': 'uniform'}
    0.8736257179085551
    MLP
    {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (45, 23), 'max_iter': 200, 'solver': 'adam'}
    0.9243063325128664



```python
print("bagging")
grid_bag.fit(Xnorm_train, Y_train)
print(grid_bag.best_params_)
print(grid_bag.best_score_)

```

    bagging
    {'n_estimators': 70}
    0.9199951517863804



```python

```


```python
# dans mes résultats, MLP meme avec paramètres optimisés n'etait pas en tête 
# pour les accuracy (et prennais enormement de temps )
# je teste plusieurs méthodes sauf MLP:
```


```python
# je mets à jour les params des algos que j'ai pu optimiser (pas le temps pour optim tous avec grid pour cet exam)
clfs = {
    'RF': RandomForestClassifier(n_estimators=70, random_state=1),
    'KNN2' : KNeighborsClassifier(n_neighbors=7, weights='uniform'),
    'ADA' : AdaBoostClassifier(n_estimators=50, random_state=1),
    'BAG' : BaggingClassifier(n_estimators=50) ,
    'NB' : GaussianNB(),
    'CART' : DecisionTreeClassifier(criterion='gini'),
    'ID3' : DecisionTreeClassifier(criterion='entropy'),
    'ST' : DecisionTreeClassifier(max_depth=1) #decisionStump
} 
```


```python
## applying KFold method to do re-sampling (cross validation) when running
def run_classifiers(clfs, X, Y):
    dico = {'classifier':[],'accuracy_mean':[],'accuracy_sd':[], 
             'precision_mean':[], 'precision_sd':[], 'recall_mean' : [], 'recall_sd' : [],
            'AUC':[], 'time_s':[]} #output into dictionnary
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for clf_id in clfs:
        initime = time.time()
        clf = clfs[clf_id]
        cvAccur = cross_val_score(clf, X, Y, cv=kf, scoring='accuracy')
        end = time.time()
        cvPrecision = cross_val_score(clf, X, Y, cv=kf, scoring='precision')
        cvRecall = cross_val_score(clf, X, Y, cv=kf, scoring='recall')
        cvAUC = cross_val_score(clf, X, Y, cv=kf, scoring='roc_auc')
        dico['classifier'].append(clf_id)
        dico['accuracy_mean'].append(round(np.mean(cvAccur),3))
        dico['accuracy_sd'].append(round(np.std(cvAccur),3))
        dico['precision_mean'].append(round(np.mean(cvPrecision),3))
        dico['precision_sd'].append(round(np.std(cvPrecision),3))
        dico['recall_mean'].append(round(np.mean(cvRecall),3))
        dico['recall_sd'].append(round(np.std(cvRecall),3))
        dico['AUC'].append(round(np.mean(cvAUC),3))
        dico['time_s'].append(round((end-initime),3))
    return dico
```


```python
dd = run_classifiers(clfs, X_train, Y_train)
pd.DataFrame.from_dict(dd)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>classifier</th>
      <th>accuracy_mean</th>
      <th>accuracy_sd</th>
      <th>precision_mean</th>
      <th>precision_sd</th>
      <th>recall_mean</th>
      <th>recall_sd</th>
      <th>AUC</th>
      <th>time_s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RF</td>
      <td>0.954</td>
      <td>0.014</td>
      <td>0.947</td>
      <td>0.009</td>
      <td>0.880</td>
      <td>0.050</td>
      <td>0.989</td>
      <td>0.896</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN2</td>
      <td>0.916</td>
      <td>0.009</td>
      <td>0.905</td>
      <td>0.013</td>
      <td>0.770</td>
      <td>0.029</td>
      <td>0.956</td>
      <td>0.117</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ADA</td>
      <td>0.952</td>
      <td>0.009</td>
      <td>0.924</td>
      <td>0.016</td>
      <td>0.900</td>
      <td>0.034</td>
      <td>0.987</td>
      <td>0.705</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BAG</td>
      <td>0.957</td>
      <td>0.015</td>
      <td>0.959</td>
      <td>0.013</td>
      <td>0.891</td>
      <td>0.036</td>
      <td>0.986</td>
      <td>1.727</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NB</td>
      <td>0.897</td>
      <td>0.024</td>
      <td>0.826</td>
      <td>0.035</td>
      <td>0.784</td>
      <td>0.095</td>
      <td>0.957</td>
      <td>0.011</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CART</td>
      <td>0.946</td>
      <td>0.016</td>
      <td>0.896</td>
      <td>0.011</td>
      <td>0.897</td>
      <td>0.042</td>
      <td>0.931</td>
      <td>0.049</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ID3</td>
      <td>0.935</td>
      <td>0.015</td>
      <td>0.876</td>
      <td>0.048</td>
      <td>0.883</td>
      <td>0.044</td>
      <td>0.922</td>
      <td>0.037</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ST</td>
      <td>0.926</td>
      <td>0.007</td>
      <td>0.947</td>
      <td>0.017</td>
      <td>0.768</td>
      <td>0.027</td>
      <td>0.876</td>
      <td>0.012</td>
    </tr>
  </tbody>
</table>
</div>




```python
# le meilleur étant le bagging, je le choisis pour la prédiction
```


```python
# HERE I DO PREDICTION
meth = BaggingClassifier(n_estimators=50, random_state=1)
meth.fit(Xnorm_train, Y_train)
predicted = meth.predict(Xnorm_test)
```


```python
len(predicted)
```




    488




```python
individus = datatest.index.values
```


```python
individus = datatest.index.values
dfout = pd.DataFrame(predicted, index=individus, columns=["predicted"])
dfout
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1639</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1640</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1641</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1642</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1643</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2122</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2123</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2124</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2125</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2126</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>488 rows × 1 columns</p>
</div>




```python
dfout.to_csv("prediction_galvis_johanna.csv", sep=",")
```


```python

```


```python

```
