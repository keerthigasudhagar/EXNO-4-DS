# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![322404729-703e88ee-f8d1-4769-b107-cd331eff38ca](https://github.com/user-attachments/assets/d898c9f3-f583-469d-85f1-183b2dca54f0)
```
data.isnull().sum()
```
![322404935-5ffc9c47-d0af-4b1c-a6ce-6042049c3549](https://github.com/user-attachments/assets/de96bdcc-fecc-4e14-8a30-ac517a009cd4)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![322405081-e4265e5e-1d00-402e-b9b5-54969f7d4bd6](https://github.com/user-attachments/assets/79b0aed9-9022-4cfa-8511-408cf9cdd819)
```
data2=data.dropna(axis=0)
data2
```
![322405250-e6c40d14-6070-4d57-825d-497937077dd6](https://github.com/user-attachments/assets/5b0923d8-b569-4c39-93d7-350621edf497)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![322405385-29fdf9fb-2aa4-4501-9e5b-a09e87a7992f](https://github.com/user-attachments/assets/adeb740e-baa0-4123-a66c-29977b1a0d63)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![322405503-d694ef71-82e8-437a-9380-291d9f655b4c](https://github.com/user-attachments/assets/48b8b393-928e-4953-bebb-5e5a233d8cfb)
```
data2
```
![322405649-77cdc673-365b-4d88-9ba3-aef0772ad4d1](https://github.com/user-attachments/assets/d01a06b2-f246-4f7a-80ac-b08bf7787f78)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![322405827-40fee06a-0d12-474c-92a8-a5d8d72ec8f8](https://github.com/user-attachments/assets/40766783-9b42-417b-923d-0393b028bd73)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![322406020-e3e6dc73-e973-4a93-82b4-0957d392bde3](https://github.com/user-attachments/assets/6141eafd-5de0-4c98-ab14-8fe4bf85330e)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![322406227-c71d0b2e-467a-45e0-989e-c03d5f448cb0](https://github.com/user-attachments/assets/b3b21eca-2056-468d-b12b-b1ade5acfc68)
```
y=new_data['SalStat'].values
print(y)
```
![322406491-69a45f59-0c62-4a1d-bab4-01fdbfbd5d27](https://github.com/user-attachments/assets/57889398-1807-41c8-8cdf-b06cc1b942bc)
```
x=new_data[features].values
print(x)
```
![322406665-45fcc088-5c3c-4952-b02d-1bc816fa242f](https://github.com/user-attachments/assets/a34f6cd1-acda-456f-b03d-0a8bf1236ce9)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![322406826-5d8fe3fb-1384-4b1b-bc7e-d68a36d88f6d](https://github.com/user-attachments/assets/834f606f-9dfb-4a54-a0c4-f4484cf08211)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![322407087-5a9670ed-8dd6-47c2-b518-e2d6a3026204](https://github.com/user-attachments/assets/658dfcac-4b3c-4998-b0cf-97f06bfa46fb)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![322407231-430a0c4b-cdce-4384-b92b-563d78757040](https://github.com/user-attachments/assets/a7598ffc-1a1c-4753-94bf-edce8db26295)

```
data.shape
```
![322407397-faf67c38-239f-43fc-aa33-c69a9907df3c](https://github.com/user-attachments/assets/072abf5a-069f-4443-b824-4b937934189e)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![322407695-0ef4a0da-64b6-4c01-ab36-bc52c5b50c02](https://github.com/user-attachments/assets/c855bb47-4f44-4f5a-840f-86e7d03378c7)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![322407803-cfc24046-f1b0-461c-a2b7-e3bbe92c3107](https://github.com/user-attachments/assets/35b0b3ea-200e-4bdb-8175-7bbf68deddf1)
```
tips.time.unique()
```
![322407990-65aad783-c72b-461e-9136-7db0d6609721](https://github.com/user-attachments/assets/c94fd1b0-3d26-4f56-95b9-0c5927d701c5)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![322408200-9fceef68-fffa-4fd3-a8e1-f627e06c0d39](https://github.com/user-attachments/assets/c6c66f5f-f85c-4ca3-89c5-bdbe8b611c24)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![322408290-1257bbd5-f921-47a9-b233-89744d3dea1e](https://github.com/user-attachments/assets/9833f33b-c8a5-4cf7-8e5a-8875544eeaf5)

# RESULT:
       Thus, Feature selection and Feature scaling has been used on thegiven dataset.
