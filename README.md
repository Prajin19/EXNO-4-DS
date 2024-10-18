# EXNO:4-To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file.
# Name: Prajin S
# Reference Number: 212223230151
# Department: Artificial Intelligence & Data Science
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

data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/aac9e8eb-5bd7-425f-b6fc-5de729b59a82)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/c5109b28-c5ad-4ee3-a4da-367fdf0112bf)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/2af2d44f-a7f4-4cf2-aa9b-b5c0fd0c0a09)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/8bbbe8b4-ff1e-4836-8bc2-8e127486b636)
```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/21678215-c421-4e8a-b3ef-a22f104c20ed)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/e332d54a-439a-4db2-93d0-39650d75edeb)
```
data2
```
![image](https://github.com/user-attachments/assets/0315e0f7-90e9-421b-aa1e-4f0a60aae06e)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/3f8ad2a2-58e1-4435-ad5b-a08622d25e8e)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/2c30275e-0c54-4963-820d-e973b6706cb0)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/e5acf80e-900c-4134-859c-b451ed664e71)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/d5843d82-c083-40d9-ac4b-5319d6471d3f)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/754f3ac9-abb7-407b-8924-4a42ee539d59)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/e32e0565-3d48-4821-a6cf-598dd1357618)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/945f59ef-ee9c-4c4e-a71f-c7dc5eb00e91)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/90db07ca-c3dd-47b4-975a-048fc43b01be)
```
data.shape
```
![image](https://github.com/user-attachments/assets/adfb8a6a-d3cd-48f2-8ac1-951b765a66ee)
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
![image](https://github.com/user-attachments/assets/e3bd05af-6d42-4327-baba-f781c4644350)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/b62c79e8-1eff-4da6-82ba-d3da068da33d)
```
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/dd92997b-fb0e-4f9e-8ea3-ca34d15134a3)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/610b838f-85d2-4115-94ec-5a06c11fbfbd)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/54fd3dfd-5163-4791-a6f4-5bdefef0a62e)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
