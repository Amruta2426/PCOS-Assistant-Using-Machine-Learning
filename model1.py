import pandas as pd
import numpy as np
import matplotlib.pyplot  
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data_woinfer1 = pd.read_csv('PCOS_data_without_infertility.csv')
data_infer1 = pd.read_csv('PCOS_infertility.csv')

#Merging both data on the basis of Patient File No
data1 = pd.merge(data_woinfer1, data_infer1, on = 'Patient File No.', suffixes = {'','_without'},how = 'left')

#Dropping repeated columns
data1 = data1.drop(['Unnamed: 44', 'Sl. No_without', 'PCOS (Y/N)_without', '  I   beta-HCG(mIU/mL)_without','II    beta-HCG(mIU/mL)_without', 'AMH(ng/mL)_without'], axis=1)

print(data1.head())

# Filling Missing Values
data1['Marraige Status (Yrs)'].fillna(data1['Marraige Status (Yrs)'].median(),inplace = True)
data1['Fast food (Y/N)'].fillna(data1['Fast food (Y/N)'].median(),inplace = True)

data1.drop (['Sl. No','Patient File No.', 'Cycle(R/I)', '  I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)','AMH(ng/mL)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)','Endometrium (mm)'  ],axis=1,inplace=True)

print(data1.head())
    
x = data1.drop(['PCOS (Y/N)'],axis = 1)
Y = data1['PCOS (Y/N)']

# print(X.head())
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.2, random_state = 0) 

#feature scaling
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

random_forest1 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
random_forest1.fit(x_train, Y_train)

# Saving model to disk
pickle.dump(random_forest1, open('model1.pkl','wb'))
