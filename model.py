import pandas as pd
import numpy as np
import matplotlib.pyplot  
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# model for general result.
data_woinfer = pd.read_csv('PCOS_data_without_infertility.csv')
data_infer = pd.read_csv('PCOS_infertility.csv')

#Merging both data on the basis of Patient File No
data = pd.merge(data_woinfer, data_infer, on = 'Patient File No.', suffixes = {'','_without'},how = 'left')
#Dropping repeated columns
data = data.drop(['Unnamed: 44', 'Sl. No_without', 'PCOS (Y/N)_without', '  I   beta-HCG(mIU/mL)_without','II    beta-HCG(mIU/mL)_without', 'AMH(ng/mL)_without'], axis=1)

print(data.head())

# Filling Missing Values
data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].median(),inplace = True)
data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].median(),inplace = True)


data.drop(['Sl. No', 'Patient File No.', 'Pulse rate(bpm) ',
       'RR (breaths/min)', 'Hb(g/dl)', '  I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 'FSH(mIU/mL)',
       'LH(mIU/mL)', 'FSH/LH',
       'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)',
       'PRG(ng/mL)', 'RBS(mg/dl)', 'BP _Systolic (mmHg)',
       'BP _Diastolic (mmHg)', 'Follicle No. (L)', 'Follicle No. (R)',
       'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)'],axis=1,inplace=True)

print(data.head())
    
X = data.drop(['PCOS (Y/N)'],axis = 1)
y = data['PCOS (Y/N)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

#feature scaling
sc = StandardScaler()

random_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
random_forest.fit(X_train, y_train)

# Saving model to disk
pickle.dump(random_forest, open('model.pkl','wb'))
