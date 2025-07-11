import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.model_selection
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date

from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, \
    confusion_matrix, classification_report, balanced_accuracy_score, roc_curve, auc, silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate, GridSearchCV, \
    validation_curve
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.model_selection
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date

from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, \
    confusion_matrix, classification_report, balanced_accuracy_score, roc_curve, auc, silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate, GridSearchCV, \
    validation_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler, OneHotEncoder, \
    PolynomialFeatures, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz, export_text
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

from statsmodels.stats.proportion import proportions_ztest

# pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


# pip install catboost
# pip install xgboost
# pip install lightgbm

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


import re
import graphviz
import pydotplus, skompiler, astor, joblib, warnings
import EDA as eda


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 30)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)



column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]



column_names_new = [
    'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol', 'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
    'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels', 'thal', 'target'
]


clevland_data = pd.read_csv("datasets/processed.cleveland.data", names=column_names_new, na_values='?')
switzerland_data = pd.read_csv("datasets/processed.switzerland.data", names=column_names_new, na_values='?')
va_data = pd.read_csv("datasets/processed.va.data", names=column_names_new, na_values='?')
hungarian_data = pd.read_csv("datasets/processed.hungarian.data", names=column_names_new, na_values='?')

datasets = [clevland_data,switzerland_data,va_data,hungarian_data]

df_ = pd.concat([clevland_data,switzerland_data,va_data,hungarian_data],ignore_index=True)
df = df_.copy()



eda.ilkbakis(df)
kategorik, sayisal, sayisal_kategorik, kategorik_kardinal,tarih = eda.degisken_siniflama(df)
eda.kategorik_ozet(df,kategorik,True)
eda.sayisal_ozet(df,sayisal,True)


# düzenlemeler
df["oldpeak"] = np.abs(df["oldpeak"])


# thal ve noof çıkarılacak - slope incele ona göre karar 
# target değişecek 2 - 3 -4 ü de 1 alacağım 
# resting bp 0 olamaz  - kolesterol



df = df[[i for i in df.columns if i not in ["noofmajorvessels","thal","slope"]]]

df["target"] = df["target"].apply(lambda x: 0 if x==0 else 1)



df = df[~(df["restingBP"]==0)]

df[df["serumcholestrol"]==0] # çok fazla eksik var
df["serumcholestrol"] = df["serumcholestrol"].apply(lambda x: np.nan if x == 0 else x)



# outlier kontrol

for i in sayisal:
    eda.outlier_kontrol(df,i,0.01,0.99)

# outlier yok



# eksik veriye bakis
df = df[~(df["restingrelectro"].isnull())]

eda.eksik_veri_kontrol(df)

# knn imputasyonu uygulayacağım




temp_df = df.copy()
temp_df = temp_df[[col for col in temp_df.columns if col not in ["id", "target"]]]





temp_df.info()
eksik_col = ['restingBP', 'serumcholestrol', 'fastingbloodsugar', 'maxheartrate', 'exerciseangia', 'oldpeak']

ss_temp = StandardScaler()
temp_df[[i for i in temp_df.columns if i not in eksik_col]] = ss_temp.fit_transform(temp_df[[i for i in temp_df.columns if i not in eksik_col ]])


# knn ile tahmin
df_for_processing = temp_df.copy()




knn_imputer = KNNImputer(n_neighbors=5)


df_imputed_array = knn_imputer.fit_transform(df_for_processing)


df_imputed_temp = pd.DataFrame(df_imputed_array,
                               columns=df_for_processing.columns,
                               index=df_for_processing.index)



eda.ilkbakis(temp_df)
eda.ilkbakis(df_imputed_temp)


temp_df["restingBP"] = df_imputed_temp["restingBP"]
temp_df["serumcholestrol"] = df_imputed_temp["serumcholestrol"]
temp_df["fastingbloodsugar"] = df_imputed_temp["fastingbloodsugar"]
temp_df["maxheartrate"] = df_imputed_temp["maxheartrate"]
temp_df["exerciseangia"] = df_imputed_temp["exerciseangia"]
temp_df["oldpeak"] = df_imputed_temp["oldpeak"]



eda.ilkbakis(temp_df)
eda.ilkbakis(df)






df["restingBP"] = temp_df["restingBP"]
df["serumcholestrol"] = temp_df["serumcholestrol"]
df["fastingbloodsugar"] = temp_df["fastingbloodsugar"]
df["maxheartrate"] = temp_df["maxheartrate"]
df["exerciseangia"] = temp_df["exerciseangia"]
df["oldpeak"] = temp_df["oldpeak"]




eda.eksik_veri_kontrol(df)
# eksik veri tamam



# feature eng 


eda.sayisal_ozet(df,"age",True)


df['agegrup'] = pd.cut(df['age'], bins=[27,37,50,65,78],
                       labels=["Yetiskin","Orta_yas","Genc_yaslı","Orta_yasli"], right=False)


df["gender_c"] = df["gender"].apply(lambda x: "erkek" if x == 1 else "kadin")


df["age_gender"] = df["agegrup"].astype(str) +"_" +df["gender_c"].astype(str)
df["age_gender"].value_counts()
df = df.drop(["agegrup","gender_c"],axis=1)

df["serumcholestrol"].describe()
df["fastingbloodsugar"] = df["fastingbloodsugar"].apply(lambda x: 1 if x >0.5 else 0)

df["komorbite"] = df["serumcholestrol"].apply(lambda x: 1 if x>240 else 0) + df["fastingbloodsugar"]
df["komorbite"].value_counts()

df["rest"] = df["restingrelectro"] + df["restingBP"].apply(lambda x: 1 if x>140 else 0)





max_hr = 220 - df["age"]


df["max_hr"] = 'Tanımsız'

df.loc[df["maxheartrate"] < (max_hr * 50/100), 'max_hr'] = 'düşük'
df.loc[(df["maxheartrate"] >= (max_hr * 50/100)) & (df["maxheartrate"] < (max_hr * 70/100)), 'max_hr'] = 'orta'
df.loc[(df["maxheartrate"] >= (max_hr * 70/100)) & (df["maxheartrate"] < (max_hr * 85/100)), 'max_hr'] = 'iyi'
df.loc[df["maxheartrate"] >= (max_hr * 85/100), 'max_hr'] = 'ideal'




def maxhr(x):
    if x == 'düşük':
        return 3
    elif x == "orta":
        return 2
    elif x=="iyi":
        return 1
    else:
        return 0

df["max_hr"] = df["max_hr"].apply(maxhr)


df['egzersiz_toleransi_skoru'] = (df["exerciseangia"].apply(lambda x: 1 if x >0.5 else 0) + df["max_hr"] + np.abs(df["oldpeak"]).astype(int))

df = df.drop("max_hr",axis=1)



# encoding age gender için
df = eda.one_hot_encoder(df,"age_gender")
df = df.drop("age_gender",axis=1)



# veri hazırlama
df.to_csv("df_diger",index=False)













