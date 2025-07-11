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


df_ = pd.read_csv("datasets/Cardiovascular_Disease_Dataset.csv")
df = df_.copy()



eda.ilkbakis(df)
kategorik, sayisal, sayisal_kategorik, kategorik_kardinal, tarih = eda.degisken_siniflama(df)
eda.kategorik_ozet(df, kategorik, plot=True)
eda.sayisal_ozet(df, sayisal, True)

# düzenlemeler
df["chestpain"] = df["chestpain"] +1
df = df[[i for i in df.columns if i not in ["noofmajorvessels","thal","slope"]]]




# kolesterol 0 olamaz
df[df["serumcholestrol"]==0]
df[df["serumcholestrol"]>0]
df["serumcholestrol"] = df["serumcholestrol"].apply(lambda x: np.nan if x == 0 else x)
# knn imputasyonu yapılacak





# outlier kontrol

for i in sayisal:
    eda.outlier_kontrol(df, i,0.1,0.9)

# outlier yok


# eksik veri

eda.eksik_veri_kontrol(df)



# knn imputasyonu uygulayacağım




temp_df = df.copy()
temp_df = temp_df[[col for col in temp_df.columns if col not in ["patientid", "target"]]]





temp_df.info()
eksik_col = ["serumcholestrol"]

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



temp_df["serumcholestrol"] = df_imputed_temp["serumcholestrol"]




eda.ilkbakis(temp_df)
eda.ilkbakis(df)



df["serumcholestrol"] = temp_df["serumcholestrol"]



eda.eksik_veri_kontrol(df)



# eksik veri tamam

eda.ilkbakis(df)




# feature eng 


df['agegrup'] = pd.cut(df['age'], bins=[20,40,50,65,81],
                       labels=["Yetiskin","Orta_yas","Genc_yaslı","Orta_yasli"], right=False)


df["gender_c"] = df["gender"].apply(lambda x: "erkek" if x == 1 else "kadin")


df["age_gender"] = df["agegrup"].astype(str) +"_" +df["gender_c"].astype(str)
df["age_gender"].value_counts()
df = df.drop(["agegrup","gender_c"],axis=1)

df["serumcholestrol"].describe()


df["komorbite"] = df["serumcholestrol"].apply(lambda x: 1 if x>240 else 0) + df["fastingbloodsugar"]
df["komorbite"].value_counts()

df["rest"] = df["restingrelectro"] + df["restingBP"].apply(lambda x: 1 if x>140 else 0)
df["rest"].value_counts()




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
df['egzersiz_toleransi_skoru'].value_counts()
np.abs(df["oldpeak"]).astype(int).value_counts()

df = df.drop("max_hr",axis=1)



# encoding age gender için
df = eda.one_hot_encoder(df,"age_gender")
df = df.drop("age_gender",axis=1)





kategorik, sayisal, sayisal_kategorik, kategorik_kardinal, tarih = eda.degisken_siniflama(df)




# veri hazırlama
df = df[[i for i in df.columns if i not in ["patientid"]]]
df.to_csv("df_india",index=False)



