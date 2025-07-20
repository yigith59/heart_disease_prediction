import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt



import warnings
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
# eda.kategorik_ozet(df, kategorik, plot=False)
# eda.sayisal_ozet(df, sayisal, False)

# düzenlemeler
# thal slope ve noof çıkarılacak
# resting bp 0 olamaz 0  - kolesterolde fazla eksik var knn imputasyonu


df["chestpain"] = df["chestpain"] +1 # diğer dataframe ile uyumlu olması için
df = df[[i for i in df.columns if i not in ["noofmajorvessels","thal","slope"]]]
df["serumcholestrol"] = df["serumcholestrol"].apply(lambda x: np.nan if x == 0 else x)
# knn imputasyonu yapılacak





# outlier kontrol

for i in sayisal:
    eda.outlier_kontrol(df, i,0.1,0.9)

# outlier yok


# eksik veriye bakis

eda.eksik_veri_kontrol(df)
# kolesterol için knn imputasyonu uygulayacağım

eda.knn_imputasyonu(df,["patientid", "target"])
eda.eksik_veri_kontrol(df)

# eksik veri tamam




# feature eng

df['agegrup'] = pd.cut(df['age'], bins=[20,40,50,65,81],
                       labels=["Yetiskin","Orta_yas","Genc_yaslı","Orta_yasli"], right=False)


df["gender_c"] = df["gender"].apply(lambda x: "erkek" if x == 1 else "kadin")


df["age_gender"] = df["agegrup"].astype(str) +"_" +df["gender_c"].astype(str)
df = df.drop(["agegrup","gender_c"],axis=1) # yeni değişken ürettiğim için bunları drop ettim




df["komorbite"] = df["serumcholestrol"].apply(lambda x: 1 if x>240 else 0) + df["fastingbloodsugar"]

df["rest"] = df["restingrelectro"] + df["restingBP"].apply(lambda x: 1 if x>140 else 0)



def maxhr_siniflama(df):
    # temp_df return ettiği için df ye atanmalı

    temp_df = df.copy()
    max_hr = 220 - temp_df["age"]
    temp_df["max_hr"] = 4

    temp_df.loc[temp_df["maxheartrate"] < (max_hr * 50/100), 'max_hr'] = 3
    temp_df.loc[(temp_df["maxheartrate"] >= (max_hr * 50/100)) & (temp_df["maxheartrate"] < (max_hr * 70/100)), 'max_hr'] = 2
    temp_df.loc[(temp_df["maxheartrate"] >= (max_hr * 70/100)) & (temp_df["maxheartrate"] < (max_hr * 85/100)), 'max_hr'] = 1
    temp_df.loc[temp_df["maxheartrate"] >= (max_hr * 85/100), 'max_hr'] = 0

    return temp_df

maxhr_siniflama(df)["max_hr"].value_counts() # kontrol amaçlı çalışıyor mu diye
df = maxhr_siniflama(df)

df['egzersiz_toleransi_skoru'] = (df["exerciseangia"].apply(lambda x: 1 if x >0.5 else 0) + df["max_hr"] + np.abs(df["oldpeak"]).astype(int))



# encoding  age_gender için
df = eda.one_hot_encoder(df,"age_gender",drop=True)



# korelasyon analizi
kategorik, sayisal, sayisal_kategorik, kategorik_kardinal,tarih = eda.degisken_siniflama(df)
conf_matrix = eda.korelasyon(df,df.columns,gorsel=True)
df = df.drop("rest",axis=1) # yüksek korelasyon nedenli drop
df = df.drop("max_hr",axis=1) # yüksek korelasyon nedenli drop


# veri hazırlama
df = df[[i for i in df.columns if i not in ["patientid"]]]
df.to_csv("df_india",index=False)



