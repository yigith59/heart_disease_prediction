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



column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
# dataframelerde var olan kolon isimleri


column_names_new = [
    'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol', 'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
    'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels', 'thal', 'target'
]
# india dataframinde bulunan kolon isimleri
# ortak kolon isimleri yapmak istiyorum ki sorun çıkmasın




clevland_data = pd.read_csv("datasets/processed.cleveland.data", names=column_names_new, na_values='?')
switzerland_data = pd.read_csv("datasets/processed.switzerland.data", names=column_names_new, na_values='?')
va_data = pd.read_csv("datasets/processed.va.data", names=column_names_new, na_values='?')
hungarian_data = pd.read_csv("datasets/processed.hungarian.data", names=column_names_new, na_values='?')

datasets = [clevland_data,switzerland_data,va_data,hungarian_data]


df_ = pd.concat([clevland_data,switzerland_data,va_data,hungarian_data],ignore_index=True)
df = df_.copy()
# datasetleri birleştirme işlemi


eda.ilkbakis(df)
kategorik, sayisal, sayisal_kategorik, kategorik_kardinal,tarih = eda.degisken_siniflama(df)

for i in kategorik:
    eda.kategorik_ozet(df, i, True)

for i in sayisal:
    eda.sayisal_ozet(df,i,True)


# düzenlemeler
# thal slope ve noof çıkarılacak
# target değişecek 2 - 3 -4 ü de 1 alacağım
# resting bp 0 olamaz  - kolesterolde fazla eksik var knn imputasyonu


df["oldpeak"] = np.abs(df["oldpeak"])
df = df[[i for i in df.columns if i not in ["noofmajorvessels","thal","slope"]]] # aşırı eksik verileri olduğu için drop
df["target"] = df["target"].apply(lambda x: 0 if x==0 else 1) # 2-3-4 kategorilerini de 1 yaptım
df = df[~(df["restingBP"]==0)] # resting bp 0 olamaz
df["serumcholestrol"] = df["serumcholestrol"].apply(lambda x: np.nan if x == 0 else x)
# knn imputasyonu için hazırladım





# outlier kontrol

for i in sayisal:
    eda.outlier_kontrol(df,i,0.01,0.99)

# outlier yok



# eksik veriye bakis
eda.eksik_veri_kontrol(df)
df = df[~(df["restingrelectro"].isnull())] # restingrelectro daki eksik veri 2 adet=> drop


# eksik diğer veriler için knn imputasyonu uygulayacağım
eda.knn_imputasyonu(df,["target"])


# eksik veri tamam
# knn imputasyonu sonucu fbs 0 ve 1 den farklı değerler var düzeltilecek



# feature eng

df['agegrup'] = pd.cut(df['age'], bins=[27,37,50,65,78],
                       labels=["Yetiskin","Orta_yas","Genc_yaslı","Orta_yasli"], right=False)


df["gender_c"] = df["gender"].apply(lambda x: "erkek" if x == 1 else "kadin")


df["age_gender"] = df["agegrup"].astype(str) +"_" +df["gender_c"].astype(str)
df = df.drop(["agegrup","gender_c"],axis=1) # yeni değişken ürettiğim için bunları drop ettim


df["fastingbloodsugar"] = df["fastingbloodsugar"].apply(lambda x: 1 if x >0.5 else 0) # üstte yazdığım problemin çözümü

df["komorbite"] = df["serumcholestrol"].apply(lambda x: 1 if x>240 else 0) + df["fastingbloodsugar"]

df["rest"] = df["restingrelectro"] + df["restingBP"].apply(lambda x: 1 if x>140 else 0)



def maxhr_siniflama(df):
    # temp_df return ettiği için df ye atanmalı

    temp_df = df.copy()
    max_hr = 220 - temp_df["age"]
    temp_df["max_hr"] = 4

    temp_df.loc[temp_df["maxheartrate"] < (max_hr * 50 / 100), 'max_hr'] = 3
    temp_df.loc[(temp_df["maxheartrate"] >= (max_hr * 50 / 100)) & (
                temp_df["maxheartrate"] < (max_hr * 70 / 100)), 'max_hr'] = 2
    temp_df.loc[(temp_df["maxheartrate"] >= (max_hr * 70 / 100)) & (
                temp_df["maxheartrate"] < (max_hr * 85 / 100)), 'max_hr'] = 1
    temp_df.loc[temp_df["maxheartrate"] >= (max_hr * 85 / 100), 'max_hr'] = 0

    return temp_df


maxhr_siniflama(df)["max_hr"].value_counts()  # kontrol amaçlı çalışıyor mu diye
df = maxhr_siniflama(df)

df['egzersiz_toleransi_skoru'] = (df["exerciseangia"].apply(lambda x: 1 if x > 0.5 else 0) + df["max_hr"] + np.abs(df["oldpeak"]).astype(int))





# encoding age_gender için
df = eda.one_hot_encoder(df,"age_gender",drop=True)




# korelasyon analizi
kategorik, sayisal, sayisal_kategorik, kategorik_kardinal,tarih = eda.degisken_siniflama(df)
conf_matrix = eda.korelasyon(df,df.columns,gorsel=True)
df = df.drop("rest",axis=1) # yüksek korelasyon nedenli drop
df = df.drop("max_hr",axis=1) # yüksek korelasyon nedenli drop


# veri hazırlama
df.to_csv("df_diger",index=False)








