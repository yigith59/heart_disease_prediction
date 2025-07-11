import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler,OneHotEncoder
from statsmodels.stats.proportion import proportions_ztest
import re


pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)


def ilkbakis(df):
    print("########### "+"shape"+" ###########", end="\n\n")
    print(df.shape, end="\n\n")
    print("########### "+"info"+" ###########", end="\n\n")
    print(df.info(), end="\n\n")
    print("########### "+"describe"+" ###########", end="\n\n")
    print(df.describe().T,  end="\n\n")
    print("########### "+"isnull"+" ###########", end="\n\n")
    print(df.isnull().values.any(), end="\n\n")
    print("########### "+"isnull"+" ###########", end="\n\n")
    print(df.isnull().sum())



def degisken_siniflama(df,cat_th=10,car_th=50):

    sayisal = list(df.select_dtypes(include=["number"]).columns)
    kategorik = list(df.select_dtypes(include=["category", "object", "bool"]).columns)

    tarih = list(df.select_dtypes(include=["datetime", "timedelta"]).columns)

    nunique_dict = df.nunique().to_dict()

    sayisal_kategorik = [i for i in sayisal if nunique_dict[i] < cat_th]
    kategorik_kardinal = [i for i in kategorik if nunique_dict[i] > car_th]

    kategorik = [i for i in df.columns if ((i in kategorik) or (i in sayisal_kategorik)) and (i not in kategorik_kardinal)]
    sayisal = [i for i in df.columns if (i in sayisal) and (i not in sayisal_kategorik)]

    print(f"kategorik = {len(kategorik)} değişken\n{kategorik}")
    print(f"sayisal = {len(sayisal)} değişken\n{sayisal}")
    print(f"numerik_kategorik = {len(sayisal_kategorik)} değişken\n{sayisal_kategorik}")
    print(f"kategorik_kardinal = {len(kategorik_kardinal)} değişken\n{kategorik_kardinal}")
    print(f"tarih = {len(tarih)} değişken\n{tarih}")

    return kategorik, sayisal, sayisal_kategorik, kategorik_kardinal,tarih


# kategorik, sayisal, sayisal_kategorik, kategorik_kardinal,tarih


def kategorik_ozet(df,cols,plot=False):

    # tekli veya çoklu kullanım

    if isinstance(cols, str):
        cols = [cols]


    for col in cols:
        if plot:

            print("######################", end="\n\n")
            print(pd.DataFrame({"Oran": df[col].value_counts(normalize=True,dropna=False)*100,
                                "Sayıları": df[col].value_counts(dropna=False)}), end="\n\n\n")

            if df[col].dtype == bool:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))  

                sns.countplot(x=df[col].astype(int), ax=axes[0],palette="muted",hue=df[col],legend=False) 
                axes[0].set_title(f"{col} - Countplot")

                df[col].value_counts().plot.pie(autopct="%1.1f%%", colormap="Set3",
                                                ax=axes[1]) 
                axes[1].set_title(f"{col} - Pie Chart")

                plt.tight_layout()
                plt.show(block=True)

            else:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 

                sns.countplot(x=df[col], ax=axes[0],palette="muted",hue=df[col],legend=False)  
                axes[0].set_title(f"{col} - Countplot")

                df[col].value_counts().plot.pie(autopct="%1.1f%%", colormap="Set3",
                                                ax=axes[1])  
                axes[1].set_title(f"{col} - Pie Chart")

                plt.tight_layout()
                plt.show(block=True)

        else:
            print("######################", end="\n\n")
            print(pd.DataFrame({"Oran": df[col].value_counts(normalize=True,dropna=False),
                                "Sayıları": df[col].value_counts(dropna=False)}).sort_index(), end="\n\n\n")


def sayisal_ozet(df,cols,plot=False):
    # tekli veya çoklu kullanım

    if isinstance(cols, str):
        cols = [cols]

    desc_df = df[cols].describe().T
    temp_df = df.copy()
    outlier_threshold_df = pd.DataFrame()

    q1_vals = temp_df[cols].quantile(0.25)
    q3_vals = temp_df[cols].quantile(0.75)
    iqr_vals = q3_vals - q1_vals
    # bu kısmı apply ile tek satırda yaz
    min_deger = q1_vals - 1.5 * iqr_vals
    max_deger = q3_vals + 1.5 * iqr_vals

    for col in cols:

        print(col, end="\n")
        print("q1: " + str(q1_vals))
        print("q3: " + str(q3_vals))
        print("iqr: " + str(iqr_vals))
        print("min_deger: " + str(min_deger))
        print("max_deger: " + str(max_deger), end="\n\n")

        deger_df = pd.DataFrame({"min_deger": min_deger,
                                 "max_deger": max_deger}, index=[col])
        outlier_threshold_df = pd.concat([outlier_threshold_df, deger_df])



        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            sns.boxplot(x=df[col], ax=ax[0])
            ax[0].set_title(f"{col} - Boxplot")

            sns.histplot(df[col], kde=True, ax=ax[1], color="purple", bins=30) 
            ax[1].set_title(f"{col} - Histogram + KDE")

            plt.show(block=True)

    print(desc_df, end="\n\n")
    print(outlier_threshold_df, end="\n\n")



    return outlier_threshold_df,desc_df



# ###############################



def threshold_hesaplama(df,col, a, b):
    temp_df = df.copy()
    outlier_threshold_df = pd.DataFrame()
    # outlier tespiti için min ve max değerleri hesaplar

    q1 = temp_df[col].quantile(a)
    q3 = temp_df[col].quantile(b)
    iqr = q3 - q1

    min_deger = q1 - 1.5 * iqr
    max_deger = q3 + 1.5 * iqr

    print(col, end="\n")
    print("q1: " + str(q1))
    print("q3: " + str(q3))
    print("iqr: " + str(iqr))
    print("min_deger: " + str(min_deger))
    print("max_deger: " + str(max_deger), end="\n\n")

    outlier_threshold_df = pd.DataFrame({col + "_min_deger": min_deger,
                                             col + "_max_deger": max_deger}, index=[col])
    print(outlier_threshold_df,end="\n\n")

    return min_deger,max_deger

def outlier_kontrol(df,col,a,b,index=False):
    # outlier var mı kontrol eder
    # tek kullanılmalı , liste kullanılacaksa for ile kullanılmalı
    # for ile kullanırken print(df[i].describe().T) de altına ilave edilebilir

    temp_df = df.copy()
    min_deger, max_deger = threshold_hesaplama(df,col,a,b)

    if (temp_df[col] > max_deger).any():
        print(str(col) + " Ust deger outlier var")
    else:
        print(str(col) + " Ust deger outlier yok")

    if (temp_df[col] < min_deger).any():
        print(str(col) + " Alt deger outlier var",end="\n\n")
    else:
        print(str(col) + " Alt deger outlier yok",end="\n\n")

    print("###############",end="\n\n")

    if index:
        return temp_df[(temp_df[col] > max_deger) | (temp_df[col] < min_deger)].index

    """else:
        return outlier_threshold_df"""

def outlier_cozum(df,col,index_del=False, secim=False, baskilama=False,a=0.25,b=0.75):
    # outlier değerlere çözüm
    # tek kullanılmalı , liste kullanılacaksa for ile kullanılmalı
    # çoklu kullanımda df = outliercozum diye eşitlenmeli yoksa kaydetmez

    # index_del = belirtilen outlier değerlerin indexlerini kullanarak temp_df den siler; yeni dataframe return edilir
    # secim = outlier değerlerın dışında kalanlar seçilir yeni df ye aktarılır
    # baskılama =  outlier değerler eşik değerleriyle değiştirir


    min_deger , max_deger = threshold_hesaplama(df,col,a,b)
    temp_df = df.copy()

    if index_del:
        temp_df.drop(temp_df[(temp_df[col] > max_deger) | (temp_df[col] < min_deger)].index, inplace=True) # default olarak indexten sildiği için .index
        # belirtilen outlier değerlerin indexlerini kullanarak df den siler


    if secim:
        temp_df = temp_df[~((temp_df[col] > max_deger) | (temp_df[col] < min_deger))] # outlier değerlerın dışında kalanlar seçilir yeni df ye aktarılır


    if baskilama: # outlier değerler eşik değerleriyle değiştirilebilir
        temp_df.loc[(temp_df[col] > max_deger), col] = max_deger
        temp_df.loc[(temp_df[col] < min_deger), col] = min_deger

    return temp_df

    """df.index.delete(df[(df["Age"]>ust_deger) | (df["Age"]<alt_deger)].index) # çıktı olarak outlier olmayanların indexlerini verir
       median,ortalama gibi şeylerle de değiştirilebilir"""



# ###############################



def eksik_veri_kontrol(df, index=False):
    print("Eksik değer: " + str(df.isnull().values.any()))  # eksik gözlem var mı yok mu
    print("####################")
    print("Toplam: " + str(df.isnull().values.sum())) # eksik gözlem toplamını verir
    print("####################")

    eksik_kolonlar = [i for i in df.columns if df[i].isnull().values.any() == True]
    # if sonrasına istenirse df[i].isnull().sum() > 0 da denebilir  # missing value si olan col ları seçer

    na_count = df[eksik_kolonlar].isnull().sum()  # toplamı
    yuzdelik = df[eksik_kolonlar].isnull().sum() * 100 / df.shape[0] # yüzdeliği
    na_df = pd.DataFrame({"Missing Values": na_count,
                          "Ratio (%)": yuzdelik})
    na_df = na_df.sort_values("Missing Values",ascending=False) # tablo halinde özet
    print(na_df)


    if index: # eksik verilerin olduğu satırların index bilgisi
        eksik_satirlar_index = df[df.isnull().any(axis=1)].index
        return eksik_kolonlar, na_df ,eksik_satirlar_index
    else:
        return eksik_kolonlar,na_df

def eksik_veri_cozum(df, tek_kolon=False, istenen_tek_col=False, cok_kolon=False,
                     kategorik_degiskene_gore=False, istenen_kategorik="", istenen_col=""):
    # eksik veriler çözümü => sayısalda ort ve kategorikte mod ile değişim
    # kardinal kategorik varsa DEĞİŞTİRMEZ!!! print ile uyarı yazdım

    # Tek col verilecekse tek_kolon True ve istenen kolon yazılır , çoklu kolon verilecekse cok_kolon True yapılır
    # coklu kolon seçiminde üstteki fonkdan oto olarak kolonları alır
    # tek kolonda ise istenen kolon seçilmeli

    # kategorik değişken kırılımına göre analiz istenirse
    # kategorik_degiskene_gore=False, istenen_kategorik="", istenen_col="" doldurulur

    # çıktı temp_df => kalıcı olması için df ye atanmalı

    eksik_kolonlar, na_df = eksik_veri_kontrol(df)
    print(df.isnull().sum(), end="\n\n\n")

    temp_df = df.copy()

    if tek_kolon:
        if temp_df[istenen_tek_col].dtype not in ["object", "bool", "category"]:
            temp_df[istenen_tek_col] = temp_df[istenen_tek_col].fillna(temp_df[istenen_tek_col].mean())
        else:
            if temp_df[istenen_tek_col].nunique() <= 10:
                temp_df[istenen_tek_col] = temp_df[istenen_tek_col].fillna(
                    temp_df[istenen_tek_col].mode()[0])  # tekli ise
            else:
                print("Kategorik değişkenin kardinal olma ihtimali var", end="\n\n\n")
    #    temp_df[istenen_tek_col] = temp_df[[istenen_tek_col]].apply(
    #    lambda x: x.fillna(x.mean()) if x.dtype not in ["object", "bool", "category"] else x)
    #    bu da olabilir ancak bazı işlemlerde hata verebilir

    if cok_kolon:
        temp_df[eksik_kolonlar] = temp_df[eksik_kolonlar].apply(
            lambda x: x.fillna(x.mean()) if x.dtype not in ["object", "bool", "category"] else x, )
        temp_df[eksik_kolonlar] = temp_df[eksik_kolonlar].apply(lambda x: x.fillna(x.mode()[0]) if (
                x.dtype in ["object", "bool", "category"] and x.nunique() <= 10) else x)
        for i in eksik_kolonlar:
            if temp_df[i].dtype in ["object", "bool", "category"] and temp_df[i].nunique() >= 10:
                print(str(i) + " Değişkeninin Kategorik değişkenin kardinal olma ihtimali var", end="\n\n\n")

    if kategorik_degiskene_gore:  # kategorik değişken kırılımına göre veri doldurmak için
        temp_df[istenen_col] = temp_df[istenen_col].fillna(
            temp_df.groupby(istenen_kategorik)[istenen_col].transform("mean"))
        # df[istenen_col].fillna(df[istenen_kategorik].map(df.groupby(istenen_kategorik)[istenen_col].mean()))
        # bu da olabilir

        # transformda kategorik değişken sütununa göre gruplama yapılır, Her grup için colna sütununun ortalaması hesaplanır. transformla fonk uygulanır ve fillnaya şutlanır
        # map te kategorik değişken sütunundaki her değere karşılık (map öncesi taraf), groupby serisinden eşleşen değeri alır ve bunları birleştirir
        # yani ilk kategorik değişkeni seçtim sonra bunla eşlemek istediğim değeri groupby ile seçtim ortaya da map koyarak birleştirdim
        # ikisinin de sonuçları aynı

    print("son hali", end="\n\n")
    print(temp_df.isnull().sum(), end="\n\n\n")

    return temp_df




# #########################################



def label_encoder(df, col):
    # tek bir col için tasarladım, birden fazla için yapılacaksa columnsların içinde olduğu liste seçilip for ile kullanılmalı
    # temp_df üzerinden yapıldığı için df e atanmalı

    temp_df = df.copy()
    data_dict = dict()
    le = LabelEncoder()
    temp_df[col + str("_encode")] = le.fit_transform(df[col])

    print(col + " Labels: " + str(le.inverse_transform(list(temp_df[col + str("_encode")].unique()))),
          end="\n\n")  # labelsları belirttim

    for i in range(len(temp_df[col + str("_encode")].unique())):
        data_dict.update({i: le.inverse_transform(list(temp_df[col + str("_encode")].unique()))[i]})

    data_series = pd.Series(data=data_dict, name=col)
    data_series = pd.Series.to_frame(data_series)

    print(data_series, end="\n\n")

    return temp_df

def one_hot_encoder(df, col):
    # tek bir col için tasarladım, birden fazla için yapılacaksa columnsların içinde olduğu liste seçilip for ile kullanılmalı
    # temp_df üzerinden yapıldığı için df e atanmalı

    temp_df = df.copy()

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop='first').set_output(transform="pandas")
    ohe_df = ohe.fit_transform(temp_df[[col]])
    temp_df = pd.concat([temp_df, ohe_df], axis=1)

    # pd.get_dummies(df,columns=[istenen col/lar],drop_first=True) # bunla da yapılabilir
    return temp_df

def rare_ozet(df, col, degisken):
    # tek veya çoklu kolon verilebilir !
    # kategorik değişkenlerin kategorilerini inceler,
    # oranlarına bakar, bağımlı değişkenle olan ilişkilerini inceler
    # çünkü bazı nadir sınıflar az sayıda gözlem içerse bile hedef değişkenle anlamlı ilişki taşıyabilir

    if type(col) != list:
        print(col, ":", len(df[col].value_counts()))
        print(pd.DataFrame({"Sayisi": df[col].value_counts(),
                            "Oran": df[col].value_counts(normalize=True) * 100,
                            "Bagimli": df.groupby(col)[degisken].mean()}), end="\n\n\n")



    else:
        for i in col:
            print(i, ":", len(df[i].value_counts()))
            print(pd.DataFrame({"Sayisi": df[i].value_counts(),
                                "Oran": df[i].value_counts(normalize=True) * 100,
                                "Bagimli": df.groupby(i)[degisken].mean()}), end="\n\n\n")

def rare_encoder(df, threshold=0.01):
    temp_df = df.copy()

    cat_cols = temp_df.select_dtypes(include="object").columns  # object tipi col seçimi yapar

    rare_cols = [i for i in cat_cols
                 if (temp_df[i].value_counts(normalize=True) < threshold).any()
                 and temp_df[i].nunique() <= 10]
    #  kategorik columnslardan oranı , belirlenen thresholddan düşük olan label içeren col seçimi yapar

    for i in rare_cols:
        rare_col_df = temp_df[i].value_counts(normalize=True)  # seçili col un label oranlarını verir

        rare_labels = rare_col_df[rare_col_df < threshold].index  # oranı, belirlenen thresholddan düşük olan labelları seçer
        # bunu yaparken değeri seçtikten sonra index bilgilerini alınca o labellara ulaşmış olur

        temp_df[i] = temp_df[i].apply(lambda x: "Rare" if x in rare_labels else x)  # seçili col un değerlerinde gezer
        # eğer seçilen label görülürse Rare yazılır yoksa aynen bırakılır , çıktı ise yeni df ye kaydedilir

    return temp_df





# #########################################







