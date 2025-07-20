import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno

from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, \
    confusion_matrix, classification_report, balanced_accuracy_score, roc_curve, auc, silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, \
    validation_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler, OneHotEncoder, \
    PolynomialFeatures, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer


# pip install catboost
# pip install xgboost
# pip install lightgbm

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



import warnings
import EDA as eda


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 30)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)


def ilkbakis(df):
    print("------------------------------------------", end="\n\n")
    print("-------------- df özellikleri --------------" , end="\n\n")

    print("-------------- head --------------", end="\n\n")
    print(df.head(), end="\n\n")

    print("-------------- shape --------------", end="\n\n")
    print(f"shape: {df.shape}", end="\n\n")

    print("-------------- info --------------", end="\n\n")
    print(df.info(), end="\n\n")

    print("-------------- dtypes --------------", end="\n\n")
    print(df.dtypes.value_counts(), end="\n\n")

    print("-------------- isnull --------------", end="\n\n")
    print(f"isnull : {df.isnull().values.any()} , sum = {df.isnull().values.sum()}", end="\n\n\n")

    if df.isnull().values.any():
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



def kategorik_ozet(df,col,plot=False):

    # tekli kullanım
    # çoklu kullanımda for döngüsü ile

        if plot:

            print(f"------------- kolon = {col} ---------------", end="\n\n")
            print(pd.DataFrame({"Oran": df[col].value_counts(normalize=True,dropna=False)*100,
                                "Sayıları": df[col].value_counts(dropna=False)}), end="\n\n\n")

            if df[col].dtype == bool:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 satırda 2 ayrı grafik

                sns.countplot(x=df[col].astype(int), ax=axes[0],palette="vlag",hue=df[col],legend=False)  # İlk grafiği sol tarafa koy
                axes[0].set_title(f"{col} - Countplot")

                n_colors = df[col].nunique()
                common_palette = sns.color_palette("pastel", n_colors)

                df[col].value_counts().plot.pie(autopct="%1.1f%%",
                                                ax=axes[1],colors=common_palette)  # İkinci grafiği sağ tarafa koy
                axes[1].set_title(f"{col} - Pie Chart")

                plt.tight_layout()
                plt.show(block=True)

            else:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 satırda 2 ayrı grafik

                sns.countplot(x=df[col], ax=axes[0],palette="vlag", hue=df[col],legend=False)  # İlk grafiği sol tarafa koy
                axes[0].set_title(f"{col} - Countplot")

                n_colors = df[col].nunique()
                common_palette = sns.color_palette("pastel", n_colors)

                df[col].value_counts().plot.pie(autopct="%1.1f%%",
                                                ax=axes[1],colors=common_palette)  # İkinci grafiği sağ tarafa koy
                axes[1].set_title(f"{col} - Pie Chart")

                plt.tight_layout()
                plt.show(block=True)

        else:
            print(f"------------- kolon = {col} ---------------", end="\n\n")
            print(pd.DataFrame({"Oran": df[col].value_counts(normalize=True,dropna=False),
                                "Sayıları": df[col].value_counts(dropna=False)}).sort_index(), end="\n\n\n")



def sayisal_ozet(df,col,plot=False):
    # tekli kullanım
    # Hesaplanan eşiklere göre gerçek aykırı değerlerin sayısını ve yüzdesini de yazdırır.


    print(f"------------- kolon = {col} ---------------", end="\n\n")

    print(" --------------- describe  ---------------" )
    print(df[col].describe(), end="\n\n\n")


    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        sns.boxplot(x=df[col], ax=ax[0], palette="pastel")
        ax[0].set_title(f"{col} - Boxplot")

        sns.histplot(df[col], kde=True, ax=ax[1], palette="pastel", bins=30)  # KDE eklenmiş histogram
        ax[1].set_title(f"{col} - Histogram + KDE")

        plt.show(block=True)




# ---------------------------------------------------------------------------------------------------------



def threshold_hesaplama(df,col, a, b):
    # outlier tespiti için min ve max değerleri hesaplar

    q1 = df[col].quantile(a)
    q3 = df[col].quantile(b)
    iqr = q3 - q1

    min_deger = q1 - 1.5 * iqr
    max_deger = q3 + 1.5 * iqr

    return min_deger,max_deger




def outlier_kontrol(df,col,a,b,index=False):

    # outlier var mı kontrol eder
    # tek kullanılmalı , liste kullanılacaksa for ile kullanılmalı

    temp_list = []
    min_deger, max_deger = threshold_hesaplama(df,col,a,b)


    if (df[col] > max_deger).any():
        print(str(col) + " Ust deger outlier var")
        temp_list.append(col)


    if (df[col] < min_deger).any():
        print(str(col) + " Alt deger outlier var")
        temp_list.append(col)


    if temp_list:
        print(f"{col} outlier içeriyor", end="\n")
        print("----------------------------", end="\n\n")


    else:
        print(f"{col} outlier içermiyor", end="\n\n")
        print("----------------------------", end="\n\n")



    if index:
        return df[(df[col] > max_deger) | (df[col] < min_deger)].index



def outlier_cozum(df,col,a=0.25,b=0.75,index_del=False, secim=False, baskilama=False):
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

    # df.index.delete(df[(df["Age"]>ust_deger) | (df["Age"]<alt_deger)].index) # çıktı olarak outlier olmayanların indexlerini verir
    # median,ortalama gibi şeylerle de değiştirilebilir



# ----------------------------------------------------------------------------------------------------------



def eksik_veri_kontrol(df, index=False,goruntu=False):

    print(f"Eksik değer : {df.isnull().values.any()} , toplam : {df.isnull().values.sum()}")
    print("-----------------------",end="\n\n")

    eksik_kolonlar = [i for i in df.columns if df[i].isnull().values.any() == True]
    # if sonrasına istenirse df[i].isnull().sum() > 0 da denebilir  # missing value olan col seçimi

    na_sayisi = df[eksik_kolonlar].isnull().sum()  # toplamı
    yuzdelik = df[eksik_kolonlar].isnull().sum() * 100 / df.shape[0] # yüzdeliği
    na_df = pd.DataFrame({"Eksik Değer": na_sayisi,
                          "Yüzdelik": yuzdelik})
    na_df = na_df.sort_values("Eksik Değer",ascending=False) # tablo halinde özet
    print(na_df)


    if goruntu:
        msno.matrix(df)
        plt.show(block=True)


    if index: # eksik verilerin olduğu satırların index bilgisi
        eksik_satirlar_index = df[df.isnull().any(axis=1)].index
        return eksik_kolonlar,eksik_satirlar_index
    else:
        return eksik_kolonlar



def knn_imputasyonu(df, olmayacak_kolonlar):
    # knn imputasyonu için

    eksik_veri_col = eda.eksik_veri_kontrol(df)
    eksik_knnkolonlar = [i for i in eksik_veri_col if i not in olmayacak_kolonlar]

    print("")
    print(f"Eksik kolonlar: {eksik_veri_col}")
    print(f"KNN için kullanılmayacak kolonlar: {olmayacak_kolonlar}")
    print(f"Eksik veri var ama KNN yapılamayacak kolonlar = {[i for i in eksik_veri_col if i not in eksik_knnkolonlar]}",end="\n\n\n")
    print("-------------------------------------------------------")

    temp_df = df.copy()
    temp_df = temp_df[[i for i in temp_df.columns if i not in olmayacak_kolonlar]]

    ss_temp = StandardScaler()
    temp_df[[i for i in temp_df.columns if i not in eksik_knnkolonlar]] = ss_temp.fit_transform(temp_df[[i for i in temp_df.columns if i not in eksik_knnkolonlar]])
    # knn uygulanmayacak kolonları std yapmak için

    df_for_processing = temp_df.copy()

    knn_imputer = KNNImputer(n_neighbors=5)

    df_imputed_array = knn_imputer.fit_transform(df_for_processing)

    df_imputed_temp = pd.DataFrame(df_imputed_array,
                                   columns=df_for_processing.columns,
                                   index=df_for_processing.index)
    print("-------------------------------------------------------",end="\n\n")

    for i in eksik_knnkolonlar:
        print(f"{i} imputasyon Öncesi : {df[i].describe()}",end="\n\n")
        print("-------------------------------------------------------")

        df[i] = df_imputed_temp[i]  # df e impute edilmiş halini şutlar

        print(f"{i} imputasyon Sonrası : {df[i].describe()}",end="\n\n")
        print("-------------------------------------------------------")
        print("-------------------------------------------------------",end="\n\n")



    print(f"Son Hali = {eda.eksik_veri_kontrol(df)} ")



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

    eksik_kolonlar, na_df = eksik_veri_kontrol(df, index=False)
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
        temp_df[eksik_kolonlar] = temp_df[eksik_kolonlar].apply(lambda x: x.fillna(x.mean()) if x.dtype not in ["object", "bool", "category"] else x )
        temp_df[eksik_kolonlar] = temp_df[eksik_kolonlar].apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype in ["object", "bool", "category"] and x.nunique() <= 10) else x)

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





# ----------------------------------------------------------------------------------------------------------



def label_encoder(df, col):

    # tek bir col için tasarladım,
    # birden fazla için yapılacaksa columnsların içinde olduğu liste seçilip for ile kullanılmalı !!!!!!!!!!
    # temp_df üzerinden yapıldığı için df e atanmalı

    temp_df = df.copy()
    data_dict = dict()

    le = LabelEncoder()
    yeni_isim = str(col) + str("_encode")
    temp_df[yeni_isim] = le.fit_transform(df[col])

    # fit kısmı: Verideki tüm eşsiz kategorileri tanımlar (kırmızı, yeşil, mavi), hesaplamayı yapar
    # transform kısmı: Bu kategorilere karşılık gelen sayısal kodları verir. hesaplamaları veriye uygular

    print(f"{col} Labels: {le.inverse_transform(list(temp_df[yeni_isim].unique()))}",end="\n\n")  # labelsları belirttim, list olmasa da olur


    for i in range(len(temp_df[yeni_isim].unique())):
        data_dict.update({i: le.inverse_transform(list(temp_df[yeni_isim].unique()))[i]})  # hangi sayısal değere hangi orijinal karakter atadı


    data_series = pd.Series(data=data_dict, name=col) # Series haline geldi ki güzel görünsün
    data_df = pd.Series.to_frame(data_series) # df yaptı

    print(data_df, end="\n\n")

    return temp_df



def one_hot_encoder(df, col,drop=False):

    # tek bir col için tasarladım, birden fazla için yapılacaksa columnsların içinde olduğu liste seçilip for ile kullanılmalı
    # temp_df üzerinden yapıldığı için df e atanmalı
    # drop true yapılırsa oluşturulan col u siler

    temp_df = df.copy()

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop='first').set_output(transform="pandas")
    ohe_df = ohe.fit_transform(temp_df[[col]])
    temp_df = pd.concat([temp_df, ohe_df], axis=1)

    if drop:
        temp_df = temp_df.drop(col, axis=1)

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
                            "Bagimli Degiskene Gore": df.groupby(col)[degisken].mean()}), end="\n\n\n")



    else:
        for i in col:
            print(i, ":", len(df[i].value_counts()))
            print(pd.DataFrame({"Sayisi": df[i].value_counts(),
                                "Oran": df[i].value_counts(normalize=True) * 100,
                                "Bagimli Degiskene Gore": df.groupby(i)[degisken].mean()}), end="\n\n\n")



def rare_encoder(df, threshold=0.01):
    temp_df = df.copy()

    kategorik_kolon = temp_df.select_dtypes(include=["object", "category"]).columns  # object tipi col seçimi yapar

    rare_kolon = [i for i in kategorik_kolon
                 if (temp_df[i].value_counts(normalize=True) < threshold).any()
                 and temp_df[i].nunique() <= 10]

    #  kategorik columnslardan oranı , belirlenen thresholddan düşük olan label içeren col seçimi yapar

    for i in rare_kolon:
        rare_col_df = temp_df[i].value_counts(normalize=True)  # seçili col un label oranlarını verir

        rare_labels = rare_col_df[rare_col_df < threshold].index  # oranı, belirlenen thresholddan düşük olan labelları seçer
        # bunu yaparken değeri seçtikten sonra index bilgilerini alınca o labellara ulaşmış olur

        temp_df[i] = temp_df[i].apply(lambda x: "Rare" if x in rare_labels else x)  # seçili col un değerlerinde gezer
        # eğer seçilen label görülürse Rare yazılır yoksa aynen bırakılır , çıktı ise yeni df ye kaydedilir

    return temp_df



# ----------------------------------------------------------------------------------------------------------

def korelasyon(df,kolonlar,gorsel = False):
    korelasyon_matrisi = df[kolonlar].corr()


    if gorsel:
        plt.figure(figsize=(10, 8)) # Grafiğin boyutunu ayarla

        sns.heatmap(
            korelasyon_matrisi,
            annot=True,      # Korelasyon değerlerini hücrelerin içine yaz
            cmap='coolwarm',
            fmt=".2f",
            linewidths=.5,   # Hücreler arasına çizgi ekle
            cbar=True  )      # Renk çubuğunu göster (değer aralığını belirtir)

        plt.title('Korelasyon Haritası', fontsize=16)
        plt.show(block=True)

        return korelasyon_matrisi

    else:

        return korelasyon_matrisi



# ----------------------------------------------------------------------------------------------------------




def algoritma_ozet(X,y,score = "accuracy"):

    algoritmalar = [
                    ("LogReg",LogisticRegression()),
                    ("KNN",KNeighborsClassifier()),
                    ("CART", DecisionTreeClassifier()),
                    ("RandFor",RandomForestClassifier()),
                    ("Adaboost",AdaBoostClassifier()),
                    ("GBM",GradientBoostingClassifier()),
                    ("LightGBM",LGBMClassifier(verbose=-1)),
                    ("XGBoost", XGBClassifier(enable_categorical=True))
                    ]



    liste = []

    for (i,j) in algoritmalar:
        cv = cross_validate(j,X,y,cv = 10, scoring=score)
        print(f'Model= {i} , Skor= {score}, {cv["test_score"].mean()}')
        liste.append((i,cv["test_score"].mean()))
    return algoritmalar,liste




def hiperparametre_optimizasyonu(modelismi,X_train,y_train,score="roc_auc"):

    param_grid_xgbboost = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': [0.05, 0.1],
            'max_depth': [3, 5],
            'n_estimators': [100, 200],
    }
    param_grid_lightgbm = {
        'n_estimators': [150, 250, 350],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [20, 31, 40],
        'max_depth': [-1, 5, 7],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'subsample': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1]
    }
    param_grid_filtered_cart = {
            'max_depth': [None, 3, 5, 7, 10, 15],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': [None, 'sqrt', 'log2', 0.5, 0.75],
            'criterion': ['gini', 'entropy']
    }
    param_grid_logistic = [
            {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'solver': ['lbfgs', 'newton-cg', 'sag'], 'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'solver': ['saga'], 'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    ]
    param_grid_gbm = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.7, 0.8, 0.9, 1.0],
    }




    liste = [
             (param_grid_xgbboost,"XGBoost",XGBClassifier(enable_categorical=True)),
             (param_grid_lightgbm,"LightGBM",LGBMClassifier(verbose=-1)),
             (param_grid_filtered_cart,"CART",DecisionTreeClassifier()),
             (param_grid_logistic,"LogReg",LogisticRegression()),
             (param_grid_gbm,"GBM",GradientBoostingClassifier())]





    for (param,name,func) in liste:
        if name == modelismi:
            model = func
            grid_search = GridSearchCV(estimator=model,
                                       param_grid=param,
                                       cv=5,
                                       scoring=score,
                                       verbose=1,
                                       n_jobs=-1
                                       )

            grid_search.fit(X_train, y_train)


            print(f"\n En iyi hiperparametreler: {grid_search.best_params_}")


            return grid_search.best_params_

        else:
            continue






























