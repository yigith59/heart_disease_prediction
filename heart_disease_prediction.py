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



ucl_df = pd.read_csv("df_ucl")
eda.ilkbakis(diger_df)

india_df = pd.read_csv("df_india")
eda.ilkbakis(india_df)


for i in ucl_df.columns:
    if ucl_df[i].dtype in ["int64","int32"]:
        ucl_df[i] = ucl_df[i].astype(float)

for i in india_df.columns:
    if india_df[i].dtype in ["int64","int32"]:
        india_df[i] = india_df[i].astype(float)



df_ = pd.concat([ucl_df,india_df],ignore_index=True)
df = df_.copy()

X = df[[i for i in df.columns if i not in ["target"]]]
y = df["target"]







X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=42)

ss = StandardScaler()
X_train[[i for i in X_train.columns]]= ss.fit_transform(X_train[[i for i in X_train.columns]])
X_test[[i for i in X_test.columns]]= ss.transform(X_test[[i for i in X_test.columns]])



def algoritma_ozet(X,y,score = "accuracy"):

    classifiers = [("LineerReg",LinearRegression()),
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

    for (name,model) in classifiers:
        cv_results = cross_validate(model,X,y,cv = 10, scoring=score)
        print(f'Model= {name} , Skor= {score}, {cv_results["test_score"].mean()}')
        liste.append((name,cv_results["test_score"].mean()))
    return classifiers,liste

classifiers, liste = algoritma_ozet(X_train,y_train,score="recall")



def hiperparametre_opt(modelismi,X_train,y_train,score="roc_auc"):

    param_grid_xgbboost = {
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

        'subsample': [0.7, 0.8, 0.9, 1.0] 

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


grid_search_best_params1 = hiperparametre_opt("LightGBM",X_train, y_train)
# best_params = {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 7, 'n_estimators': 150,'num_leaves': 20, 'reg_alpha': 0, 'reg_lambda': 0, 'subsample': 0.7}


model_l = LGBMClassifier(**grid_search_best_params1)
model_l.get_params()
model_l.fit(X_train,y_train)


y_pred = model_l.predict(X_test)
y_train_pred = model_l.predict(X_train)


y_prob = model_l.predict_proba(X_test)[:, 1]
y_train_prob = model_l.predict_proba(X_train)[:, 1]



print(classification_report(y_test,y_pred))

confusion_matrix(y_test,y_pred) 



print("\n--- Model Performansı (Eğitim Seti) ---")
print("Sınıflandırma Raporu:\n", classification_report(y_train, y_train_pred))
print("Karmaşıklık Matrisi:\n", confusion_matrix(y_train, y_train_pred))
print(f"ROC AUC Skoru: {roc_auc_score(y_train, y_train_prob):.4f}")


print("\n--- Model Performansı (Test Seti) ---")
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("Karmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))
print(f"ROC AUC Skoru: {roc_auc_score(y_test, y_prob):.4f}")


