
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, fbeta_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold


def SCORES(y_val, pred, proba, str=None, cls_type=None) :
    if cls_type == "m" :
        # print("===========Multi Classifier======")
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average='macro')
        auc = roc_auc_score(y_val, proba, average='macro', multi_class='ovo')
        print('{} acc {:.4f}  f1 {:.4f}  auc {:.4f}'.format(str, acc, f1, auc))
    else :
        # print("===========Binary Classifier======")
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred)
        auc = roc_auc_score(y_val, proba[:,1])
        print('acc {:.4f}  f1 {:.4f}  auc {:.4f}  {}'.format(acc, f1, auc, str))


# dataset = load_iris()
# df = pd.DataFrame(data=dataset.data,
#                   #columns=dataset.feature_names
#                   columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#                   )
# clstype = "m"

# df = pd.read_csv("cancer_wisconsin.csv")
# print(df.info())
# df.drop(["id","Unnamed: 32"], axis=1, inplace=True)
# df["diagnosis"] = df["diagnosis"].map({"B":0,"M":1}).astype('int32')
# for col in df.columns:
#     df[col].fillna(df[col].mean(), inplace=True)
# # print(df.info())
# y_df = df['diagnosis']
# X_df = df.drop(["diagnosis"], axis=1)
# # print(X_df.shape, y_df.shape)  #(569,) (569, 30)


dataset = load_breast_cancer()
df = pd.DataFrame(data=dataset.data,
    columns=dataset.feature_names
)
clstype = "s"

df["target"] = dataset.target
X_df = df.iloc[: , :-1]
y_df = df.iloc[: , -1]
X_train7 , X_test3, y_train7, y_test3 = train_test_split(X_df, y_df, test_size=0.3, random_state=121)

model1 = RandomForestClassifier(random_state=11)
model2 = SVC(probability=True)
model3 = DecisionTreeClassifier(random_state=11) #LogisticRegression()

model2.fit(X_train7 , y_train7)
pred = model2.predict(X_test3)
proba = model2.predict_proba(X_test3)
SCORES(y_test3, pred,proba, "[SVC] ", cls_type='s')
#acc 0.9240  f1 0.9412  auc 0.9855  [SVC]

# def SCORES(y_val, pred, proba, str=None, cls_type=None) :
# f1 0.9717  f1:0.9412
#=================================================================================


#
#
fold_test_tot1 = np.zeros((X_train7.shape[0], 1))
fold_test_tot2 = np.zeros((X_train7.shape[0], 1))
fold_test_tot3 = np.zeros((X_train7.shape[0], 1))
#
X_test3_tot1 = []
X_test3_tot2 = []
X_test3_tot3 = []
#

skFold = StratifiedKFold(n_splits=5, random_state=111, shuffle=True) #(398,30) (398,)
for loop_cnt, (train_fold_idx, val_fold_idx) in enumerate(skFold.split(X_train7, y_train7)):
    X_train_fold4 = X_train7.iloc[train_fold_idx]
    y_train_fold4 = y_train7.iloc[train_fold_idx]
    X_val_fold1 = X_train7.iloc[val_fold_idx]
    y_val_fold1 = y_train7.iloc[val_fold_idx]

    model1.fit(X_train_fold4, y_train_fold4)
    model2.fit(X_train_fold4, y_train_fold4)
    model3.fit(X_train_fold4, y_train_fold4)

    pred1 = model1.predict(X_val_fold1).reshape(-1,1)
    pred2 = model2.predict(X_val_fold1).reshape(-1,1)
    pred3 = model3.predict(X_val_fold1).reshape(-1,1)

    fold_test_tot1[val_fold_idx, :]=pred1
    fold_test_tot2[val_fold_idx, :] = pred2
    fold_test_tot3[val_fold_idx, :] = pred3

    X_test3_tot1.append(model1.predict(X_test3)) #(170,30) --> [(171,),(171,)...]
    X_test3_tot2.append(model2.predict(X_test3))
    X_test3_tot3.append(model3.predict(X_test3))

stacking_new_train = np.concatenate([fold_test_tot1, fold_test_tot2, fold_test_tot3], axis=1) # 노란 세로 박스 318.4 * 3개 모델
#np.hstack([]) #axis=1

X_test3_tot1 = np.mean(X_test3_tot1, axis=0).reshape(-1,1) #np와 df는 axis 값을 반대로 생각하면 된다. np에서 axis=0은 세로줄 합산
X_test3_tot2 = np.mean(X_test3_tot2, axis=0).reshape(-1,1)
X_test3_tot3 = np.mean(X_test3_tot3, axis=0).reshape(-1,1)

stacking_new_test = np.concatenate([X_test3_tot1, X_test3_tot2, X_test3_tot3], axis=1)



    # bagg.fit(X_train, y_train)
    # pred = bagg.predict(X_test)
    # df_score = accuracy_score(pred, y_test)
    # scores.append(df_score)
    # print("Accuracy : {:.6f}".format(df_score))


#     #doto....
#
# #===============================================================================================================
xgb = XGBClassifier()
xgb.fit(stacking_new_train, y_train7)
xgb_pred = xgb.predict(stacking_new_test)
xgb_proba = xgb.predict_proba(stacking_new_test)
SCORES(y_test3, xgb_pred, xgb_proba, "[XGBClassifier] ", cls_type=clstype)
#
# #===============================================================================================================
lgbm = LGBMClassifier()
lgbm.fit(stacking_new_train, y_train7)
lgbm_pred = lgbm.predict(stacking_new_test)
lgbm_proba = lgbm.predict_proba(stacking_new_test)
SCORES(y_test3, lgbm_pred, lgbm_proba, "[LGBMClassifier] ", cls_type=clstype)

#스태킹은 데이터 양이 많을 때 유리(집단지성 발휘)