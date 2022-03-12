# %%
import pandas as pd
import os
import seaborn as sns
import numpy as np
import category_encoders as ce
import xgboost as xgb

# %%
# Setting Pandas column display option
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# %%
idee = pd.read_csv('train_identity.csv')
transaction = pd.read_csv('train_transaction.csv')

# %%
merge = transaction.merge(idee, how='outer', on='TransactionID')

# %%
merge.dropna(thresh=0.7*len(merge), axis=1, inplace=True)

# %%
# selects grupos of variables to analyse correlation
v_cols = ['isFraud'] + list(filter(lambda x: x.startswith('V'), merge.columns.to_list()))
c_cols = ['isFraud'] + list(filter(lambda x: x.startswith('C'), merge.columns.to_list()))
d_cols = ['isFraud'] + list(filter(lambda x: x.startswith('D'), merge.columns.to_list()))

columns_groups = [v_cols, c_cols, d_cols]

# %%
# function to analyse correlation between variables. if correlation > 0.50, keep only one of the variables
def make_corr(columns):
    a = merge[columns].corr().abs().unstack().sort_values(kind="quicksort", ascending=False)
    a = a[a.between(0.50, 0.999999)]     
    b = list(set([i for j in a.index.to_list() for i in j]))
    return a, b

# apply function for each group of varibales
corr_groups = []
for i in columns_groups:
    a, b = make_corr(i)
    corr_groups.append(b)

# %%
# verify results for each group
for i in corr_groups:
    print(len(i))

# %%
# join dataframes
merge.loc[:, ~merge.columns.str.startswith('V')].columns.to_list()

A = merge[merge.loc[:, ~merge.columns.str.startswith('V')].columns.to_list()]
B = merge[corr_groups[0]]

merge = A.join(B)

# %%
merge_cols = merge.columns.to_list()
merge_cols.remove('isFraud')

# %%
numbers = merge.select_dtypes('number')
numbers.drop(columns={'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2'}, inplace=True)

# %%
# imput nans with the median for each column

numbers_imputed = numbers.fillna(value=numbers.median())

# %%
# separate categorical varibles to encode
objects = merge.select_dtypes('object')
objects = objects.join(merge[['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']])
objects.fillna("Unknown", inplace=True)
objects = objects.astype('object')
objects_columns = objects.columns

# %%
# hash encoding categoricals

hashing_enc = ce.HashingEncoder(cols=objects_columns, n_components=300, max_process=6).fit(objects, numbers.isFraud)

objects_encoded = hashing_enc.transform(objects.reset_index(drop=True))

# %%
X = numbers_imputed.join(objects_encoded).drop(columns='isFraud')
target = numbers.isFraud

print(X.shape, target.shape)

# %%
# XGBoost Model
xgmodel = xgb.XGBClassifier(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.04, max_depth=50, 
                             min_child_weight=1.7817, n_estimators=200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, random_state=7, nthread=-1)
xgmodel.fit(X, target)

# %%
# same pipeline for test set
idee_test = pd.read_csv('test_identity.csv')
transaction_test = pd.read_csv('test_transaction.csv') 
merge_test = transaction_test.merge(idee_test, how='outer', on='TransactionID')
merge_test = merge_test[merge_cols]

# %%
numbers_test = merge_test.select_dtypes('number').drop(columns=['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2'])
numbers_test_imputed = numbers_test.fillna(value=numbers.median()) # imputs based on the train set

# %%
objects_test = merge_test.select_dtypes('object')
objects_test = objects_test.join(merge_test[['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']])
objects_test.fillna("Unknown", inplace=True)
objects_test = objects_test.astype('object')

# %%
# hashing encoding categoricals
objects_test_encoded = hashing_enc.transform(objects_test.reset_index(drop=True))

# %%
X_test = numbers_test_imputed.join(objects_test_encoded)

# %%
y_pred_xg = xgmodel.predict_proba(X_test)

# %%
sub = pd.DataFrame()
sub['TransactionID'] = merge_test.TransactionID
sub['isFraud'] = y_pred_xg[:, 1]
sub.to_csv('submission_xgb.csv', index=False)

# %%



