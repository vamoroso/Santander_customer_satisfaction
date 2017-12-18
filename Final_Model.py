#Load packages

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import cross_val_score, cross_val_predict
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb


## Note: This model takes 2 + hours to run due to 5-fold CV

# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

constant = []
keep_col = []
del_col = []
error_cols = []

# Drop ID and TARGET for modeling
target = train['TARGET']
train = train.drop(['ID', 'TARGET'], axis=1)
test_id = test.ID
test = test.drop(["ID"], axis=1)

# Create features for the number of zeros
train['Num Zeros'] = (train == 0).astype(int).sum(axis=1)
test['Num Zeros'] = (test == 0).astype(int).sum(axis=1)

# Mod 3 Count
train['mod3'] = train.apply(mod3_count, axis=1)
test['mod3'] = test.apply(mod3_count, axis=1)

# Indicator variable for errored observations
train['var03_na'] = [1 if x == -999999 else 0 for x in train.var3]
test['var03_na'] = [1 if x == -999999 else 0 for x in test.var3]

# Replace errored values in var3 with most common value of 2
train.var3 = train.var3.replace(-999999, 2)
test.var3 = test.var3.replace(-999999, 2)

# Indicator variable for most common values
train['var38_mc'] = [1 if x == 117310.979016494 else 0 for x in train.var38]
test['var38_mc'] = [1 if x == 117310.979016494 else 0 for x in test.var38]

train['var36_mc'] = [1 if x == 99 else 0 for x in train.var36]
test['var36_mc'] = [1 if x == 99 else 0 for x in test.var36]

# Calculating balances for previous time periods for var42
train[
    'sum_saldo_var42_hace3'] = train.saldo_medio_var5_hace3 + train.saldo_medio_var8_hace3 + train.saldo_medio_var29_hace3
train[
    'sum_saldo_var42_hace2'] = train.saldo_medio_var5_hace2 + train.saldo_medio_var8_hace2 + train.saldo_medio_var29_hace2
train['sum_saldo_var42_ult1'] = train.saldo_medio_var5_ult1 + train.saldo_medio_var8_ult1 + train.saldo_medio_var29_ult1
train['sum_saldo_var42_ult3'] = train.saldo_medio_var5_ult3 + train.saldo_medio_var8_ult3 + train.saldo_medio_var29_ult3

test['sum_saldo_var42_hace3'] = test.saldo_medio_var5_hace3 + test.saldo_medio_var8_hace3 + test.saldo_medio_var29_hace3
test['sum_saldo_var42_hace2'] = test.saldo_medio_var5_hace2 + test.saldo_medio_var8_hace2 + test.saldo_medio_var29_hace2
test['sum_saldo_var42_ult1'] = test.saldo_medio_var5_ult1 + test.saldo_medio_var8_ult1 + test.saldo_medio_var29_ult1
test['sum_saldo_var42_ult3'] = test.saldo_medio_var5_ult3 + test.saldo_medio_var8_ult3 + test.saldo_medio_var29_ult3

train['diff_ind5'] = train.apply(diff_ind5, axis=1)
test['diff_ind5'] = test.apply(diff_ind5, axis=1)

# Remove columns that are all constant (all zero's)
for col in train.columns:
    if train[col].nunique() == 1:
        constant.append(col)
train = train[list(np.setdiff1d(train.columns, constant))]
test = test[list(np.setdiff1d(test.columns, constant))]

# Remove Duplicate Columns
rl = remove_duplicates(train)
train = train[list(np.setdiff1d(train.columns, rl))]
test = test[list(np.setdiff1d(test.columns, rl))]

# Map var38 to log
train['var38'] = train.var38.map(np.log)
test['var38'] = train.var38.map(np.log)

# Limit variables based on min and max of test data
print('Setting min-max lims on test data')
for f in train.columns:
    lim = train[f].min()
    test[test[f] < lim] = lim

    lim = train[f].max()
    test[test[f] > lim] = lim

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.00375, random_state=3542)


#Create the sub models using the parameters from 4 of the best CV-grid search results
estimators = []
model1 = XGBClassifier(missing=np.nan, max_depth=5, n_estimators=560, learning_rate=0.02, n_jobs=4, subsample=0.7, colsample_bytree=0.7)
estimators.append(('xgb1', model1))
model2 = XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, subsample=0.95, colsample_bytree=0.85, random_state=6744)
estimators.append(('xgb2', model2))
model3 = XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, random_state=6744)
estimators.append(('xgb3', model3))
model4 = XGBClassifier(missing=np.nan, max_depth=5, n_estimators=400, learning_rate=0.02, subsample=0.7, colsample_bytree=0.9, random_state=6744)
estimators.append(('xgb4', model4))

#Fit Ensemble
eclf = VotingClassifier(estimators=[('xgb1', model1), ('xgb2', model2), ('xgb3', model3), ('xgb4', model4)], voting='soft')
eclf= eclf.fit(X_train, y_train)

#Calculate AUC for heldout test-set
print '\nVoting Ensemble Results:'
y_pred = eclf.predict_proba((X_test))[:,1]
print 'Last fitted model AUC:', roc_auc_score(y_test,y_pred)

#Calculate 5-Fold Cross Validated Score for model
n = len(X_train)
score = cross_val_score(eclf, X_train, y_train, cv=5, scoring= 'roc_auc').mean()
print 'Cross-Validated Score:', score

#Make predictions on Kaggle test file
y_pred = eclf.predict_proba((test))[:,1]

# submission = pd.DataFrame({"ID":test_id, "TARGET": y_pred})
# submission.to_csv("submission_vote2.csv", index=False)