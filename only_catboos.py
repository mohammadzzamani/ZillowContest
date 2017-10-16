import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostRegressor
from tqdm import tqdm
import gc
import datetime as dt
from Util import *
from sklearn.ensemble import GradientBoostingRegressor
from time import gmtime, strftime






print('Loading Properties ...')
properties2016 = pd.read_csv('zillow_data/properties_2016.csv', low_memory = False)
properties2017 = pd.read_csv('zillow_data/properties_2017.csv', low_memory = False)

# properties2016 = properties2016.sample(frac=0.01)
# properties2017 = properties2017.sample(frac=0.01)

properties2016, properties2017 = cats_to_int(properties2016, properties2017)

print('Loading Train ...')
train2016 = pd.read_csv('zillow_data/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
train2017 = pd.read_csv('zillow_data/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)

# train2016 = train2016.sample(frac=0.01)
# train2017 = train2017.sample(frac=0.01)

print ('adding date features')
train2016 = add_date_features(train2016)
train2017 = add_date_features(train2017)

print('Loading Sample ...')
sample_submission = pd.read_csv('zillow_data/sample_submission.csv', low_memory = False)
# sample_submission = sample_submission.sample(frac=0.004)


print('Merge Train with Properties ...')
train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')
train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')

# print('Tax Features 2017  ...')
# train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan

print('Concat Train 2016 & 2017 ...')
train_df = pd.concat([train2016, train2017], axis = 0)
test_df = pd.merge(sample_submission[['ParcelId']], properties2017.rename(columns = {'parcelid': 'ParcelId'}), how = 'left', on = 'ParcelId')

del  properties2017, train2016, train2017#, properties2016
gc.collect();

print('Remove missing data fields ...')

missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % len(exclude_missing))

del num_rows, missing_perc_thresh
gc.collect();

print ("Remove features with one unique value !!")
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % len(exclude_unique))

print ("Define training features !!")
exclude_other = ['parcelid', 'logerror','propertyzoningdesc']
train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % len(train_features))

print ("Define categorial features !!")
cat_feature_inds = []
cat_unique_thresh = 1000
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh \
       and not 'sqft' in c \
       and not 'cnt' in c \
       and not 'nbr' in c \
       and not 'number' in c:
        cat_feature_inds.append(i)

print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

print ("Replacing NaN values by -999 !!")
train_df.fillna(-999, inplace=True)
# train_df = outlier_detection(train_df)
test_df.fillna(-999, inplace=True)

# train_df = train_df.sample(frac=0.1)

print ("Training time !!")
Xtrain = train_df[train_features]
Ytrain = train_df.logerror
print(Xtrain.shape, Ytrain.shape)


submission_df = get_submission_format(test_df)
print ('-------- shapes : ' , submission_df.shape , ' , ', test_df.shape)
test_df = pd.merge(test_df, submission_df, how='left', on='ParcelId')


# test_df_temp['transactiondate'] = pd.Timestamp('2016-10-01')
# test_df_final = pd.concat([test_df, train2017], axis = 0)
print ('adding date features to test data')
test_df = add_date_features(test_df, drop_transactiondate=False)
test_df.set_index(['ParcelId', 'transactiondate'], inplace=True)
Xtest = test_df[train_features]
print(Xtest.shape)

ESTIMATORS = [
            # mean_est(),
            CatBoostRegressor(iterations=650, learning_rate=0.02, depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=5, rsm=0.85),
            GradientBoostingRegressor(n_estimators= 300, loss='lad', random_state=0, subsample=0.85, max_depth=6, max_features=0.75,  min_impurity_decrease=0.03, learning_rate=0.02),
            CatBoostRegressor(iterations=650, learning_rate=0.025, depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=6, rsm=0.9),
            GradientBoostingRegressor(n_estimators= 250, loss='lad', random_state=1, subsample=0.85, max_depth=6, max_features=0.8,  min_impurity_decrease=0.04, learning_rate=0.02),
            CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=7, rsm=0.9),
            GradientBoostingRegressor(n_estimators= 250, loss='lad', random_state=2, subsample=0.8, max_depth=6, max_features=0.75,  min_impurity_decrease=0.03, learning_rate=0.03),
            CatBoostRegressor(iterations=600, learning_rate=0.025, depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=8, rsm=0.85),
            GradientBoostingRegressor(n_estimators= 200, loss='lad', random_state=3, subsample=0.75, max_depth=6, max_features=0.8,  min_impurity_decrease=0.04, learning_rate=0.03),
            CatBoostRegressor(iterations=550, learning_rate=0.025, depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=9, rsm=0.9),
            GradientBoostingRegressor(n_estimators= 300, loss='lad', random_state=4, subsample=0.8, max_depth=6, max_features=0.75,  min_impurity_decrease=0.04, learning_rate=0.03)
    ]


# num_ensembles = len(ESTIMATORS)
Ypreds = None

for cntr in range(len(ESTIMATORS)):
    estimator = ESTIMATORS[cntr]
    print ('cntr: ' , cntr)
    print ('fitting ...', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    if cntr % 2 == 0:
        estimator.fit(Xtrain, Ytrain, cat_features=cat_feature_inds)
    else:
        estimator.fit(Xtrain, Ytrain)
    print ('predicting ...', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    Ypred = estimator.predict(Xtest)


    Ypreds = stack_folds_preds(Ypred, Ypreds, 0)
    print ('Ypred: ' , Ypred)
    print ('Ypreds.shape: ', Ypreds.shape)
    print ('Ypreds: ' , Ypreds)


if len(ESTIMATORS)>1:
    ypred_mean  = np.mean(Ypred, axis=0)


prepare_final_submission(test_df, ypred_mean)
# test_df['logerror'] = ypred_mean





#     model = CatBoostRegressor(
#         iterations=630, learning_rate=0.03,
#         depth=6, l2_leaf_reg=3,
#         loss_function='MAE',
#         eval_metric='MAE',
#         random_seed=i)
#     print ( 'fitting ... ')
#     model.fit(
#         X_train, y_train,
#         cat_features=cat_feature_inds)
#     print ( 'predicting ...')
#     y_pred += model.predict(X_test)
# y_pred /= num_ensembles
#
# submission = pd.DataFrame({
#     'ParcelId': test_df['ParcelId'],
# })
# test_dates = {
#     '201610': pd.Timestamp('2016-09-30'),
#     '201611': pd.Timestamp('2016-10-31'),
#     '201612': pd.Timestamp('2016-11-30'),
#     '201710': pd.Timestamp('2017-09-30'),
#     '201711': pd.Timestamp('2017-10-31'),
#     '201712': pd.Timestamp('2017-11-30')
# }
# for label, test_date in test_dates.items():
#     print("Predicting for: %s ... " % (label))
#     submission[label] = y_pred
#
# submission.to_csv('Only_CatBoost.csv', float_format='%.6f',index=False)