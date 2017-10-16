import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostRegressor
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor
import gc
import datetime as dt
from Util import *

from time import gmtime, strftime


def cross_validation(data, train_feats= [], nolang_feats = [], folds = 5):
    fold_sizes = data.shape[0]*1.0/folds

    # Ytest = all_df.iloc[:,0]
    Ytest = data.logerror

    all_dfs = {'lang':data[nolang_feats], 'nolang':data}

    YpredsAll = { 'lang' : None , 'nolang' :None}
    for i in range(0,folds):
        for name, all_df in all_dfs.items():
            test_start = i* fold_sizes
            test_end = (i+1) * fold_sizes
            selection = [ True if ( i >=test_start and i < test_end) else False for i in range(all_df.shape[0])]
            deselection = [ False if val == True else True for val in selection]

            train = all_df.iloc[deselection]
            test = all_df.iloc[selection]

            train = outlier_detection(train, thresh=1)

            # trnsfrm = None #transformation()
            # train = trnsfrm.transform(train)


            print ('train.shape: ' , train.shape)
            print ('test.shape: ' , test.shape)

            Xtrain = train[train_features]
            Ytrain = train.logerror
            Xtest = test[train_features]
            thisYtest = test.logerror


            # Xtrain = train.iloc[:,1:]
            # Xtest = test.iloc[:,1:]
            # Ytrain = train.iloc[:,0]
            # thisYtest = test.iloc[:,0]

            Ypreds = None


            print (' ---- ')
            cntr = 0
            for estimator in ESTIMATORS:
                print ('cntr: ' , cntr)
                print ('fitting ...', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
                if cntr % 2 == 1:
                    estimator.fit(Xtrain, Ytrain, cat_features=cat_feature_inds)
                else:
                    estimator.fit(Xtrain, Ytrain)
                print ('predicting ...', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
                Ypred = estimator.predict(Xtest)

                cntr+=1

                Ypreds = stack_folds_preds(Ypred, Ypreds, 0)

                evaluate(thisYtest, Ypred, pre=str(cntr))

            if len(ESTIMATORS)>1:
                mean_  = np.mean(Ypreds[1:], axis=0)
                evaluate(thisYtest, mean_, pre= 'mean: ')
                mean_all  = np.mean(Ypreds, axis=0)
                evaluate(thisYtest, mean_all, pre= 'mean_all: ')
                # Ypreds = np.vstack((Ypreds , np.mean(Ypreds[1:], axis=0)))
                avg = Ypred[0,:]
                # wmean_ = [ mean_[i] if (np.sign(mean_[i]) == np.sign(avg[i]) and np.abs(mean_[i])> np.abs(avg[i]) ) else mean_all[i] for i in range(len(mean_)) ]
                # evaluate(thisYtest, wmean_, pre= 'wmean: ')
                Ypreds = np.vstack((Ypreds, mean_))
                Ypreds = np.vstack((Ypreds, mean_all))
                # Ypreds = np.vstack((Ypreds, wmean_))

            YpredsAll[name] = stack_folds_preds(Ypreds, YpredsAll[name], 1)


        for name, all_df in all_dfs.items():
            for ypred in YpredsAll[name]:
                evaluate(Ytest, ypred)



print('Loading Properties ...')
properties2016 = pd.read_csv('zillow_data/properties_2016.csv', low_memory = False)
properties2017 = pd.read_csv('zillow_data/properties_2017.csv', low_memory = False)

print('Loading Train ...')
train2016 = pd.read_csv('zillow_data/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
train2017 = pd.read_csv('zillow_data/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)

print('Loading Language ...')
house_region = pd.read_csv('zillow_data/hid_rid.csv', low_memory = False)
region_feat = pd.read_csv('zillow_data/rid_feat.csv', low_memory = False)
region_featcount = pd.read_csv('zillow_data/rid_feat_count.csv', low_memory = False)

language = pd.merge(house_region, region_feat, how = 'left', on = 'rid')
language = pd.merge(language, region_featcount, how = 'left', on = 'rid')
language.rename(columns = {'hid': 'parcelid'})

train2016 = add_date_features(train2016)
train2017 = add_date_features(train2017)

language_features = language.columns

print('Loading Sample ...')
sample_submission = pd.read_csv('zillow_data/sample_submission.csv', low_memory = False)

print('Merge Train with Properties ...')
train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')
train2016 = pd.merge(train2016, language, how = 'left', on = 'parcelid')
train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')
train2017 = pd.merge(train2017, language, how = 'left', on = 'parcelid')

# print('Tax Features 2017  ...')
# train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan

# print('Concat Train 2016 & 2017 ...')
train_df = pd.concat([train2016, train2017], axis = 0)
# test_df = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns = {'parcelid': 'ParcelId'}), how = 'left', on = 'ParcelId')

del properties2017#, test_df, properties2016, train2016, train2017
gc.collect();


# ESTIMATORS = [
#             mean_est(),
#             # GradientBoostingRegressor(n_estimators= 150, loss='lad', random_state=0, subsample=0.75, max_depth=6, max_features=0.75,  min_impurity_decrease=0.05, learning_rate=0.01),
#             # GradientBoostingRegressor(n_estimators= 250, loss='lad', random_state=1, subsample=0.75, max_depth=6, max_features=0.75,  min_impurity_decrease=0.04, learning_rate=0.02),
#             GradientBoostingRegressor(n_estimators= 250, loss='lad', random_state=2, subsample=0.75, max_depth=6, max_features=0.75,  min_impurity_decrease=0.03, learning_rate=0.03),
#             # GradientBoostingRegressor(n_estimators= 200, loss='lad', random_state=3, subsample=0.75, max_depth=6, max_features=0.75,  min_impurity_decrease=0.04, learning_rate=0.03),
#             # GradientBoostingRegressor(n_estimators= 200, loss='lad', random_state=4, subsample=0.75, max_depth=6, max_features=0.75,  min_impurity_decrease=0.04, learning_rate=0.03),
#             CatBoostRegressor(iterations=500, learning_rate=0.02,depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=5),
#             # CatBoostRegressor(iterations=530, learning_rate=0.03,depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=6),
#             # CatBoostRegressor(iterations=600, learning_rate=0.02,depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=7),
#             # CatBoostRegressor(iterations=500, learning_rate=0.02,depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=8),
#             # CatBoostRegressor(iterations=400, learning_rate=0.03,depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=9),
#         ]


ESTIMATORS = [
            mean_est(),
            # GradientBoostingRegressor(n_estimators= 150, loss='lad', random_state=0, subsample=0.75, max_depth=6, max_features=0.75,  min_impurity_decrease=0.05, learning_rate=0.01),

            CatBoostRegressor(iterations=500, learning_rate=0.02, depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=5, rsm=0.7),
            GradientBoostingRegressor(n_estimators= 250, loss='lad', random_state=0, subsample=0.75, max_depth=6, max_features=0.7,  min_impurity_decrease=0.03, learning_rate=0.02),



            # CatBoostRegressor(iterations=600, learning_rate=0.02, depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=5, rsm=0.85),
            # GradientBoostingRegressor(n_estimators= 300, loss='lad', random_state=0, subsample=0.85, max_depth=6, max_features=0.8,  min_impurity_decrease=0.03, learning_rate=0.02),
            # CatBoostRegressor(iterations=600, learning_rate=0.025, depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=6, rsm=0.9),
            # GradientBoostingRegressor(n_estimators= 250, loss='lad', random_state=1, subsample=0.85, max_depth=6, max_features=0.8,  min_impurity_decrease=0.04, learning_rate=0.02),
            # CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=7, rsm=0.9),
            # GradientBoostingRegressor(n_estimators= 250, loss='lad', random_state=2, subsample=0.8, max_depth=6, max_features=0.75,  min_impurity_decrease=0.03, learning_rate=0.03),
            # CatBoostRegressor(iterations=600, learning_rate=0.025, depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=8, rsm=0.85),
            # GradientBoostingRegressor(n_estimators= 200, loss='lad', random_state=3, subsample=0.75, max_depth=6, max_features=0.8,  min_impurity_decrease=0.04, learning_rate=0.03),
            # CatBoostRegressor(iterations=550, learning_rate=0.025, depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=9, rsm=0.9),
            # GradientBoostingRegressor(n_estimators= 300, loss='lad', random_state=4, subsample=0.8, max_depth=6, max_features=0.75,  min_impurity_decrease=0.04, learning_rate=0.03)

            # CatBoostRegressor(iterations=530, learning_rate=0.03,depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=6),
            # CatBoostRegressor(iterations=600, learning_rate=0.02,depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=7),
            # CatBoostRegressor(iterations=500, learning_rate=0.02,depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=8),
            # CatBoostRegressor(iterations=400, learning_rate=0.03,depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=9),
        ]

# for train_df in [train2016, train2017]:
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
# test_df.fillna(-999, inplace=True)


# train_df = outlier_detection(train_df, thresh = 1)
train_df = cats_to_int(train_df)
train_features = train_features + ['logerror']
print ' ... ' , len(train_features)
nolanguage_features = [ f for f in train_features if not f in language_features ]
# train_df = train_df[ train_features+['logerror']]
print ' ... ' , len(train_features) , ' , ', nolanguage_features, ' , ', train_df.shape


# print ("Training time !!")
# X_train = train_df[train_features]
# y_train = train_df.logerror
# print(X_train.shape, y_train.shape)

# test_df['transactiondate'] = pd.Timestamp('2016-12-01')
# test_df = add_date_features(test_df)
# X_test = test_df[train_features]
# print(X_test.shape)

# num_ensembles = 5
# y_pred = 0.0

cross_validation(train_df, train_features, nolanguage_features)














# for i in tqdm(range(num_ensembles)):
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


