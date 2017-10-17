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
# properties2016 = pd.read_csv('zillow_data/properties_2016.csv', low_memory = False)
properties2017 = pd.read_csv('zillow_data/properties_2017.csv', low_memory = False)

# properties2016 = properties2016.sample(frac=0.01)
# properties2017 = properties2017.sample(frac=0.01)

# properties2016 = cats_to_int_1param(properties2017)

print('Loading Train ...')
train2016 = pd.read_csv('zillow_data/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
train2017 = pd.read_csv('zillow_data/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)


print('Loading Language ...')
house_region = pd.read_csv('zillow_data/hid_rid.csv', low_memory = False)
region_feat = pd.read_csv('zillow_data/rid_feat.csv', low_memory = False)
region_featcount = pd.read_csv('zillow_data/rid_feat_count.csv', low_memory = False)

print ('merging language ...')
region_df = pd.merge(region_feat, region_featcount, how = 'left', on = 'rid')
language = pd.merge(house_region, region_feat, how = 'left', on = 'rid')
language.drop('rid', axis=1, inplace=True)
language = language.rename(columns = {'hid': 'parcelid'})

# train2016 = train2016.sample(frac=0.01)
# train2017 = train2017.sample(frac=0.01)

print ('adding date features')
train2016 = add_date_features(train2016)
train2017 = add_date_features(train2017)

print('Loading Sample ...')
sample_submission = pd.read_csv('zillow_data/sample_submission.csv', low_memory = False)
# sample_submission = sample_submission.sample(frac=0.004)


print('Concat Train 2016 & 2017 ...')
train_df = pd.concat([train2016, train2017], axis = 0)

print('Merge Train16 with Properties ...')
train_df = pd.merge(train_df, properties2017, how = 'left', on = 'parcelid')
train_df = pd.merge(train_df, language, how = 'left', on = 'parcelid')

# print('Merge Train17 with Properties ...')
# train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')
# train2017 = pd.merge(train2017, language, how = 'left', on = 'parcelid')

# print('Tax Features 2017  ...')
# train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan



language = language.rename(columns = {'parcelid': 'ParcelId'})


print ('merging sample submission with properties')
test_df = pd.merge(sample_submission[['ParcelId']], properties2017.rename(columns = {'parcelid': 'ParcelId'}), how = 'left', on = 'ParcelId')

print ('merging test_df with lang')
test_df = pd.merge(test_df, language, how = 'left', on = 'ParcelId')

del  properties2017, train2016, train2017
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
train_df = outlier_detection(train_df, thresh=0.9)
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



num_ensembles = 9
ypred = 0.0
for i in tqdm(range(num_ensembles)):

    model = CatBoostRegressor(
        iterations=630, learning_rate=0.03,
        depth=6, l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=i)
    print ('fitting ...', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    model.fit(
        Xtrain, Ytrain,
        cat_features=cat_feature_inds)
    print ( 'predicting ...', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    ypred += model.predict(Xtest)
ypred /= num_ensembles




prepare_final_submission(test_df, ypred, output_filename='data/lang_cbgb_finalprediction.csv')



