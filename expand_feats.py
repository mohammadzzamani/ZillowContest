from sklearn import linear_model
import csv
import pandas as pd
import numpy as np


# language_file = 'data/lang_feats.txt'
# language_file = 'data/LangReduced.csv'
language_file = 'data/csvDall.csv'
zillow_file = 'zillow_data/properties_2016.csv'
label_file = 'zillow_data/train_2016_v2.csv'
zillow_file_17 = 'zillow_data/properties_2017.csv'
label_file_17 = 'zillow_data/train_2017.csv'
output_file = 'data/zillow_lang_features3.csv'
sample_submission = 'zillow_data/sample_submission.csv'
revised_sample_submission = 'data/revised_sample_submission.csv'

important_feat = ['parcelid', 'transactiondate', 'logerror', 'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'fips', 'latitude', 'longitude', 'propertycountylandusecode', 'propertylandusetypeid', 'rawcensustractandblock', 'regionidcounty', 'regionidzip', 'roomcnt', 'yearbuilt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount', 'censustractandblock'  ,'garagecarcnt', 'garagetotalsqft', 'fireplacecnt', 'fips' , 'heatingorsystemtypeid', 'buildingqualitytypeid']

def cats_to_int(data):
        print ('cats_to_int...')
        cat_columns = data.select_dtypes(['category','object']).columns
        print ('cat_columns: ' , cat_columns)

        for col in cat_columns:
            print ('col:  ' , col)
            data[col] = pd.Categorical(data[col])
            data[col] = data[col].astype('category').cat.codes
        return data

def multiply(fin_z, fin_l):
    print ('multiply...')
    all_df = fin_z
    for col in fin_l.columns:
        fin_z_col = fin_z.multiply(fin_l[col], axis="index")
        fin_z_col.columns = [ s+'_'+col for s in fin_z.columns]
        all_df = pd.concat([all_df, fin_z_col] , axis=1, join='inner')

    all_df.to_csv(output_file)
    return all_df

def break_transactiondate(dataframe):
    print ('break_transactiondate...')
    # dataframe['Month'] = (pd.DatetimeIndex(dataframe['transactiondate']).year - 2016) * 12 +  (pd.DatetimeIndex(dataframe['transactiondate']).month - 9 )
    dataframe['Year'] = pd.DatetimeIndex(dataframe['transactiondate']).year
    dataframe['Month'] = pd.DatetimeIndex(dataframe['transactiondate']).month
    # dataframe['Day'] = pd.DatetimeIndex(dataframe['transactiondate']).day
    return dataframe


def prepare_submission_data(submission_df):
    submission_df = submission_df[['parcelid']]
    print ('prepare_submission_data.....')
    print ('submission_df.shape: ' , submission_df.shape)
    print (submission_df.columns)
    cols = ['parcelid', '10/1/16', '11/1/16', '12/1/16', '10/1/17', '11/1/17', '12/1/17']
    for i in range(1 ,len(cols)):
        c = cols[i]
        submission_df[c] = 0
    submission_df.columns = cols
    print (submission_df.columns)
    submission_df = pd.melt(submission_df, id_vars=["parcelid"],var_name="transactiondate", value_name="logerror")
    print (submission_df)
    print ('submission_df.shape: ' , submission_df.shape)
    submission_df = break_transactiondate(submission_df)
    print ('submission_df.shape: ' , submission_df.shape)

    submission_df.set_index(['parcelid', 'transactiondate'], inplace=True)
    print ('submission_df.shape: ' , submission_df.shape)
    submission_df.to_csv(revised_sample_submission)
    print (submission_df)

def read_files():
    print ('read_files...')
    fin_z = pd.read_csv(zillow_file_17)
    fin_z = cats_to_int(fin_z)
    print (fin_z.columns)
    # important_feats = [ c for c in fin_z.columns if c in important_feat]
    # print (important_feats)
    # print ([ c for c in important_feats if c in fin_z.columns])
    # fin_z = fin_z[ [ c for c in important_feats if c in fin_z.columns] ]
    labels_16 = pd.read_csv(label_file)
    labels_17 = pd.read_csv(label_file_17)
    labels = pd.concat([labels_16, labels_17], ignore_index=True)
    fin_l = pd.read_csv(language_file)
    return fin_z, fin_l, labels

def original_data_preparation(fin_z, labels, fin_l=None):
    print ('original_data_preparation...')
    labels = break_transactiondate(labels)
    all_df = pd.merge(labels, fin_z,  how='inner', on='parcelid')
    print ('all_df.shape: ' , all_df.shape)
    all_df_columns = all_df.columns
    if fin_l is not None:
        # all_df=pd.merge(all_df, fin_l, on='parcelid', how='inner')
        all_df = pd.merge(all_df, fin_l, on='parcelid', how='left', indicator='Exist')
        all_df = all_df[[c for c in all_df.columns if (c in all_df_columns or c is 'Exist')]]
        all_df['Exist'] = np.where(all_df.Exist == 'both', 1.0, 0.0)


    all_df.set_index(['parcelid', 'transactiondate'], inplace=True)

    # print 'shapes: '
    # print 'submission_df.shape:  ', submission_df.shape
    print ('all_df.shape: ' , all_df.shape)
    # submission_df.fillna(all_df.mean(), inplace=True)


    all_df.fillna(all_df.mean(), inplace=True)
    print ('all_df.shape: ', all_df.shape)
    # print sum(pd.isnull(all_df['bathroomcnt']))
    # print all_df
    all_df.dropna(axis=1, how='any', inplace=True)
    all_df = all_df.sample(frac=1.0)

    # submission_df = submission_df[all_df.columns]

    all_df = all_df[all_df.Month >= 10]
    print ('final all_df.shape: ', all_df.shape)
    print ('final all_df.columns: ', all_df.columns)

    return all_df

def min_max_transformation(fin_l):
    print ('min_max_transformation...')
    fin_l.set_index('parcelid', inplace=True)
    fin_l = (fin_l - fin_l.min())/(fin_l.max()- fin_l.min())
    fin_l.dropna(axis=1, how='any', inplace=True)
    fin_l.reset_index(inplace=True)
    return fin_l






# fin_z = pd.read_csv(zillow_file)
# print 'finz_z.shape: ' , fin_z.shape
# fin_z_17 = pd.read_csv(zillow_file_17)
# print 'finz_z_17.shape: ' , fin_z_17.shape
# fin_z = cats_to_int(fin_z_17)


####### read csv files
[fin_z, fin_l, labels] = read_files()




# submission_df = pd.read_csv(revised_sample_submission)
# print ('submission_df.shape: ' , submission_df.shape)

#### need it just for the first run
## prepare_submission_data(fin_z)
## exit()


# submission_df = pd.merge(submission_df, fin_z, on='parcelid', how='inner')
# print ('submission_df.columns: ', submission_df.columns)
# print ('submission_df.shape: ', submission_df.shape)
# submission_df.set_index(['parcelid', 'transactiondate'], inplace=True)
# print ('submission_df.shape: ', submission_df.shape)

# ##### just for submission
# submission_df = fin_z.copy()
# submission_df.set_index('parcelid', inplace=True)
# #####
# submission_df = submission_df.iloc[:1000,:]

#### test on original data
print ('fin_l.shape: ' , fin_l.shape)
# fin_l = min_max_transformation(fin_l)
all_df = original_data_preparation(fin_z, labels, fin_l= None)
all_df = all_df.iloc[:20000,:]
#
# print ('all_df.shape: ' , all_df.shape)
# all_df=pd.concat([all_df, fin_l], axis=1, join='inner')
# print 'all_df.shape: ' , all_df.shape

# ##### just for submission
# submission_df.fillna(all_df.mean(), inplace=True)
# submission_df = submission_df[[c for c in all_df.columns if c in submission_df.columns]]
# print ('shapes: ' , submission_df.shape, ' , ', all_df.shape)
# print ('columns: ' )
# print (submission_df.columns)
# print ( all_df.columns)
# #####


#removing those rows in properties that we don't need
###all = pd.merge(labels, fin_z,  how='inner', on='parcelid')
# all_df.reset_index(inplace=True)
# fin_z = all_df[ [ c for c in fin_z.columns if c in all_df.columns] ]
# labels = all_df[ [ c for c in labels.columns if c in all_df.columns] ]

'''

####test on language adapted method

##now dropping categorical columns
fin_z=cats_to_int(fin_z)
# fin_l=cats_to_int(fin_l)
print (fin_z.shape)
print (fin_l.shape)

all_df = pd.merge(fin_l, fin_z, on='parcelid', how='inner')

#extracting lang feats and zillow feats
fin_l = all_df[fin_l.columns]
fin_z = all_df[fin_z.columns]

fin_z.set_index('parcelid', inplace=True)
fin_l.set_index('parcelid', inplace=True)


#multiplication:


print 'fin_z.columns: ' , fin_z.shape , ' , ' , fin_z.columns
print fin_z
print 'fin_l.columns: ' , fin_l.shape, ' , ', fin_l.columns
print fin_l
all_df = multiply(fin_z,fin_l)

all_df.reset_index(inplace=True)
all_df=pd.merge(labels,all_df, how='inner', on='parcelid')
all_df.set_index(['parcelid', 'transactiondate'], inplace=True)

print 'cols: '
print 'fin_z.columns: ' , fin_z.shape , ' , ' , fin_z.columns
print 'fin_l.columns: ' , fin_l.shape, ' , ', fin_l.columns
print 'labels.columns: ' , labels.shape, ' , ', labels.columns
print 'all_df.columns: ' , all_df.shape , ' , ',  all_df.columns



# fill missing values with mean value
all_df.fillna(all_df.mean(), inplace=True)
all_df.dropna(axis=1, how='any', inplace=True)


# #add transction data and logerror to the final features
# # print ('all_df.shape: ' , all_df.shape )
# # all_df=pd.concat([all_df, fin_l], axis=1, join='inner')
# print ('all_df.shape: ' , all_df.shape )
# all_df.reset_index(inplace=True)
# #fin_l.reset_index(inplace=True)
# all_df=pd.merge(labels,all_df, how='inner', on='parcelid')
# print ('all_df.shape: ' , all_df.shape )
# #all_df=pd.concat([all_df, fin_l], axis=1, join='inner')
# #print ('all_df.shape: ' , all_df.shape )
# #all_df=pd.merge(all_df, fin_l, how='inner', on='parcelid')
# all_df.set_index(['parcelid', 'transactiondate'], inplace=True)
# print ('all_df.shape: ', all_df.shape)
# print ('all_df.columns: ', all_df.columns)



# cols = ['logerror']  + [ c for c in fin_z.columns.values] # + [ c for c in fin_l.columns.values]
# print 'cols: ' , cols
# all_df = all_df[ [ c for c in cols if c in all_df.columns] ]
# print ('all_df.columns: ' )
# print all_df.columns

# all_df = final


print ('all_df.shape: ', all_df.shape)
print ('all_df.columns: ', all_df.columns)



'''




