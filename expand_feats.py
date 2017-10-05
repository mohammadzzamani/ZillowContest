from sklearn import linear_model
import csv
import pandas as pd
import numpy as np


language_file = 'data/csvDall.csv'
# language_file = 'data/LangReduced.csv'
zillow_file = 'data/properties_2016.csv'
label_file = 'data/train_2016.csv'
output_file = 'data/zillow_lang_features3.csv'

def cats_to_int(data):
        # full_train_df.dtypes
        cat_columns = data.select_dtypes(['category','object']).columns
        print ('cat_columns: ' , cat_columns)
        # data = data.drop(cat_columns, axis=1)

        for col in cat_columns:
            print 'col:  ' , col
            data[col] = pd.Categorical(data[col])
            data[col] = data[col].astype('category').cat.codes


        # for col in cat_columns:
        #     print 'col:  ' , col
        #     ls = list(data[col].values)
        #     # ls.extend(list(test[col].values))
        #     cats_list = list(set(ls))
        #     print 'cat_list: ' , cats_list
        #     # cats_list = list(cats)
        #     data[col] = data[col].map(lambda x: cats_list.index(x) )
        #     # test[col] = test[col].map(lambda x: cats_list.index(x) )
        return data #[train, test]

def multiply(fin_z, fin_l):
    all_df = fin_z
    for col in fin_l.columns:
        fin_z_col = fin_z.multiply(fin_l[col], axis="index")
        fin_z_col.columns = [ s+'_'+col for s in fin_z.columns]
        all_df = pd.concat([all_df, fin_z_col] , axis=1, join='inner')

    all_df.to_csv(output_file)
    return all_df



#read csv files
fin_z = pd.read_csv(zillow_file)
fin_l = pd.read_csv(language_file)
labels = pd.read_csv(label_file)


#min_max_transformation
fin_l.set_index('parcelid', inplace=True)
fin_l = (fin_l - fin_l.min())/(fin_l.max()- fin_l.min())
fin_l.dropna(axis=1, how='any', inplace=True)
fin_l.reset_index(inplace=True)

#########
labels['Year'] = pd.DatetimeIndex(labels['transactiondate']).year
labels['Month'] = pd.DatetimeIndex(labels['transactiondate']).month
labels['Day'] = pd.DatetimeIndex(labels['transactiondate']).day
fin_z = cats_to_int(fin_z)
all_df = pd.merge(labels, fin_z,  how='inner', on='parcelid')
all_df.set_index(['parcelid', 'transactiondate'], inplace=True)
all_df.fillna(all_df.mean(), inplace=True)
all_df.dropna(axis=1, how='any', inplace=True)
print ('all.shape: ', all_df.shape)
print ('all.columns: ', all_df.columns)
#########

'''
#removing those rows in properties that we don't need
all = pd.merge(labels, fin_z,  how='inner', on='parcelid')
fin_z = all[fin_z.columns]
labels = all[labels.columns]





#alternative
# all_df = pd.concat([fin_z, fin_l], axis=1, join='inner', join_axes=['parcelid'])
# print ( all_df.shape)
# all_df = cats_to_int(all_df)

#now dropping categorical columns
fin_z=cats_to_int(fin_z)
fin_l=cats_to_int(fin_l)
print (fin_z.shape)
print (fin_l.shape)

all_df = pd.merge(fin_l, fin_z, on='parcelid', how='inner')

#extracting lang feats and zillow feats
fin_l = all_df[fin_l.columns]
fin_z = all_df[fin_z.columns]

fin_z.set_index('parcelid', inplace=True)
fin_l.set_index('parcelid', inplace=True)


#multiplication:
all_df = multiply(fin_z,fin_l)

#fill missing values with mean value

all_df.fillna(all_df.mean(), inplace=True)
# all_df.fillna(0, inplace=True)
all_df.dropna(axis=1, how='any', inplace=True)
# print all_df[:2]

#add transction data and logerror to the final features
print ('all_df.shape: ' , all_df.shape )
all_df=pd.concat([all_df, fin_l], axis=1, join='inner')
print ('all_df.shape: ' , all_df.shape )
all_df.reset_index(inplace=True)
#fin_l.reset_index(inplace=True)
all_df=pd.merge(labels,all_df, how='inner', on='parcelid')
print ('all_df.shape: ' , all_df.shape )
#all_df=pd.concat([all_df, fin_l], axis=1, join='inner')
#print ('all_df.shape: ' , all_df.shape )
#all_df=pd.merge(all_df, fin_l, how='inner', on='parcelid')
all_df.set_index(['parcelid', 'transactiondate'], inplace=True)
print ('all_df.shape: ', all_df.shape)
print ('all_df.columns: ', all_df.columns)



# cols = ['logerror']  + [ c for c in fin_z.columns.values] # + [ c for c in fin_l.columns.values]
# print 'cols: ' , cols
# all_df = all_df[ [ c for c in cols if c in all_df.columns] ]
# print ('all_df.columns: ' )
# print all_df.columns

# all_df = final
# print ('all_df.shape: ', all_df.shape)
# print ('all_df.columns: ', all_df.columns)


'''




