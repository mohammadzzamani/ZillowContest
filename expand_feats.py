from sklearn import linear_model
import csv
import pandas as pd



zillow_file = 'data/z_reduced.csv'
language_file = 'data/csvX0.csv'
zillow_file = 'data/properties_2016.csv'
label_file = 'data/train_2016.csv'
output_file = 'data/zillow_lang_features3.csv'

def cats_to_int(data):
        # full_train_df.dtypes
        cat_columns = data.select_dtypes(['category','object']).columns
        print 'cat_columns: ' , cat_columns
        data = data.drop(cat_columns, axis=1)
        return data
        # for col in cat_columns:
        #     print 'col:  ' , col
        #     ls = list(data[col].values)
        #     # ls.extend(list(test[col].values))
        #     cats_list = list(set(ls))
        #     # cats_list = list(cats)
        #     data[col] = data[col].map(lambda x: cats_list.index(x) )
        #     # test[col] = test[col].map(lambda x: cats_list.index(x) )
        # return data #[train, test]

def multiply(fin_z, fin_l):
    all_df = fin_z
    for col in fin_l.columns:
        fin_z_col = fin_z.multiply(fin_l[col], axis="index")
        all_df = pd.concat([all_df, fin_z_col] , axis=1, join='inner')
    print ('shape: ' , all_df.shape)
    all_df.to_csv(output_file)
    return all_df



#read csv files
fin_z = pd.read_csv(zillow_file)
fin_l = pd.read_csv(language_file)
labels = pd.read_csv(label_file)


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
print fin_z.shape
print fin_l.shape

all_df = pd.merge(fin_l, fin_z, on='parcelid', how='inner')

#extracting lang feats and zillow feats
fin_l = all_df[fin_l.columns]
fin_z = all_df[fin_z.columns]

fin_z.set_index('parcelid', inplace=True)
fin_l.set_index('parcelid', inplace=True)
# print all_df.shape
# print fin_z.shape
# print fin_l.shape



#multiplication:
all_df = multiply(fin_z,fin_l)

#fill missing values with mean value
all_df.fillna(all_df.mean())



#add transction data and logerror to the final features
all_df.reset_index(inplace=True)
all_df=pd.merge(labels,all_df, how='inner', on='parcelid')
all_df.set_index(['parcelid', 'transactiondate'], inplace=True)
print 'all_df.shape: ', all_df.shape
print 'all_df.columns: ', all_df.columns





