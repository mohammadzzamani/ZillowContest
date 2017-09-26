from sklearn import linear_model
import csv
import pandas as pd


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

zillow_file = 'data/z_reduced.csv'
language_file = 'data/csvX0_2row.csv'
zillow_file = 'data/properties_2016.csv'
output_file = 'data/zillow_lang_features3.csv'


fin_z = pd.read_csv(zillow_file)
fin_l = pd.read_csv(language_file)


fin_z.set_index('parcelid', inplace=True)
fin_l.set_index('parcelid', inplace=True)

# all_df = pd.concat([fin_z, fin_l], axis=1, join='inner', join_axes=['parcelid'])
# print ( all_df.shape)
# all_df = cats_to_int(all_df)

fin_z=cats_to_int(fin_z)
fin_l=cats_to_int(fin_l)
all_df = pd.concat([fin_z, fin_l], axis=1, join='inner')#, join_axes=['parcelid'])
print ( all_df.shape)

fin_z = all_df[fin_z.columns]
print fin_z.shape
fin_l = all_df[fin_l.columns]
print fin_l.shape


#multiplication:
all_df = multiply(fin_z,fin_l)




