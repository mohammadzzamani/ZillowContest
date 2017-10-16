import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class transformation:

    tr_mean = 0
    tr_var = 1

    def transform(self,data, column):
        self.tr_mean = data[column].mean()
        self.tr_var = data[column].var()

        data[column] = data[column].map(lambda x: (x - self.tr_mean)/ self.tr_var )

    def transform_back(self, values):
        values = (values * self.tr_var ) + self.tr_mean
        return values



def stack_folds_preds( pred_fold, pred_all=None, axis=0):
    print ('stack_folds_preds...')
    if pred_all is None:
        pred_all = pred_fold
    else:
        # if axis==0:
        pred_all = np.vstack((pred_all, pred_fold)) if axis==0 else np.hstack((pred_all, pred_fold))
        # else:
        #     pred_all = np.vstack((pred_all, pred_fold))
    return pred_all


def outlier_detection(data, column = 'logerror' , thresh=1):
    print ('outlier_detection...')
    print (data.shape)
    # print type(data)
    # data = data[np.where(data[:,0]<0.9)]
    data = data[abs(data[column] ) < thresh]
    print (data.shape)
    return data


def into_classes(data):
    # print 'data: ' , data
    if abs(data)<=1:
        return 0
    elif data<-1:
        return -1
    else: return 1
    # mid = data[abs(data.logerror)<=1]
    # low = data[data.logerror< -1]
    # high = data[data.logerror> 1]
    # print low.shape, ' , ', mid.shape, ' , ', high.shape



class mean_est:
    def __init__(self,type='regression'):
        self.type = type
        self.mean = None

    def fit(self, X, y):
        print ('fit: ' , X.shape ,  '  , ' , y.shape)
        self.mean = np.mean(y)
        if self.type is 'classification':
            self.mean = np.sign(self.mean)

    def predict(self, X):
        print ('predict: ' , X.shape )
        return np.array([ self.mean for i in range(X.shape[0])])



# def evaluate(Ytrue, Ypred, type='regression',  pre = 'pre ', mea=None, va=None):
#     if not mea is None:
#         Ytrue = transform_back(Ytrue, mea, va)
#         Ypred = transform_back(Ypred, mea, va)
#
#     mae = mean_absolute_error(Ytrue,Ypred)
#     mse = mean_squared_error(Ytrue,Ypred)
#     with open("res.txt", "a") as myfile:
#
#         if type is 'regression':
#             myfile.write(pre + 'mae: ' + str(mae)+ ' , mse: ' + str(mse) + ' \n' )
#             print ('mae: ' , mae, ' , mse: ', mse)
#         elif type is 'classification2':
#             print ('accuracy: ' , (2-mae)/2)
#     return [mae , mse]

def evaluate(Ytrue, Ypred, type='regression',  pre = 'pre ', trnsfrm = None):
    print ('evaluate...')
    if not trnsfrm is None:
        Ytrue = trnsfrm.transform_back(Ytrue)
        Ypred = trnsfrm.transform_back(Ypred)

    # print ('before mae')
    mae = mean_absolute_error(Ytrue,Ypred)
    mse = mean_squared_error(Ytrue,Ypred)
    # print ('mae: '  , mae)
    with open("res.txt", "a") as myfile:
        if type is 'regression':
            # print ('type: ' , type)
            myfile.write(pre + 'mae: ' + str(mae)+ ' , mse: ' + str(mse) + ' \n' )
            print (pre, ' mae: ' , mae, ' , mse: ', mse)
        elif type is 'classification2':
            print ('accuracy: ' , (2-mae)/2)
    return [mae , mse]


def cats_to_int(data):
        print ('cats_to_int...')
        cat_columns = data.select_dtypes(['category','object']).columns
        print ('cat_columns: ' , cat_columns)

        for col in cat_columns:
            print ('col:  ' , col)
            data[col] = pd.Categorical(data[col])
            data[col] = data[col].astype('category').cat.codes
        return data


def add_date_features(df, drop_transactiondate=True):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = (df["transactiondate"].dt.year - 2016)*12 + df["transactiondate"].dt.month
    # df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016)*4 +df["transactiondate"].dt.quarter
    if drop_transactiondate:
        df.drop(["transactiondate"], inplace=True, axis=1)
    return df

def get_submission_format(data):
    print ('get_submission_format ...')
    data = data[['parcelid']]
    print (data)
    cols = ['ParcelId', '10/1/16', '11/1/16', '12/1/16', '10/1/17', '11/1/17', '12/1/17']
    for i in range(1 ,len(cols)):
        c = cols[i]
        data[c] = 0
    data.columns = cols
    print (data.columns)
    print (data.shape)
    submission_df = pd.melt(data, id_vars=["parcelid"],var_name="transactiondate", value_name="logerror")
    print ('submission_df: ')
    print (submission_df)
    print (submission_df.shape)
    return submission_df

def prepare_final_submission(submission_df, Ypred, type= 0):
    ##### prepare submission dataframe to look like the actual submission file (using pivot_table)
    submission_df['logerror'] = Ypred
    submission_df = submission_df[['logerror']]

    # if ('Date' in submission_df.columns):
    if type == 0:
        submission_df.reset_index(inplace=True)
        print (submission_df.iloc[1:50, :])


        submission_df = submission_df.pivot_table(values='logerror', index='parcelid', columns='transactiondate')

        submission_df.reset_index(inplace=True)

        submission_df.columns = ['ParcelId' , '201610' , '201710', '201611', '201711', '201612', '201712']

        submission_df = submission_df[['ParcelId' , '201610' ,  '201611', '201612', '201710','201711', '201712' ]]

        submission_df.set_index('ParcelId', inplace=True)


    else:
        cols = ['201610' , '201611', '201612', '201710', '201711', '201712']
        for i in range(len(cols)):
            c = cols[i]
            submission_df[c] = submission_df['logerror']
        submission_df = submission_df[cols]


    print ('final_submission_df.shape: ' , submission_df.shape)
    print ('final_submission_df.columns: ' , submission_df.columns)
    print (submission_df)
    final_submission_name = 'data/final_submission_outlierDetection.csv'

    submission_df.to_csv(final_submission_name)