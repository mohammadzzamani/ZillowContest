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


def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = (df["transactiondate"].dt.year - 2016)*12 + df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016)*4 +df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df