#This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import pylab
import calendar
#import seaborn as sn
from scipy import stats
#import missingno as msno
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import warnings

from sklearn.svm import SVR
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn import linear_model

#matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")
#matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn import preprocessing
from sklearn import tree

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import random
# import seaborn as sns
# color = sns.color_palette()
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB

import heapq
import math
import operator

class Zillow:
    data_path = '/Users/Mz/Google_Drive/university/research/zillow/data/'
    def check_nan(self,df, exempt_list, start,end, nan_weights = []):
        n_df = df.copy()
        nan_count = []
        for column in n_df.columns:
            if column not in exempt_list:
                n_df[column] = n_df[column].isnull()
                nan_count.append(np.sum(n_df[column]))
        if len(nan_weights) <= 0:
            nan_weights = [ 100.0/(i+1) for i in nan_count]
        n_df['nan_sum'] = n_df[n_df.columns.values[start:end]].sum(axis=1)
        n_df['nan_wsum'] = n_df[n_df.columns.values[start:end]].dot(nan_weights)
        return n_df, nan_weights

    def regional(self,region_train_df, region_test_df, C , gamma):
        import numpy as np
        from sklearn.svm import SVR
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_absolute_error

        X_tr = region_train_df.values[:,:2]
        y_tr = region_train_df.values[:,2]

        X_te = region_test_df.values[:,:2]
        y_te = region_test_df.values[:,2]

        print (len(X_tr) , ', ', len(X_tr[0]) , ' , ', len(y_tr))
        print (len(X_te) , ', ', len(X_te[0]) , ' , ', len(y_te))

        svr_rbf = SVR(kernel='rbf', C=C, gamma=gamma)
        #svr_lin = SVR(kernel='linear', C=1e-2)
        #svr_poly = SVR(kernel='poly', C=1e-2, degree=2)
        print (svr_rbf.C , ' , ', svr_rbf.gamma )

        model = svr_rbf.fit(X_tr, y_tr)
        y_rbf = model.predict(X_te)

        print (len(y_rbf), ' , ',len(X_te[:,0]), ' , ',  len(X_te[:,1]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_te[:,0], X_te[:,1], y_rbf, marker='o', color='navy', label='rbf')
        ax.scatter(X_te[:,0], X_te[:,1], y_te, color='darkorange', label='data')

        error = mean_absolute_error(y_rbf, y_te)
        print ( 'error: ' , error )

        return y_rbf


    def get_avg(self, x, y):
        maximum = int(np.max(x)) +2
        buckets = [ [] for i in xrange(maximum)]
        buckets_avg = [ 0 for i in xrange(maximum)]
        for i in xrange(maximum):
            buckets[i] = [ y[j] for j in xrange(len(x)) if x[j] == i]
        for i in xrange(maximum):
            buckets_avg[i] = np.mean(buckets[i]) if len(buckets[i]) > 0 else 0
        return buckets_avg

    def decision_tree(self):


        #### latitude & longitude analysis
        # self.train['abs_logerror'] = self.train.logerror.map(lambda x: np.abs(x))
        # self.train['sign_logerror'] = self.train.logerror.map(lambda x: np.sign(x))
        # self.test['abs_logerror'] = self.test.logerror.map(lambda x: np.abs(x))
        # self.test['sign_logerror'] = self.test.logerror.map(lambda x: np.sign(x))
        #
        # region_train = self.train[[ 'latitude', 'longitude', 'abs_logerror']]
        # region_test = self.test[[ 'latitude', 'longitude', 'abs_logerror']]
        #
        # region_train = region_train.dropna()
        # region_test = region_test.dropna()
        #
        # y_pr = self.regional(region_train, region_test, 1e-4, 0.001)


        # temp = self.merged.copy()
        # self.merged = self.merged[np.abs(self.merged.logerror) > 0.03 ]
        # self.merged = self.merged[self.merged.logerror < 0.00001 ]

        #####least null features{

        for f in self.merged.columns:
            if self.merged[f].dtype=='object':
                print 'f: ', f
                self.merged[f] = self.merged.fillna(0)
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(self.merged[f].values))
                self.merged[f] = lbl.transform(list(self.merged[f].values))
        train = self.merged[self.merged.transactiondate<'2016-10-01']
        test = self.merged[self.merged.transactiondate>'2016-09-30' ]


        mytrain = train[['parcelid', 'transactiondate', 'logerror', 'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'fips', 'latitude', 'longitude', 'propertycountylandusecode', 'propertylandusetypeid', 'rawcensustractandblock', 'regionidcounty', 'regionidzip', 'roomcnt', 'yearbuilt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount', 'censustractandblock'  ]]
        mytest = test[['parcelid', 'transactiondate', 'logerror', 'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'fips', 'latitude', 'longitude', 'propertycountylandusecode', 'propertylandusetypeid', 'rawcensustractandblock', 'regionidcounty', 'regionidzip', 'roomcnt', 'yearbuilt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount', 'censustractandblock'  ]]
        mytrain.set_index(['parcelid', 'transactiondate'], inplace = True)
        mytest.set_index(['parcelid', 'transactiondate'], inplace = True)
        mytrain = mytrain.dropna()
        mytest = mytest.dropna()

        train.set_index(['parcelid', 'transactiondate'], inplace = True)
        test.set_index(['parcelid', 'transactiondate'], inplace = True)


        print 'mytrain and mytest shapes:'
        print mytrain.shape
        print mytest.shape

        ytr = mytrain.logerror.values
        ytr_abs = mytrain.logerror.map(lambda x: np.abs(x))
        ytr_sign = mytrain.logerror.map(lambda x: np.sign(x))
        Xtr = mytrain.drop([ "logerror"], axis=1)

        yte = mytest.logerror.values
        yte_abs = mytest.logerror.map(lambda x: np.abs(x))
        yte_sign = mytest.logerror.map(lambda x: np.sign(x))
        Xte = mytest.drop(["logerror"], axis=1)
        ######}



        ypr1, ypr2= self.DT('regression','reg_tree.dot',Xtr, ytr, Xte, yte)
        ypr1_abs, ypr2_abs = self.DT('regression', 'reg_abs_tree.dot',Xtr,ytr_abs, Xte, yte_abs)
        ypr1_sign , ypr2_sign = self.DT('classification','clf_sign_tree.dot', Xtr, ytr_sign, Xte, yte_sign)

        ypr = [a*b for a,b in zip(ypr1_abs,ypr1_sign)]
        error = mean_absolute_error(yte, ypr)
        print 'final error1: ' , error


        ypr = self.regression('ridge_regression',Xtr, ytr, Xte, yte)
        ypr_abs = np.abs(self.regression('ridge_regression', Xtr,ytr_abs, Xte, yte_abs))
        ypr_sign = self.regression('classification', Xtr, ytr_sign, Xte, yte_sign)

        ypr = [a*b for a,b in zip(ypr_abs,ypr_sign)]
        error = mean_absolute_error(yte, ypr)
        print 'final error2: ' , error

        ypr = [a*b for a,b in zip(ypr1_abs,ypr_sign)]
        error = mean_absolute_error(yte, ypr)
        print 'final error mixed: ' , error


        ''' '''
        # self.merged = temp
        # self.merged = self.merged[np.abs(self.merged.logerror) > 0.03 ]
        # self.merged = self.merged[self.merged.logerror < 0.00001 ]

        train_n, nan_weights = self.check_nan(self.train, ['parcelid', 'transactiondate', 'logerror'], 3, self.train.shape[1], [])
        test_n ,nan_weights = self.check_nan(self.test, ['parcelid', 'transactiondate', 'logerror'], 3, self.test.shape[1], nan_weights)

        self.merged = self.merged.fillna(-1000)
        for f in self.merged.columns:
            if self.merged[f].dtype=='object':
                # self.merged[f] = self.merged.fillna(-1000)
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(self.merged[f].values))
                self.merged[f] = lbl.transform(list(self.merged[f].values))
            # elif self.merged[f].dtype <> 'datetime64[ns]':
            #     # print 'f: ' , f,  ' , ', self.merged[f].dtype
            #     self.merged[f] = self.merged.fillna(np.mean(self.merged[f].values))

        self.train = self.merged[self.merged.transactiondate<'2016-10-01']
        self.test = self.merged[self.merged.transactiondate>'2016-09-30' ]



        train_n.drop('logerror', axis=1, inplace=True)
        test_n.drop('logerror', axis=1, inplace=True)
        full_train  = pd.merge(self.train , train_n, on=['parcelid', 'transactiondate'], how='left', suffixes=('_r', '_b'))
        full_train.set_index(['parcelid', 'transactiondate'], inplace = True)
        full_test  = pd.merge(self.test , test_n, on=['parcelid', 'transactiondate'], how='left', suffixes=('_r', '_b'))
        full_test.set_index(['parcelid', 'transactiondate'], inplace = True)
        remaining_test = full_test[~ full_test.index.isin(mytest.index)]
        print 'remaining_test.shape: ' , remaining_test.shape
        full_test = remaining_test
        # full_train = self.train
        # full_test = self.test
        print ('full_train.shape: ', full_train.shape)
        print ('full_test.shape: ' , full_test.shape)
        # print (full_train.columns.values)

        ytr = full_train.logerror.values
        ytr_abs = full_train.logerror.map(lambda x: np.abs(x))
        ytr_sign = full_train.logerror.map(lambda x: np.sign(x))
        Xtr = full_train.drop([ "logerror"], axis=1)

        yte = full_test.logerror.values
        yte_abs = full_test.logerror.map(lambda x: np.abs(x))
        yte_sign = full_test.logerror.map(lambda x: np.sign(x))
        Xte = full_test.drop(["logerror"], axis=1)


        ypr1, ypr2= self.DT('regression','reg_tree.dot',Xtr, ytr, Xte, yte)
        ypr1_abs, ypr2_abs = self.DT('regression', 'reg_abs_tree.dot',Xtr,ytr_abs, Xte, yte_abs)
        ypr1_abs = np.abs(ypr1_abs)
        ypr2_abs = np.abs(ypr2_abs)
        ypr1_sign , ypr2_sign = self.DT('classification','clf_sign_tree.dot', Xtr, ytr_sign, Xte, yte_sign)

        ypr = [a*b for a,b in zip(ypr1_abs,ypr1_sign)]
        error = mean_absolute_error(yte, ypr)
        print 'final error1: ' , error


        ypr = self.regression('ridge_regression',Xtr, ytr, Xte, yte)
        ypr_abs = np.abs(self.regression('ridge_regression', Xtr,ytr_abs, Xte, yte_abs))
        ypr_sign = self.regression('classification', Xtr, ytr_sign, Xte, yte_sign)

        ypr = [a*b for a,b in zip(ypr_abs,ypr_sign)]
        error = mean_absolute_error(yte, ypr)
        print 'final error2: ' , error

        ypr = [a*b for a,b in zip(ypr1_abs,ypr_sign)]
        error = mean_absolute_error(yte, ypr)
        print 'final error mixed: ' , error


        full_test['2016_09'] = ypr





    def partially_linear(self):

        # temp = self.merged.copy()
        # self.merged = self.merged[np.abs(self.merged.logerror) > 0.03 ]
        # self.merged = self.merged[self.merged.logerror > 0.00001 ]

        train_n, nan_weights = self.check_nan(self.train, ['parcelid', 'transactiondate', 'logerror'], 3, self.train.shape[1], [])
        test_n ,nan_weights = self.check_nan(self.test, ['parcelid', 'transactiondate', 'logerror'], 3, self.test.shape[1], nan_weights)

        # self.merged = self.merged.fillna(-1000)
        mean_values = self.merged.mean(axis=0)
        # print mean_values
        self.merged  = self.merged.fillna(mean_values, inplace=True)
        for f in self.merged.columns:
            if self.merged[f].dtype=='object':
                # print 'f: ' , f
                self.merged[f] = self.merged.fillna(0)
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(self.merged[f].values))
                self.merged[f] = lbl.transform(list(self.merged[f].values))
            # elif self.merged[f].dtype <> 'datetime64[ns]':
            #     # print 'f: ' , f,  ' , ', self.merged[f].dtype
            #     self.merged[f] = self.merged.fillna(np.mean(self.merged[f].values))

        self.train = self.merged[self.merged.transactiondate<'2016-10-01']
        self.test = self.merged[self.merged.transactiondate>'2016-09-30' ]

        train_n.drop('logerror', axis=1, inplace=True)
        test_n.drop('logerror', axis=1, inplace=True)
        full_train  = pd.merge(self.train , train_n, on=['parcelid', 'transactiondate'], how='left', suffixes=('_r', '_b'))
        full_train.set_index(['parcelid', 'transactiondate'], inplace = True)
        full_test  = pd.merge(self.test , test_n, on=['parcelid', 'transactiondate'], how='left', suffixes=('_r', '_b'))
        full_test.set_index(['parcelid', 'transactiondate'], inplace = True)
        # remaining_test = full_test[~ full_test.index.isin(mytest.index)]
        print ('full_train.shape: ', full_train.shape)
        print ('full_test.shape: ' , full_test.shape)

        ytr = full_train.logerror.values
        ytr_abs = full_train.logerror.map(lambda x: np.abs(x))
        ytr_sign = full_train.logerror.map(lambda x: np.sign(x))
        Xtr = full_train.drop([ "logerror"], axis=1)

        yte = full_test.logerror.values
        yte_abs = full_test.logerror.map(lambda x: np.abs(x)).values
        yte_sign = full_test.logerror.map(lambda x: np.sign(x)).values
        Xte = full_test.drop(["logerror"], axis=1)


        ypr1, ypr2= self.DT('regression','reg_tree.dot',Xtr, ytr, Xte, yte)
        ypr1_abs, ypr2_abs = self.DT('regression', 'reg_abs_tree.dot',Xtr,ytr_abs, Xte, yte_abs)
        ypr1_abs = np.abs(ypr1_abs)
        ypr2_abs = np.abs(ypr2_abs)
        ypr1_sign  = self.DT_classification('classification','clf_sign_tree.dot', Xtr, ytr_sign, Xte, yte_sign)
        # ypr1_sign , ypr2_sign = self.DT('classification','clf_sign_tree.dot', Xtr, ytr_sign, Xte, yte_sign)

        ypr = [a*b for a,b in zip(ypr1_abs,ypr1_sign)]
        error = mean_absolute_error(yte, ypr)
        print 'final error1: ' , error , ' \n\n\n'

        ypr = self.regression('linear_regression',Xtr, ytr, Xte, yte)
        ypr_abs = np.abs(self.regression('linear_regression', Xtr,ytr_abs, Xte, yte_abs))
        # ypr_sign = self.classification('classification', Xtr, ytr_sign, Xte, yte_sign)
        ypr_sign , ypr2_sign = self.DT('classification','clf_sign_tree.dot', Xtr, ytr_sign, Xte, yte_sign)

        ypr_sign , ypr2_sign = self.regr('classification','clf_sign_tree.dot', Xtr, ytr_sign, Xte, yte_sign)

        ypr = [a*b for a,b in zip(ypr_abs,ypr_sign)]
        error = mean_absolute_error(yte, ypr)
        print 'final error2: ' , error

        # ypr = [a*b for a,b in zip(ypr_abs,ypr1_sign)]
        # error = mean_absolute_error(yte, ypr)
        # print 'final error mixed: ' , error

        # full_test['2016_09'] = ypr

    def partially_linear_3(self):

        # temp = self.merged.copy()
        # self.merged = self.merged[np.abs(self.merged.logerror) > 0.03 ]
        # self.merged = self.merged[self.merged.logerror > 0.00001 ]

        train_n, nan_weights = self.check_nan(self.train, ['parcelid', 'transactiondate', 'logerror'], 3, self.train.shape[1], [])
        test_n ,nan_weights = self.check_nan(self.test, ['parcelid', 'transactiondate', 'logerror'], 3, self.test.shape[1], nan_weights)

        # self.merged = self.merged.fillna(-1000)
        mean_values = self.merged.mean(axis=0)
        # print mean_values
        self.merged  = self.merged.fillna(mean_values, inplace=True)
        for f in self.merged.columns:
            if self.merged[f].dtype=='object':
                # print 'f: ' , f
                self.merged[f] = self.merged.fillna(0)
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(self.merged[f].values))
                self.merged[f] = lbl.transform(list(self.merged[f].values))
            # elif self.merged[f].dtype <> 'datetime64[ns]':
            #     # print 'f: ' , f,  ' , ', self.merged[f].dtype
            #     self.merged[f] = self.merged.fillna(np.mean(self.merged[f].values))

        self.train = self.merged[self.merged.transactiondate<'2016-10-01']
        self.test = self.merged[self.merged.transactiondate>'2016-09-30' ]

        train_n.drop('logerror', axis=1, inplace=True)
        test_n.drop('logerror', axis=1, inplace=True)
        full_train  = pd.merge(self.train , train_n, on=['parcelid', 'transactiondate'], how='left', suffixes=('_r', '_b'))
        full_train.set_index(['parcelid', 'transactiondate'], inplace = True)
        full_test  = pd.merge(self.test , test_n, on=['parcelid', 'transactiondate'], how='left', suffixes=('_r', '_b'))
        full_test.set_index(['parcelid', 'transactiondate'], inplace = True)
        # remaining_test = full_test[~ full_test.index.isin(mytest.index)]
        print ('full_train.shape: ', full_train.shape)
        print ('full_test.shape: ' , full_test.shape)

        # ytr = full_train.logerror.values
        # ytr_abs = full_train.logerror.map(lambda x: np.abs(x))
        ytr_sign = full_train.logerror.map(lambda x: np.sign(x))
        Xtr = full_train.drop([ "logerror"], axis=1)

        # yte = full_test.logerror.values
        # yte_abs = full_test.logerror.map(lambda x: np.abs(x)).values
        yte_sign = full_test.logerror.map(lambda x: np.sign(x)).values
        Xte = full_test.drop(["logerror"], axis=1)


        # ypr1, ypr2= self.DT('regression','reg_tree.dot',Xtr, ytr, Xte, yte)
        # ypr1_abs, ypr2_abs = self.DT('regression', 'reg_abs_tree.dot',Xtr,ytr_abs, Xte, yte_abs)
        # ypr1_abs = np.abs(ypr1_abs)
        # ypr2_abs = np.abs(ypr2_abs)
        ypr_sign  = self.DT_classification('classification','clf_sign_tree.dot', Xtr, ytr_sign, Xte, yte_sign)
        # ypr1_sign , ypr2_sign = self.DT('classification','clf_sign_tree.dot', Xtr, ytr_sign, Xte, yte_sign)

        full_test['pred_sign'] = ypr_sign
        full_train_positive = full_train[full_train.logerror >= 0]
        full_train_negative = full_train[full_train.logerror < 0]
        full_test_positive = full_test[full_test.pred_sign >=0]
        full_test_negative = full_test[full_test.pred_sign < 0]

        ytr = full_train_positive.logerror.values
        ytr_abs = full_train_positive.logerror.map(lambda x: np.abs(x))
        Xtr = full_train_positive.drop([ "logerror" ], axis=1)

        yte = full_test_positive.logerror.values
        yte_abs = full_test_positive.logerror.map(lambda x: np.abs(x)).values
        Xte = full_test_positive.drop(["logerror", 'pred_sign'], axis=1)

        ypr_abs = np.abs(self.regression('linear_regression', Xtr,ytr_abs, Xte, yte_abs))
        ypr = ypr_abs
        error = mean_absolute_error(yte, ypr)
        print 'error1: ' , error



        full_test['pred_sign'] = ypr_sign
        full_train_negative = full_train[full_train.logerror >= 0]
        full_train_negative = full_train[full_train.logerror < 0]
        full_test_negative = full_test[full_test.pred_sign >=0]
        full_test_negative = full_test[full_test.pred_sign < 0]

        ytr = full_train_negative.logerror.values
        ytr_abs = full_train_negative.logerror.map(lambda x: np.abs(x))
        Xtr = full_train_negative.drop([ "logerror"], axis=1)

        yte = full_test_negative.logerror.values
        yte_abs = full_test_negative.logerror.map(lambda x: np.abs(x)).values
        Xte = full_test_negative.drop(["logerror",'pred_sign'], axis=1)

        ypr_abs = np.abs(self.regression('linear_regression', Xtr,ytr_abs, Xte, yte_abs))
        ypr = ypr_abs * -1.0
        error = mean_absolute_error(yte, ypr)
        print 'error2: ' , error

    def partially_linear_2(self):

        # temp = self.merged.copy()
        # self.merged = self.merged[np.abs(self.merged.logerror) > 0.03 ]
        # self.merged = self.merged[self.merged.logerror > 0.00001 ]

        train_n, nan_weights = self.check_nan(self.train, ['parcelid', 'transactiondate', 'logerror'], 3, self.train.shape[1], [])
        test_n ,nan_weights = self.check_nan(self.test, ['parcelid', 'transactiondate', 'logerror'], 3, self.test.shape[1], nan_weights)

        # self.merged = self.merged.fillna(-1000)
        mean_values = self.merged.mean(axis=0)
        print mean_values
        self.merged  = self.merged.fillna(mean_values, inplace=True)
        for f in self.merged.columns:
            if self.merged[f].dtype=='object':
                # print 'f: ' , f
                self.merged[f] = self.merged.fillna(0)
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(self.merged[f].values))
                self.merged[f] = lbl.transform(list(self.merged[f].values))
            # elif self.merged[f].dtype <> 'datetime64[ns]':
            #     # print 'f: ' , f,  ' , ', self.merged[f].dtype
            #     self.merged[f] = self.merged.fillna(np.mean(self.merged[f].values))

        self.train = self.merged[self.merged.transactiondate<'2016-10-01']
        self.test = self.merged[self.merged.transactiondate>'2016-09-30' ]

        train_n.drop('logerror', axis=1, inplace=True)
        test_n.drop('logerror', axis=1, inplace=True)
        full_train  = pd.merge(self.train , train_n, on=['parcelid', 'transactiondate'], how='left', suffixes=('_r', '_b'))
        full_train.set_index(['parcelid', 'transactiondate'], inplace = True)
        full_test  = pd.merge(self.test , test_n, on=['parcelid', 'transactiondate'], how='left', suffixes=('_r', '_b'))
        full_test.set_index(['parcelid', 'transactiondate'], inplace = True)
        # remaining_test = full_test[~ full_test.index.isin(mytest.index)]
        print ('full_train.shape: ', full_train.shape)
        print ('full_test.shape: ' , full_test.shape)

        ytr = full_train.logerror.values
        ytr_abs = full_train.logerror.map(lambda x: np.abs(x))
        ytr_sign = full_train.logerror.map(lambda x: np.sign(x))
        Xtr = full_train.drop([ "logerror"], axis=1)

        yte = full_test.logerror.values
        yte_abs = full_test.logerror.map(lambda x: np.abs(x))
        yte_sign = full_test.logerror.map(lambda x: np.sign(x))
        Xte = full_test.drop(["logerror"], axis=1)


        ypr1, ypr2= self.DT('regression','reg_tree.dot',Xtr, ytr, Xte, yte)
        ypr1_abs, ypr2_abs = self.DT('regression', 'reg_abs_tree.dot',Xtr,ytr_abs, Xte, yte_abs)
        ypr1_abs = np.abs(ypr1_abs)
        ypr2_abs = np.abs(ypr2_abs)
        ypr1_sign , ypr2_sign = self.DT('classification','clf_sign_tree.dot', Xtr, ytr_sign, Xte, yte_sign)

        ypr = [a*b for a,b in zip(ypr1_abs,ypr1_sign)]
        error = mean_absolute_error(yte, ypr)
        print 'final error1: ' , error


        ypr = self.regression('linear_regression',Xtr, ytr, Xte, yte)
        ypr_abs = np.abs(self.regression('linear_regression', Xtr,ytr_abs, Xte, yte_abs))
        ypr_sign = self.classification('classification', Xtr, ytr_sign, Xte, yte_sign)

        ypr = [a*b for a,b in zip(ypr_abs,ypr_sign)]
        error = mean_absolute_error(yte, ypr)
        print 'final error2: ' , error

        ypr = [a*b for a,b in zip(ypr1_abs,ypr_sign)]
        error = mean_absolute_error(yte, ypr)
        print 'final error mixed: ' , error

        # full_test['2016_09'] = ypr


    def DT_classification(self, type , filename, Xtr , ytr, Xte, yte):
        ypr = []
        models = []
        params= [ [4, 50 , 5],  [ 6, 500 , 10]  , [ 10 , 500 , 50] ] #[4, 500 , 50] , [4, 200 , 1], [ 5, 50 , 5] , [5, 100, 5]  ,  [ 6, 100, 5] , [ 6, 100, 10] , [6 , 200, 10] , [7,100, 10] , [7 , 200, 10] , [8 , 200 , 20], [8, 500 , 20] , [8, 500 , 50] , [9 , 500, 50] , [9 , 500 , 100] , [10 , 500 , 50] , [ 10 , 500 , 100]]
        for p in params:
            models.append( DecisionTreeClassifier(max_depth=p[0], min_samples_split = p[1], min_samples_leaf=p[2]) )
            #   model_2 = DecisionTreeClassifier(criterion='entropy',max_depth=10, min_samples_split=500, min_samples_leaf=100)
        models.append( RandomForestClassifier(random_state=1) )
        # models.append( RandomForestClassifier(random_state=1, n_estimators= 20) )
        models.append( RandomForestClassifier(random_state=1, n_estimators= 20, min_samples_split = 50) )
        models.append(  GaussianNB())
        clf = VotingClassifier(estimators=[ (str(i) , models[i]) for i in xrange(len(models))], voting='soft')
            #fit
        clf.fit(Xtr, ytr)
        # tree.export_graphviz(model,out_file=filename)

            # Predict
        ypr = clf.predict(Xte)

        print 'ypr: ' , ypr[:20]
        print 'yte; ' , yte[:20]

        # for i in xrange(len(ypr)):
        # ypr_sum = np.sum(ypr,axis=0)
        # print len(ypr_sum)

        # ypr = [ ]
        print 'yte.shape: ' , yte.shape
        print 'ypr.shape: ' , ypr.shape
        #error rate
        error = mean_absolute_error(yte, ypr)

        print ' score: ', clf.score(Xte, yte)


        print 'errors: ', error
        #
        return ypr

    def DT(self , type , filename, Xtr , ytr, Xte, yte):

        #define learners
        rng = np.random.RandomState(1)
        if type == 'regression':
            model_1 = DecisionTreeRegressor(criterion = 'mse',max_depth=4, min_samples_split= 50)
            model_2 = DecisionTreeRegressor(criterion= 'mse', max_depth=10, min_samples_split = 500, min_samples_leaf=100)
            # model_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng)
        else:
            model_1 = DecisionTreeClassifier(max_depth=4, min_samples_split = 50)
            model_2 = DecisionTreeClassifier(criterion='entropy',max_depth=10, min_samples_split=500, min_samples_leaf=100)

        #fit
        model_1.fit(Xtr, ytr)
        model_2.fit(Xtr, ytr)
        tree.export_graphviz(model_1,out_file=filename)

        # Predict
        ypr1 = model_1.predict(Xte)
        ypr2 = model_2.predict(Xte)

        # if type <> 'regression':
        #     print 'preds: ' , ypr1[:20], ' , ', ypr2[:20]
        #error rate
        error1 = mean_absolute_error(yte, ypr1)
        error2 = mean_absolute_error(yte, ypr2)

        print 'errors: ', error1 , ' , ', error2

        return ypr1, ypr2

    def regional(self, train, test, C , gamma):

        X_tr = train.values[:,:2]
        y_tr = train.values[:,2]

        X_te = test.values[:,:2]
        y_te = test.values[:,2]

        print (len(X_tr) , ', ', len(X_tr[0]) , ' , ', len(y_tr))
        print (len(X_te) , ', ', len(X_te[0]) , ' , ', len(y_te))

        svr_rbf = SVR(kernel='rbf', C=C, gamma=gamma)
        #svr_lin = SVR(kernel='linear', C=1e-2)
        #svr_poly = SVR(kernel='poly', C=1e-2, degree=2)
        print (svr_rbf.C , ' , ', svr_rbf.gamma )

        model = svr_rbf.fit(X_tr, y_tr)
        y_rbf = model.predict(X_te)

        print (len(y_rbf), ' , ',len(X_te[:,0]), ' , ',  len(X_te[:,1]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_te[:,0], X_te[:,1], y_rbf, marker='o', color='navy', label='rbf')
        ax.scatter(X_te[:,0], X_te[:,1], y_te, color='darkorange', label='data')

        error = mean_absolute_error(y_rbf, y_te)
        print ( 'rbf error : ' , error )

        return y_rbf




    def plot_nan_sum(self):
        # self.merged = self.merged[np.abs(self.merged.logerror) > 0.05 ]
        print 'self.merged.shape: ', self.merged.shape
        merged_n, nan_weights = self.check_nan(self.merged, ['parcelid', 'transactiondate', 'logerror'], 3, self.merged.shape[1], [])
        #full_test_n_df = self.make_dataset(full_test_df, ['ParcelId' ,'logerror', 'transactiondate'], 3, full_test_df.shape[1])
        merged_n['abs_logerror'] = merged_n.logerror.map(lambda x: np.abs(x))
        merged_n['sign_logerror'] = merged_n.logerror.map(lambda x: np.sign(x))


        print 'greater than 32: ' , merged_n[merged_n.nan_sum>32 ].shape
        print 'greater than 32: ' , merged_n[merged_n.nan_sum>50 ].shape

        y = np.array(merged_n.abs_logerror)
        x = np.array(merged_n.nan_sum)
        fig = plt.figure()
        plt.plot(x, y, 'b+')
        buckets_avg = self.get_avg(x,y)
        print 'abs_sum: ', buckets_avg
        x = np.arange(len(buckets_avg))
        plt.plot(x, buckets_avg, 'r+')
        plt.savefig('abs_sum.png')



        y = np.array(merged_n.logerror)
        x = np.array(merged_n.nan_sum)
        fig = plt.figure()
        plt.plot(x, y, 'b+')
        buckets_avg = self.get_avg(x,y)
        print 'error_sum: ', buckets_avg
        x = np.arange(len(buckets_avg))
        plt.plot(x, buckets_avg, 'r+')
        plt.savefig('error_sum.png')



        y = np.array(merged_n.sign_logerror)
        x = np.array(merged_n.nan_sum)
        fig = plt.figure()
        plt.plot(x, y, 'b+')
        buckets_avg_s = self.get_avg(x,y)
        print 'sign_sum: ', buckets_avg_s
        x = np.arange(len(buckets_avg_s))
        plt.plot(x, buckets_avg_s, 'r+')
        plt.savefig('sign_sum.png')



        x = np.array(merged_n.nan_sum)
        yte = np.array(merged_n.logerror)
        ypr = [ 0 for i in xrange(len(x))]
        for i in xrange(len(x)):
            r = random.random()*2 -1
            if r < buckets_avg_s[x[i]]:
                ypr[i] = buckets_avg[x[i]]
            else:
                ypr[i] = -1 * buckets_avg[x[i]]

        error = mean_absolute_error(yte, ypr)
        print ' size: ' , len(ypr) , ' error: ', error


        y = np.array(merged_n.abs_logerror)
        x = np.array(merged_n.nan_wsum)
        fig = plt.figure()
        axes = plt.gca()
        axes.set_xlim([0,5])
        plt.plot(x, y, 'b+')

        buckets_avg = self.get_avg(x,y)
        print 'abs_wsum: ', buckets_avg
        x = np.arange(len(buckets_avg))
        axes.set_xlim([0,5])
        plt.plot(x, buckets_avg, 'r+')
        plt.savefig('abs_wsum.png')


        # col = 'nan_sum'
        # ulimit = np.percentile(merged_n[col].values, 99.5)
        # llimit = np.percentile(merged_n[col].values, 0.5)
        # merged_n[col].ix[merged_n[col]>ulimit] = ulimit
        # merged_n[col].ix[merged_n[col]<llimit] = llimit
        #
        # fig = plt.figure(figsize=(12,12))
        # sns.jointplot(x=merged_n[col].values, y=merged_n.logerror.values, size=10, kind="hex",color="#34495e")
        # plt.ylabel('Log Error', fontsize=12)
        # plt.xlabel(col, fontsize=12)
        # plt.title((col +' Vs Log error'), fontsize=15)
        # # plt.show()
        # # plt.plot(x, buckets_avg, 'r+')
        # plt.savefig('nan_sum vs logerror.png')


    def regression(self, type, Xtr, ytr, Xte, yte):

        print '<<< ' , type , ' >>>'
        if type == 'ridge_regression':
            cvParams = {'ridgecv': [{'alphas': np.array([1, .1, .01, .001, .0001, 10, 100, 1000, 10000, 100000, 100000, 1000000, 10000000, 100000000, 1000000000])}]}
            model = RidgeCV()
            model.set_params(**dict((k, v[0] if isinstance(v, list) else v) for k,v in cvParams['ridgecv'][0].iteritems()))
        elif type == 'linear_regression':
            model = linear_model.LinearRegression()
        else:
            model = linear_model.LogisticRegression(C=1)

        model.fit(Xtr, ytr)
        ypr = model.predict(Xte)

        # if type == 'ridge_regression':
        #     print ('model.alpha_: ' , model.alpha_)
        #     print ('model.coef_:  ' , model.coef_)

        if type =='classification':
            labels =  list(set(yte))
            for label in labels:
                print 'label: ' , label, ' ,  count: ' , len([x for x in yte if x == label])
            yprob = model.predict_proba(Xte)
            print yprob[:20], ' , ', ypr[:20]

        error = mean_absolute_error(yte, ypr)
        print 'error: ', error

        return ypr


    def classification(self, type, Xtr, ytr, Xte, yte):
        print 'Xtr.shape: ' , Xtr.shape
        labels =  list(set(yte))
        for label in labels:
                print 'label: ' , label, ' ,  count: ' , len([x for x in yte if x == label])
        for c in [ 0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1 , 10 , 100, 1000]:

            model = linear_model.LogisticRegression(C=c)
            model.fit(Xtr, ytr)
            ypr = model.predict(Xte)

            yprob = model.predict_proba(Xte)
            print yprob[:20], ' , ', ypr[:20]

            error = mean_absolute_error(yte, ypr)
            print 'error: ', error

        return ypr

    def init(self):
        self.data = pd.read_csv(self.data_path+'train_2016.csv', parse_dates=["transactiondate"])
        self.properties = pd.read_csv(self.data_path+ 'properties_2016.csv')
        print ("Shape Of Data: ",self.data.shape)
        print ("Shape Of Properties: ",self.properties.shape)

        self.merged = pd.merge(self.data,self.properties,on="parcelid",how="left")
        self.merged = self.merged[np.abs(self.merged['logerror']) < 1]
        # self.merged = self.merged.fillna(-1000)
        # for f in self.merged.columns:
        #     if self.merged[f].dtype=='object':
        #         lbl = preprocessing.LabelEncoder()
        #         lbl.fit(list(self.merged[f].values))
        #         self.merged[f] = lbl.transform(list(self.merged[f].values))


        self.train = self.merged[self.merged.transactiondate<'2016-10-01']
        self.test = self.merged[self.merged.transactiondate>'2016-09-30' ]
        print ("Shape Of Train: ",self.train.shape)
        print ("Shape Of Test: ",self.test.shape)
        # print ('columns: ' , self.merged.columns)



    def train_lat_long(self, train):
        print (train.shape)
        train.dropna(subset = ['latitude', 'longitude'], how = 'any', inplace = True)
        # train = train[np.abs(train['logerror']) < 0.5]
        print (train.shape)
        # for i in train.columns:
        #     print (i)
        print ('start')
        max_lat = np.max(train.latitude)
        min_lat = np.min(train.latitude)
        max_long = np.max(train.longitude)
        min_long = np.min(train.longitude)
        max_mins = {}
        max_mins['max_lat'] = max_lat
        max_mins['min_lat'] = min_lat
        max_mins['max_long'] = max_long
        max_mins['min_long'] = min_long
        nsquares = 100
        print ('start')
        lat_step = (max_lat-min_lat)*1.0/nsquares
        long_step = (max_long-min_long)*1.0/nsquares
        print (lat_step , ' , ', long_step )
        train['lat_cat'] = train.latitude.map(lambda x: int((x-min_lat)*1.0/lat_step))
        train['long_cat'] = train.longitude.map(lambda x: int((x-min_long)*1.0/long_step))
        print ('start')
        # train['sign_logerror'] = np.abs(train['logerror'])
        train['sign_logerror'] = train['logerror']
        # train['sign_logerror'] = np.sign(train['logerror'])
        train1 = train.groupby(['lat_cat', 'long_cat']).sign_logerror.mean().reset_index()
        print ('train1.size:' , train1.size)
        plt.hist(train1.sign_logerror, normed=True, bins=100)
        plt.ylabel('Probability');
        # plt.show()
        plt.savefig('hist_train1.png')
        exit()
        train2 = train.groupby(['lat_cat', 'long_cat']).sign_logerror.median().reset_index()
        print ('train2.size:' ,train2.size)
        train3 = train.groupby(['lat_cat', 'long_cat']).sign_logerror.size().reset_index()
        train3.columns = ['lat_cat', 'long_cat', 'sign_logerror']
        # print (train3.columns)
        print (train1[:10])
        print (train2[:10])
        print (train3[:10])
        print (train1.shape, ' , ', train2.shape, ' , ', train3.shape)
        train1['mea'] = train1.sign_logerror
        train1['med'] = train2.sign_logerror
        train1['siz'] = train3.sign_logerror
        print (train1[:10])
        print (train2[:10])
        print (train3[:10])
        print (train1.corr())

        plt.scatter(train1.lat_cat,train1.long_cat,c=train1.sign_logerror, alpha = 0.7, marker='s' , edgecolors='none', s = 6)
        print (train1.sign_logerror.max(),train1.sign_logerror.min() )
        # plt.show()
        plt.savefig('lat_long1.png')
        plt.scatter(train1.lat_cat,train1.long_cat,c=-1*train1.sign_logerror, alpha = 0.7, marker='s' , edgecolors='none', s = 6)
        print ('min_max:' , train1.sign_logerror.max(), ' , ' , train1.sign_logerror.min()  , ' , ', np.max(train1.sign_logerror) , ' , ', np.min(train1.sign_logerror))
        # plt.show()
        plt.savefig('lat_long-1.png')
        plt.scatter(train2.lat_cat,train2.long_cat,c=train2.sign_logerror, alpha = 0.3, marker='s' , edgecolors='none', s = 6)
        print ('min_max:' ,  train2.sign_logerror.max(), ' , ', train2.sign_logerror.min()  , ' , ', np.max(train2.sign_logerror) , ' , ', np.min(train2.sign_logerror))
        plt.savefig('lat_long2.png')
        plt.scatter(train3.lat_cat,train3.long_cat,c=train3.sign_logerror, alpha = 0.3, marker='s' , edgecolors='none', s = 6)
        print ('min_max:' ,  train3.sign_logerror.max(), ' , ', train3.sign_logerror.min()  , ' , ', np.max(train3.sign_logerror) , ' , ', np.min(train3.sign_logerror))
        plt.savefig('lat_long3.png')



        train = pd.merge(train,train1,on=['lat_cat' , 'long_cat'],how="left")
        mean = train.logerror.mean()
        median = train.logerror.median()
        print ('mean: ' , mean)
        print ( 'median: ' , median)
        print (train.shape)
        err = 0
        err2 = 0
        err3 = 0
        for index, row in train.iterrows():
            err = err +  np.abs(row['logerror'] - row['med'])
            err2 = err2 + np.abs(row['logerror'] - mean )
            err3 = err3 + np.abs(row['logerror'] - median )
        err = err/ train.shape[0]
        err2 = err2/ train.shape[0]
        err3 = err3/ train.shape[0]
        print err , ' , ', err2,  ' , ', err3




        print 'train1.columns: ', train1.columns
        trained_df = train1[['lat_cat', 'long_cat', 'mea', 'med', 'siz']]
        # print 'trained_df.columns: ', trained_df.columns
        print 'trained_df.columns: ', trained_df.columns

        print 'trained_df.shape: ', trained_df.shape
        trained_df.loc[trained_df.shape[0]] = [-1, -1, mean, median, 0]
        # for i in range(-1,100):
        #     for j in range(-1,100):
        #         lat = trained_df.lat_cat == i
        #         long = trained_df.long_cat == j
        #         if trained_df[lat & long].shape[0] <=0:
        #             trained_df.loc[trained_df.shape[0]] = [i,j, mean, median, 0]
        print 'trained_df.shape: ', trained_df.shape
        return max_mins, trained_df, mean, median


    def distance(self, row, lat, long):
        latN = row.latitude * 0.001
        longN = row.longitude * 0.001
        # print lat, ' , ', long, ' , ', latN , ' , ', longN
        # print math.sqrt( pow(lat - latN, 2) + pow(long - longN , 2) )
        return math.sqrt( pow(lat - latN, 2) + pow(long - longN , 2) )




    def knn(self, train, test):
        max_heap_size = 50
        test = test.sample(frac=1)

        mean = train.logerror.mean()
        med = train.logerror.median()

        train.dropna(subset = ['latitude', 'longitude'], how = 'any', inplace = True)
        # train = train[np.abs(train['logerror']) < 0.5]
        test.dropna(subset = ['latitude', 'longitude'], how = 'any', inplace = True)
        logerrorNMean = []
        logerrorNMed = []
        logerrorTrue = []

        for index, row in test.iterrows():
            lat = row.latitude * 0.001
            long = row.longitude * 0.001
            heap_size = 0
            heap = []
            for i , r in train.iterrows():
                dist = -1 * self.distance(r, lat, long)
                # print dist
                if heap_size < max_heap_size:
                    heapq.heappush(heap, (dist, i, r.logerror))
                    heap_size +=1
                    # print 'heapq[0][0]: ', heap[0]
                    # print heap[0][0]
                elif dist > heap[0][0]:
                    heapq.heapreplace(heap, (dist, i, r.logerror))
                # print heap

                # logerrors = [ tupl[2] for tupl in heap]
                # logerrorN = np.average(logerrors)
                # print logerrors, ' -- ' , logerrorN
                # print heap[:][0]
                # print heap[:][0][2]
                # print heap[:][:]
                # logerrorN = np.average(heap[:][2])
                # print heap[:][2]
                # print logerrorN
            logerrors = [ tupl[2] for tupl in heap]
            logerrorNmean = np.average(logerrors)
            logerrorNmed =  np.median(logerrors)

            logerrorTrue.append( row.logerror)
            logerrorNMean.append( logerrorNmean )
            logerrorNMed.append( logerrorNmed)
            print  heap[0][0] , ' --  ',logerrorNmean, ' , ',  logerrorNmed, ' , ', row.logerror ,  ' (  ' , mean , ', ' , med , ' )'
            print 'size: ', len(logerrorTrue)
            print np.average( np.abs(map(operator.sub, logerrorNMean , logerrorTrue)) )
            print np.average( np.abs(map(operator.sub, logerrorNMed , logerrorTrue)) )
            print np.average( np.abs(map(operator.sub, [mean for i in logerrorTrue] , logerrorTrue)) )
            print np.average( np.abs(map(operator.sub, [med for i in logerrorTrue] , logerrorTrue)) )






    def test_lat_long(self, max_mins, trained_df, test, mean = 0 , median = 0):

        print (test.shape)
        # test.dropna(subset = ['latitude', 'longitude'], how = 'any', inplace = True)
        # test = test[np.abs(test['logerror']) < 0.5]
        print (test.shape)
        # for i in self.train.columns:
        #     print (i)
        print ('start')
        # max_lat = np.max(self.train.latitude)
        # min_lat = np.min(self.train.latitude)
        # max_long = np.max(self.train.longitude)
        # min_long = np.min(self.train.longitude)
        nsquares = 50
        print ('start')
        lat_step = (max_mins['max_lat']-max_mins['min_lat'])*1.0/nsquares
        long_step = (max_mins['max_long']-max_mins['min_long'])*1.0/nsquares
        print (lat_step , ' , ', long_step )
        test['lat_cat'] = test.latitude.map(lambda x: int((x-max_mins['min_lat'])*1.0/lat_step) if x==x and x > max_mins['min_lat'] and x < max_mins['max_lat'] else -1)
        test['long_cat'] = test.longitude.map(lambda x: int((x-max_mins['min_long'])*1.0/long_step) if x==x and x > max_mins['min_long'] and x < max_mins['max_long'] else -1)

        print 'test.columns: ', test.columns
        test_df = pd.merge(test, trained_df ,on=['lat_cat' , 'long_cat'],how="left")
        print 'test_df.columns: ', test_df.columns

        err = 0
        err2 = 0
        err3 = 0
        count = 0
        for index, row in test_df.iterrows():
            # print row['logerror']
            # print row['med']
            if not np.abs(row['logerror'] - row['med']) == np.abs(row['logerror'] - row['med']):
                print index
                print row
            if row['siz'] > 100:
                err = err +  np.abs(row['logerror'] - row['med'])
                err2 = err2 + np.abs(row['logerror'] - mean )
                err3 = err3 + np.abs(row['logerror'] - median )
                count = count+1
        err = err/ count # test_df.shape[0]
        err2 = err2/count # test_df.shape[0]
        err3 = err3/ count #test_df.shape[0]
        print 'count: ' , count , ' , ' , test_df.shape[0]
        print err , ' , ', err2,  ' , ', err3


    def estimate(self):

        data_path = '/Users/Mz/Google_Drive/university/research/zillow/data/'
        property_file_name = 'properties_2016.csv'
        train_file_name = 'train_2016.csv'
        test_file_name = 'sample_submission.csv'
        submission_file_name = 'my_submission.csv'

        # sample_eval_submission_file_name = 'sample_eval_submission.csv'
        # true_eval_file_name = 'true_evaluation.csv'
        # my_train_file_name = 'my_train_2016.csv'
        train_df = pd.read_csv(data_path+train_file_name)
        property_df = pd.read_csv(data_path+property_file_name)
        test_df = pd.read_csv(data_path+test_file_name)
        print 'test_df.shape: ' ,test_df.shape
        print 'train_df.shape: ' , train_df.shape



        # merging train_df and property_df
        full_train_df  = pd.merge(property_df, train_df, on='parcelid', how='inner')
        full_train_df = full_train_df.set_index('parcelid')
        train_mean = np.mean(full_train_df.logerror)


        test_df.ix[:,'201610'] = train_mean
        test_df.ix[:,'201611'] = train_mean
        test_df.ix[:,'201612'] = train_mean
        test_df.ix[:,'201710'] = train_mean
        test_df.ix[:,'201711'] = train_mean
        test_df.ix[:,'201712'] = train_mean

        test_df.to_csv(data_path+submission_file_name, sep=',', encoding='utf-8')


    def random_select(self):



# sampleWeight = [200 , 300 , 400, 500 , 600 , 700 , 800 , 900, 1000 , 2000]
#
# sampleWeight = np.array([ 8.0 / ( 1 + np.exp( - (np.minimum(val, 1000 ) - 600.0 )/ 800)) - 3 for val in sampleWeight])
# print (sampleWeight)
#
zillow = Zillow()
zillow.init()
# zillow.lat_long()

max_mins, trained_df, mean, median = zillow.train_lat_long(zillow.train)
# zillow.test_lat_long(max_mins, trained_df, zillow.test, mean , median)

# zillow.knn(zillow.train, zillow.test)

# zillow.plot_nan_sum()
## zillow.decision_tree()
#zillow.partially_linear()


zillow.random_select()