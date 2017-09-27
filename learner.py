from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, LassoCV
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

import numpy as np
import pandas as pd
import expand_feats as ef
from sklearn.metrics import mean_absolute_error, mean_squared_error


alphas=[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

def cross_validation(all_df, clf, folds = 10):
    all_df['selection'] = 0
    fold_sizes = all_df.shape[0]*1.0/folds;
    mae_sum = 0
    mse_sum = 0

    Ytest = all_df.iloc[:,0]

    YpredsAll = None
    for i in range(0,folds):
        test_start = i* fold_sizes
        test_end = (i+1) * fold_sizes
        selection = [ True if ( i >=test_start and i < test_end) else False for i in range(all_df.shape[0])]
        deselection = [ False if val == True else True for val in selection]

        train = all_df.iloc[deselection]
        test = all_df.iloc[selection]

        Xtrain = train.iloc[:,1:]
        Xtest = test.iloc[:,1:]
        Ytrain = train.iloc[:,0]
        # Ytest = test.iloc[:,0]

        # clf.fit(X=Xtrain,y=Ytrain)
        # Ypred=clf.predict(X=Xtest)
        #
        # mae = mean_absolute_error(Ypred,Ytest.values)
        # mse = mean_squared_error(Ypred,Ytest.values)
        # mae_sum = mae_sum + mae
        # mse_sum = mse_sum + mse

        Ypreds = None

        '''
        ESTIMATORS = {
            # "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0),
            "K-nn": KNeighborsRegressor(n_neighbors=20),
            # "Linear regression": LinearRegression(),
            "Ridge": RidgeCV(alphas=alphas),
            "lasso": LassoCV(alphas=alphas)
        }
        Ypred = {}
        for name, estimator in ESTIMATORS.items():
            estimator.fit(Xtrain, Ytrain)
            Ypred[name] = estimator.predict(Xtest)
            if Ypreds is None:
                Ypreds = Ypred[name]
            else:
                Ypreds = np.vstack((Ypreds, Ypred[name]))
            print 'shape: ' , Ypreds.shape
        '''

        ESTIMATORS = [
            # ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0),
            KNeighborsRegressor(n_neighbors=20),
            RidgeCV(alphas=alphas),
            LassoCV(alphas=alphas),
            # AdaBoostRegressor(random_state=0)
            # GradientBoostingRegressor(random_state=0)
        ]
        for estimator in ESTIMATORS:
            estimator.fit(Xtrain, Ytrain)
            Ypred = estimator.predict(Xtest)
            if Ypreds is None:
                Ypreds = Ypred
            else:
                Ypreds = np.vstack((Ypreds, Ypred))
            print 'shape: ' , Ypreds.shape

        if len(ESTIMATORS)>1:
            Ypreds = np.vstack((Ypreds , np.mean(Ypreds, axis=0)))
            print Ypreds.shape

        if YpredsAll is None:
            YpredsAll = Ypreds
        else:
            YpredsAll = np.hstack((YpredsAll, Ypreds))
        print 'YpredsAll.shape: ' , YpredsAll.shape

    if len(ESTIMATORS)<1:
        YpredsAll = YpredsAll.reshape(len(YpredsAll),1)
        for ypred in YpredsAll.T:
            ypred = ypred.T
            print evaluate(Ytest, ypred)
    else:
        for ypred in YpredsAll:
            ypred = ypred
            print evaluate(Ytest, ypred)




    # print 'mae_sum: ' , mae_sum
    # print 'mse_sum: ' , mse_sum

def evaluate(Ytrue, Ypred):
    mae = mean_absolute_error(Ytrue,Ypred)
    mse = mean_squared_error(Ytrue,Ypred)
    return [mae , mse]

all_df = (ef.all_df-ef.all_df.min())/(ef.all_df.max()-ef.all_df.min())
all_df.dropna(axis=1, how='any', inplace=True)

clf = RidgeCV(alphas=[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
cross_validation(all_df,clf , 10)

# X = ef.all_df[ef.all_df.columns[1:]]
# Y = ef.all_df[ef.all_df.columns[0]]

X = all_df.ix[:,1:]
Y = all_df.ix[:,0]
print X.shape
print Y.shape

# clf = RidgeCV(alphas=[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
# clf.fit(X=X,y=Y)
# Ypred=clf.predict(X=X)
# print 'mae:  ', mean_absolute_error(Ypred,Y.values)
# print 'mse:  ', mean_squared_error(Ypred,Y.values)

lm = Y.mean()
baseline = [ lm for y in Y ]
print 'baseline mae:  ', mean_absolute_error(baseline, Y.values)
print 'baseline mse:  ', mean_squared_error(baseline, Y.values)


