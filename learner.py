from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, LassoCV, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from  catboost import CatBoostRegressor

import my_tensorflow as mtf


import numpy as np
import pandas as pd
import expand_feats as ef
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing



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

alphas=[0.0000000001,0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001 , 0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]


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


def outlier_detection(data):
    print ('outlier_detection...')
    print (data.shape)
    # print type(data)
    # data = data[np.where(data[:,0]<0.9)]
    data = data[abs(data.logerror ) < 15]
    print (data.shape)
    return data

def cross_validation_clf(all_df, clf, folds = 10):

    fold_sizes = all_df.shape[0]*1.0/folds
    mae_sum = 0
    mse_sum = 0

    Ytest_original = all_df.iloc[:,0]

    all_df.logerror = all_df.logerror.map(lambda x: into_classes(x))
    Ytest = all_df.iloc[:,0]
    # Ytest = np.sign(all_df.iloc[:,0])

    # into_classes(all_df)

    YpredsAll = None
    for i in range(0,folds):
        test_start = i* fold_sizes
        test_end = (i+1) * fold_sizes
        selection = [ True if ( i >=test_start and i < test_end) else False for i in range(all_df.shape[0])]
        deselection = [ False if val == True else True for val in selection]

        train = all_df.iloc[deselection]
        test = all_df.iloc[selection]


        # train = outlier_detection(train)

        print ('train.shape: ' , train.shape)
        print ('test.shape: ' , test.shape)

        Xtrain = train.iloc[:,1:]
        Xtest = test.iloc[:,1:]
        Ytrain = train.iloc[:,0]
        # Ytrain = np.sign(train.iloc[:,0])
        thisYtest = test.iloc[:,0]
        # thisYtest = np.sign(test.iloc[:,0])

        # ESTIMATORS = [
        #     ('gnb', GaussianNB()),
        #     ('dtc', DecisionTreeClassifier(random_state=0)),
        #     ('rfc1' , RandomForestClassifier(max_depth=2, random_state=0)),
        #     ('rfc2', RandomForestClassifier(max_depth=3, random_state=1, n_estimators=20, criterion='entropy')),
        #     ('rfc3', RandomForestClassifier(max_depth=3, random_state=2, n_estimators=10, criterion='gini')),
        #     ('lr', LogisticRegression(random_state=1))
        # ]
        # voting = VotingClassifier(
        #     estimators= ESTIMATORS,
        #     voting='soft', weights=[1,1,1,1,1,1],
        #     #flatten_transform=True
        # )
        #
        # voting.fit(Xtrain,Ytrain)
        # Ypred = voting.predict(Xtest)
        # print 'Ypred.shape: ' , Ypred.shape
        # YpredsAll = stack_folds_preds(Ypred, Ypreds,1)

        Ypreds = None
        ESTIMATORS = [
            GaussianNB(),
            DecisionTreeClassifier(random_state=0, max_depth=3),
            # RandomForestClassifier(max_depth=2, random_state=1),
            # RandomForestClassifier(max_depth=3, random_state=2, n_estimators=20, criterion='entropy'),
            RandomForestClassifier(max_depth=4, random_state=3, n_estimators=10, criterion='gini'),
            # RandomForestClassifier(max_depth=2, random_state=4, n_estimators=10, criterion='entropy'),
            AdaBoostClassifier(),
            BaggingClassifier(n_estimators=50, max_samples=0.9, max_features=0.9),
            GradientBoostingClassifier()
            # QuadraticDiscriminantAnalysis(),
            # SVC(kernel="linear", C=0.025),
            # SVC(gamma=2, C=1),
            # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
            # MLPClassifier(alpha=1)
            # LogisticRegression(random_state=4),
            # mean_est(type='classification')
        ]
        for estimator in ESTIMATORS:
            estimator.fit(Xtrain, Ytrain)
            Ypred = estimator.predict(Xtest)
            Ypreds = stack_folds_preds(Ypred, Ypreds, 0)
            print ('shape: ' , Ypreds.shape)
            print ('evaluate: ' )
            evaluate(thisYtest, Ypred)
            # print ('evaluate: ' , evaluate(np.sign(thisYtest), np.sign(Ypred)) )

        if len(ESTIMATORS)>1:
            print ('.... ', Ypreds.shape)
            Ypreds = np.vstack((Ypreds , np.mean(Ypreds, axis=0)))


        YpredsAll = stack_folds_preds(Ypreds, YpredsAll, 1)
        print ('YpredsAll.shape: ' , YpredsAll.shape)



    print ('classes:')
    for ypred in YpredsAll:
        ypred = ypred
        print ('final_eval: ')
        evaluate(Ytest, ypred) #, type='classification2')
        # evaluate(Ytest, np.sign(ypred), type='classification2')

    print ('real_values:')
    Ytest = Ytest_original
    for ypred in YpredsAll:
        ypred = ypred
        print ('final_eval: ')
        evaluate(Ytest, ypred)

    cols = ['ytrue', 'gnb', 'dtc', 'rfc3',  'ab', 'bc', 'gbc', 'avg']
    Ytest = all_df.iloc[:,0]
    print (Ytest.shape, ' , ', YpredsAll.shape)
    YAll = np.vstack( (np.array(Ytest).reshape(1, len(Ytest)), YpredsAll))
    print (YAll.shape)
    YAll = pd.DataFrame(data=YAll.T, columns=cols)
    print ('corr: ' , YAll.corr())


def stack_folds_preds(pred_fold, pred_all=None, axis=0):
    if pred_all is None:
        pred_all = pred_fold
    else:
        # if axis==0:
        pred_all = np.vstack((pred_all, pred_fold)) if axis==0 else np.hstack((pred_all, pred_fold))
        # else:
        #     pred_all = np.vstack((pred_all, pred_fold))
    return pred_all

def cross_validation(all_df, clf, folds = 10):
    all_df['selection'] = 0
    fold_sizes = all_df.shape[0]*1.0/folds
    mae_sum = 0
    mse_sum = 0

    Ytest = all_df.iloc[:,0]

    YpredsAll = None
    for i in range(0,folds):
        test_start = i* fold_sizes
        test_end = (i+1) * fold_sizes
        selection = [ True if ( i >=test_start and i < test_end) else False for i in range(all_df.shape[0])]
        deselection = [ False if val == True else True for val in selection]

        print ('all_df.shape in cv is: ' , all_df.shape)

        train = all_df.iloc[deselection]
        test = all_df.iloc[selection]

        train = outlier_detection(train)

        train_orig_feats = train[[c for c in ef.fin_z.columns if c in train.columns]]
        test_orig_feats = test[[c for c in ef.fin_z.columns if c in test.columns]]
        orig_Xtrain = train_orig_feats.iloc[:,:]
        orig_Xtest = test_orig_feats.iloc[:,:]




        print ('train.shape: ' , train.shape)
        #train = train[ abs(train.logerror) < 0.75 ]
        print ('test.shape: ' , test.shape)

        # catboost = True
        # if catboost == True:
        #     Xtrain = Xtrain[[c in Xtrain]]
        Xtrain = train.iloc[:,1:]
        Xtest = test.iloc[:,1:]
        Ytrain = train.iloc[:,0]
        thisYtest = test.iloc[:,0]



        print ('cntr = 0 ')
        print (Xtrain.shape)
        print (Xtest.shape)
        #
        # print ('cntr = 1 ')
        # print (orig_Xtrain.shape)
        # print (orig_Xtest.shape)


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
            # KNeighborsRegressor(n_neighbors=5),
            # KNeighborsRegressor(n_neighbors=10),
            # KNeighborsRegressor(n_neighbors=5),
            mean_est(),
            # RidgeCV(alphas=alphas),
            # RidgeCV(),
            # LassoCV(alphas=alphas),
            # GradientBoostingRegressor(loss='lad', random_state=5, n_estimators=200, subsample=0.8 ),
            # GradientBoostingRegressor(loss='lad', random_state=8, n_estimators=50, subsample=0.7 , max_depth=4, max_features=0.8),
            # GradientBoostingRegressor(loss='lad', random_state=7, n_estimators=50, subsample=0.7 , max_depth=4),
            GradientBoostingRegressor(n_estimators= 100, loss='lad', random_state=8, subsample=0.75, max_depth=6, max_features=0.75), #, min_impurity_decrease=0.05),
            # GradientBoostingRegressor(n_estimators= 200, loss='lad', random_state=7, subsample=0.75, max_depth=6, max_features=0.75, min_impurity_decrease=0.05),
            # GradientBoostingRegressor(n_estimators= 500, learning_rate= 0.1, loss='lad', random_state=6, subsample=0.75, max_depth=5, max_features=0.75, min_impurity_decrease=0.05),
            # GradientBoostingRegressor(n_estimators= 500, learning_rate= 0.2, loss='lad', random_state=5, subsample=0.75, max_depth=6, max_features=0.75, min_impurity_decrease=0.025),
            # GradientBoostingRegressor(n_estimators= 200, loss='lad', random_state=7, subsample=0.75, max_depth=6, max_features=0.75, min_impurity_decrease=0.05),
            # CatBoostRegressor( iterations=200, learning_rate=0.03,depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE',random_seed=i)
            # GradientBoostingRegressor(loss='lad', random_state=7, n_estimators=100, subsample=0.6 ),
            # BaggingRegressor(n_estimators=20, max_samples=0.9, max_features=0.9, random_state=7),
            # mean_est(),
            # mean_est(),
            # mean_est()
            # mean_est()
            # mean_est()
            # mean_est()

            #LassoCV()
            #AdaBoostRegressor(random_state=0)
            # GradientBoostingRegressor(random_state=0)
        ]
        print (' ---- ')
        cntr = 0

        Xtrain_v = Xtrain.values
        Xtest_v = Xtest.values
        Ytrain_v = Ytrain.values.reshape(Ytrain.shape[0],1)
        thisYtest_v = thisYtest.values.reshape(thisYtest.shape[0],1)


        # ypred_tf = mtf.initialize(test, test)
        print ('Xtrain_v type: ' , type(Xtrain_v) , ' , ', Xtrain_v.shape)
        print ('Xtest_v type: ' , type(Xtest_v) , ' , ', Xtest_v.shape)
        print ('Ytrain_v type: ' , type(Ytrain_v) , ' , ', Ytrain_v.shape)
        print ('thisYtest_v type: ' , type(thisYtest_v) , ' , ', thisYtest_v.shape)

        print ('max, min: ' , np.max(Ytrain_v), ' , ', np.min(Ytrain_v))
        print ('max, min: ' , np.max(thisYtest_v), ' , ', np.min(thisYtest_v))

        ypred_tf = mtf.run_(Xtrain_v, Ytrain_v, Xtest_v, thisYtest_v)

        # print ('thisYtest:')
        # print (thisYtest)
        # print ('ypred_tf:')
        # print (ypred_tf)
        evaluate(thisYtest, np.array(ypred_tf), '\n' + str(i) + ' ,  deep learning' , mea=transformation_mean, va=transformation_var)

        for estimator in ESTIMATORS:
            # if check < 0:
            #     check+=1
            #     knn_Xtrain = train[['latitude','longitude']] #.iloc[:,1:]
            #     knn_Xtest = test[['latitude','longitude']] #.iloc[:,1:]
            #     estimator.fit(knn_Xtrain, Ytrain)
            #     Ypred = estimator.predict(knn_Xtest)
            #     print ('shape: ' , knn_Xtrain.shape, ' , ', knn_Xtest.shape, ' , ' , Ypred.shape)
            # else:
            if cntr == 10:
                estimator.fit(orig_Xtrain, Ytrain)
                # print ('alpha: ' , estimator.alpha_)
                # print (estimator.coef_)
                Ypred = estimator.predict(orig_Xtest)
            else:
                estimator.fit(Xtrain, Ytrain)
                # print ('alpha: ' , estimator.alpha_)
                # print (estimator.coef_)
                Ypred = estimator.predict(Xtest)
            cntr+=1


            Ypreds = stack_folds_preds(Ypred, Ypreds, 0)

            # print ('shape: ' , Ypreds.shape)
            # print ('evaluate: ')
            evaluate(thisYtest, Ypred, mea=transformation_mean, va=transformation_var)
            # evaluate(np.sign(thisYtest), np.sign(Ypred))



        # print ('thisYtest:')
        # print (thisYtest)
        # print ('ypred_tf:')
        # print (ypred_tf)
        # evaluate(thisYtest, np.array(ypred_tf), mea=transformation_mean, va=transformation_var)


        # print ('shapes:  ', thisYtest.shape, ' , ', ypred_tf.shape)
        # temp = np.hstack((np.array(thisYtest).reshape(thisYtest.shape[0],1), np.array(ypred_tf)))
        # print 'temp.shape: ' , temp.shape
        # temp = pd.DataFrame(data=temp, columns=['ytrue', 'ypred_tf'])
        # print 'temp.shape: ' , temp.shape
        # print ' temp.corr():  ' , temp.corr()
        # Ypreds = np.vstack((Ypreds , np.array(ypred_tf).reshape(1,ypred_tf.shape[0])))

        if len(ESTIMATORS)>1:
            # print ('.... ', Ypreds.shape)
            Ypreds = np.vstack((Ypreds , np.mean(Ypreds, axis=0)))
            # Ypreds = np.vstack((Ypreds , np.mean(Ypreds[1:4], axis=0)))
            # print (Ypreds.shape)


        YpredsAll = stack_folds_preds(Ypreds, YpredsAll, 1)
        # if YpredsAll is None:
        #     YpredsAll = Ypreds
        # else:
        #     YpredsAll = np.hstack((YpredsAll, Ypreds))
        # print ('YpredsAll.shape: ' , YpredsAll.shape)



    YAll = np.vstack( (np.array(Ytest).reshape(1, len(Ytest)), YpredsAll))



    if len(ESTIMATORS)<1:
        YpredsAll = YpredsAll.reshape(len(YpredsAll),1)
        for ypred in YpredsAll.T:
            ypred = ypred.T
            # print ('final_eval: ')
            # evaluate(np.sign(Ytest), np.sign(ypred), type='classification2')
            evaluate(Ytest, ypred, mea=transformation_mean, va=transformation_var)
    else:
        print ('final_eval: ')
        for ypred in YpredsAll:
            ypred = ypred
            # evaluate(np.sign(Ytest), np.sign(ypred), type='classification2')
            evaluate(Ytest, ypred, mea=transformation_mean, va=transformation_var)
        # print 'shapes: ' , YpredsAll.shape
        YAll = np.vstack( (np.array(Ytest).reshape(1, len(Ytest)), YpredsAll))

        # YAll = np.vstack( (YAll , np.sign(YAll[0] - YAll[1])))
        # YAll = np.vstack( (YAll , np.sign(YAll[0] - YAll[4])))
        # print (' sign eval with knn: ')
        # evaluate(np.sign(YAll[4] - YAll[1]), np.sign(YAll[4] - YAll[0]), type='classification2')
        # print (' sign eval with knn: ')
        # evaluate(np.sign(YAll[5] - YAll[1]), np.sign(YAll[5] - YAll[0]), type='classification2')
        # print ('YAll.shape: .... ', YAll.shape)

        YAll = YAll.T
        for i in range(YAll.shape[1]):
            print (' <<<<<< ', i , ' >>>>>>')
            print (YAll[:,i])

        YAll_abs = abs(YAll)
        all_mean = np.mean(YAll_abs,axis=0)
        print ('all_mean:')
        print (all_mean)



def learning_for_submisstion_cv(all_df, folds = 10, submission_df = None):
    print ('learning_for_submisstion_cv ...')
    # all_df['selection'] = 0
    fold_sizes = all_df.shape[0]*1.0/folds
    print ('submission_df:' , submission_df)
    # Ytest = all_df.iloc[:,0]

    YpredsAll = None
    for i in range(0,folds):
        test_start = i* fold_sizes
        test_end = (i+1) * fold_sizes
        selection = [ True if ( i >=test_start and i < test_end) else False for i in range(all_df.shape[0])]
        deselection = [ False if val == True else True for val in selection]

        print ('all_df.shape in cv is: ' , all_df.shape)

        train = all_df.iloc[deselection]
        # test = all_df.iloc[selection]

        train = outlier_detection(train)


        print ('train.shape: ' , train.shape)
        # train = train[ abs(train.logerror) < 0.75 ]
        # print ('test.shape: ' , test.shape)


        Xtrain = train.iloc[:,1:]
        # Xtest = test.iloc[:,1:]
        Xtest = submission_df.iloc[:,:]
        Ytrain = train.iloc[:,0]
        # thisYtest = test.iloc[:,0]

        estimator = GradientBoostingRegressor(n_estimators= 100, loss='lad', random_state=8, subsample=0.75, max_depth=6, max_features=0.75, min_impurity_decrease=0.05)
        print (' ---- ')

        # for estimator in ESTIMATORS:
        estimator.fit(Xtrain, Ytrain)
        Ypred = estimator.predict(Xtest)


        YpredsAll = stack_folds_preds(Ypred.reshape(Ypred.shape[0],1), YpredsAll, 1)
        print ('YpredsAll.shape: ' , YpredsAll.shape)



    # print ('.... ', Ypreds.shape)
    # YpredsAll = np.hstack((YpredsAll , np.mean(YpredsAll, axis=1)))

    Ypred = np.mean(YpredsAll, axis=1)
    Ypred = transform_back(Ypred, transformation_mean, transformation_var)

    prepare_final_submission(submission_df,Ypred)






def transform_back(values, m, v):

    #print 'v: ', type(v), ' , ', type(m) , ' , ', type(values)
    # print 'm: ', m
    #print values.shape
    #print m.shape
    values = (values*v ) + m
    return values

def evaluate(Ytrue, Ypred, type='regression',  pre = 'pre ', mea=None, va=None):
    if not mea is None:
        Ytrue = transform_back(Ytrue, mea, va)
        Ypred = transform_back(Ypred, mea, va)

    mae = mean_absolute_error(Ytrue,Ypred)
    mse = mean_squared_error(Ytrue,Ypred)
    with open("res.txt", "a") as myfile:

        if type is 'regression':
            myfile.write(pre + 'mae: ' + str(mae)+ ' , mse: ' + str(mse) + ' \n' )
            print ('mae: ' , mae, ' , mse: ', mse)
        elif type is 'classification2':
            print ('accuracy: ' , (2-mae)/2)
    return [mae , mse]

def prepare_final_submission(submission_df, Ypred):
    ##### prepare submission dataframe to look like the actual submission file (using pivot_table)
    submission_df['logerror'] = Ypred
    submission_df = submission_df[['logerror']]

    if ('Date' in submission_df.columns):
        submission_df.reset_index(inplace=True)
        print (submission_df.iloc[1:50, :])


        submission_df = submission_df.pivot_table(values='logerror', index='parcelid', columns='transactiondate')

        submission_df.reset_index(inplace=True)

        submission_df.columns = ['ParcelId' , '201610' , '201710', '201611', '201711', '201612', '201712']

        submission_df = submission_df[['ParcelId' , '201610' ,  '201611','201612', '201710','201711', '201712' ]]

        submission_df.set_index('ParcelId', inplace=True)


    else:
        cols = ['201610' , '201611', '201612', '201710', '201711', '201712']
        for i in range(len(cols)):
            c = cols[i]
            submission_df[c] = submission_df['logerror']
        submission_df = submission_df[cols]
        # submission_df.columns = cols

    print ('final_submission_df.shape: ' , submission_df.shape)
    print ('final_submission_df.columns: ' , submission_df.columns)
    print (submission_df)
    final_submission_name = 'data/final_submission_outlierDetection_cv.csv'
    # submission_df.set_index('ParcelId', inplace=True)
    submission_df.to_csv(final_submission_name)

def learning_for_submisstion(all_df, submission_df):
    print ('learning_for_submisstion')

    all_df = outlier_detection(all_df)

    Xtrain = all_df.iloc[:,1:]

    Xtest = submission_df.iloc[:,:]
    Ytrain = all_df.iloc[:,0]

    estimator = GradientBoostingRegressor(loss='lad', random_state=6, subsample=0.75, max_depth=6, max_features=0.75, min_impurity_decrease=0.05)

    estimator.fit(Xtrain, Ytrain)

    # training error
    train_pred = estimator.predict(Xtrain)
    print ('train evaluation:')
    evaluate(Ytrain, train_pred, mea=transformation_mean, va=transformation_var)

    # training error of baseline
    lm = train_pred.mean()
    baseline = [ lm for y in train_pred ]
    print ('baseline mae:  ')
    evaluate(np.array(baseline), train_pred, mea=transformation_mean, va=transformation_var)


    Ypred = estimator.predict(Xtest)
    Ypred = transform_back(Ypred, transformation_mean, transformation_var)

    prepare_final_submission(submission_df,Ypred)

    # ##### prepare submission dataframe to look like the actual submission file (using pivot_table)
    # submission_df['logerror'] = Ypred
    # submission_df = submission_df[['logerror']]
    # submission_df.reset_index(inplace=True)
    # print (submission_df.iloc[1:50, :])
    # submission_df = submission_df.pivot_table(values='logerror', index='parcelid', columns='transactiondate')
    #
    # submission_df.reset_index(inplace=True)
    #
    # submission_df.columns = ['ParcelId' , '201610' , '201710', '201611', '201711', '201612', '201712']
    #
    # submission_df = submission_df[['ParcelId' , '201610' ,  '201611','201612', '201710','201711', '201712' ]]
    #
    # submission_df.set_index('ParcelId', inplace=True)
    #
    # print ('final_submission_df.shape: ' , submission_df.shape)
    # print ('final_submission_df.columns: ' , submission_df.columns)
    # print (submission_df)
    # final_submission_name = 'data/final_submission.csv'
    #
    #
    #
    #
    # submission_df.to_csv(final_submission_name)


    return submission_df




#normalizing data
transformation_mean = ef.all_df.logerror.mean()
transformation_var = ef.all_df.logerror.var()
all_df_mean = ef.all_df.mean()
all_df_var = ef.all_df.var()
print ('all_df.shape in here is: ' , ef.all_df.shape)
all_df  = (ef.all_df - ef.all_df.mean() ) / ef.all_df.var()
all_df.dropna(axis=1, how='any', inplace=True)
print ('shapes..... ')
print (all_df_mean.shape)
print (all_df_var.shape)
# print ef.submission_df.shape

#### just for submission case
'''
for c in ef.submission_df.columns:
    print 'column: ' , c
    m = all_df_mean[c]
    v = all_df_var[c]
    ef.submission_df[c] = ef.submission_df[c].map(lambda x: (x - m)/v )


submission_df = ef.submission_df
# submission_df = (ef.submission_df - all_df_mean ) / all_df_var
print ('columns: ' )
print (all_df.columns)
print (submission_df.columns)
submission_df = submission_df[[c for c in all_df.columns if c in submission_df.columns]]
print (submission_df.columns)

# null_df  = pd.isnull(submission_df).sum() > 0
# print 'null_df: '
# print null_df


# learning_for_submisstion_cv(all_df,10,submission_df)
learning_for_submisstion(all_df, submission_df)

exit()
'''

# all_df.logerror = all_df.logerror* transformation_var  + transformation_mean
# all_df = (ef.all_df-ef.all_df.min())/(ef.all_df.max()-ef.all_df.min())


clf = RidgeCV(alphas=[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
print ('all_df.shape in here is: ' , all_df.shape)
cross_validation(all_df,clf , 10)

# X = ef.all_df[ef.all_df.columns[1:]]
# Y = ef.all_df[ef.all_df.columns[0]]

X = all_df.iloc[:,1:]
Y = all_df.iloc[:,0]
print (X.shape)
print (Y.shape)

# clf = RidgeCV(alphas=[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
# clf.fit(X=X,y=Y)
# Ypred=clf.predict(X=X)
# print 'mae:  ', mean_absolute_error(Ypred,Y.values)
# print 'mse:  ', mean_squared_error(Ypred,Y.values)

lm = Y.mean()
baseline = [ lm for y in Y ]
print ('baseline mae:  ')
evaluate(np.array(baseline), Y.values, mea=transformation_mean, va=transformation_var)
# mean_absolute_error(baseline, Y.values))
# print ('baseline mse:  ', mean_squared_error(baseline, Y.values))


