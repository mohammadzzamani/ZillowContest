from sklearn.linear_model import Ridge, RidgeCV
import numpy as np
import pandas as pd
import expand_feats as ef

print type(ef.all_df)


# X = ef.all_df[ef.all_df.columns[1:]]
# Y = ef.all_df[ef.all_df.columns[0]]
X = ef.all_df.ix[:,1:]
Y = ef.all_df.ix[:,0]

print X.shape
print Y.shape

clf = RidgeCV(alphas=[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])



