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

import sys
import MySQLdb
import os
class Zillow:
	data_path = ''
	user= 'mzamani'
	password = ''
	database = 'zillowChallenge'
	host = 'localhost'
	table = 'all_msgs'

	def connectMysqlDB(self):
		conn = MySQLdb.connect(self.host, self.user, self.password, self.database)
		c = conn.cursor()
		return c   
	

	def func(self):
		try:
			self.cursor = self.connectMysqlDB()
		except:
			print("error while connecting to database:", sys.exc_info()[0])
			raise
		if(self.cursor is not None):
			sql = "select message_id, user_id, message, coordinates  from {0} ".format(self.table)
			self.cursor.execute(sql)
			res = self.cursor.fetchall()
			columns = ['message_id', 'user_id', 'message', 'coordinates' ]
			messages_df = pd.dataFrame( data = res, columns = columns)
			print( 'messages_df.shape: ' , messages_df.shape)
	def init(self):
		self.data = pd.read_csv(self.data_path+'train_2016.csv', parse_dates=["transactiondate"])
		self.properties = pd.read_csv(self.data_path+ 'properties_2016.csv')
		print ("Shape Of Data: ",self.data.shape)
		print ("Shape Of Properties: ",self.properties.shape)

		self.merged = pd.merge(self.data,self.properties,on="parcelid",how="left")
		self.merged = self.merged[np.abs(self.merged['logerror']) < 1]

		self.train = self.merged[self.merged.transactiondate<'2016-10-01']
		self.test = self.merged[self.merged.transactiondate>'2016-09-30' ]
		print ("Shape Of Train: ",self.train.shape)
		print ("Shape Of Test: ",self.test.shape)
				
		merged_sampled = self.merged.sample(2000)
		


zillow = Zillow()
zillow.init()
zillow.func()
