import sys
import MySQLdb
import math
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import pandas as pd

class ColToRow:
        data_path = 'data/'
        user= ''
        password = ''
        database = 'mztwitter'
        host = ''
        table = 'properties_2000selected'
        train_file = 'train_2016.csv'
	properties_file = 'properties_2016.csv'
	msgs_table = 'final_msgs'
	zillow_train_table = 'zillow_train'

	def connectToDB(self):
    		# Create SQL engine
    		myDB = URL(drivername='mysql', database=self.database, query={
        	    'read_default_file' : '/home/mzamani/.my.cnf' })
    		engine = create_engine(name_or_url=myDB)
    		engine1 = engine
    		connection = engine.connect()
		return connection

	def connectMysqlDB(self): 
		conn = MySQLdb.connect(self.host, self.user, self.password, self.database)
                c = conn.cursor()
                return c

	def handle_categories(self):
		self.properties = pd.read_csv(self.data_path+ 'properties_2016.csv')
                self.properties.set_index('parcelid', inplace=True)
		self.properties = self.properties[ self.properties.index.isin(self.selected_ids) ]
		print ( ' len(self.properties): ' , self.properties.shape) 
		print (self.properties.dtypes)
		for col in self.properties.columns:
			if  self.properties[col].dtype <> np.float64:
				self.properties[col] = pd.Categorical(self.properties[col])
				#print (len(self.properties[col].cat.codes))
				self.properties[col] = self.properties[col].cat.codes
				#print (self.properties[col].dtype)
			#print type(self.properties[col])

	def store_train_to_db(self):
		self.train = pd.read_csv(self.data_path+ ''+self.train_file)
		self.train.set_index('parcelid', inplace=True)
		#print (self.selected_ids)
		#print (self.train)
		print self.selected_ids[:10]
		print (self.train.index[:10])
		self.train = self.train[ self.train.index.isin(self.selected_ids) ]
		print ('len(self.train): ' , self.train.shape)
		try:
                        cursor = self.connectToDB()
                except:
                        print("error while connecting to database:", sys.exc_info()[0])
                        raise
                if(cursor is not None):
			sql = "drop table if exists {0}".format(self.zillow_train_table)
			print ('sql: ' , sql)
			query = cursor.execute(sql)
			sql = "create table {0} (parcelid varchar(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL, logerror float DEFAULT NULL, PRIMARY KEY (`parcelid`))".format(self.zillow_train_table)
			print ('sql: ' , sql)
			query = cursor.execute(sql)
			for index , row in self.train.iterrows():
				#print (index)
                        	sql = "insert ignore into {0} values ({1}, {2} )".format(self.zillow_train_table, index, row.logerror)
                        	#print ('sql: ' , sql)
				query = cursor.execute(sql)


	def retrieve_selected(self):
		try:
                        cursor = self.connectToDB()
                except:
                        print("error while connecting to database:", sys.exc_info()[0])
                        raise
                if(cursor is not None):
			sql = "select distinct(parcelid) from {0}".format(self.msgs_table)
			query = cursor.execute(sql)
			result =  query.fetchall()
			#for row in result:
			self.selected_ids = [ int(row[0]) for row in result ]


        def retrieve_cols(self):
                print 'db:retrieve_rows'
		columns = self.properties.columns
                try:
                	cursor = self.connectToDB()
                except:
                        print("error while connecting to database:", sys.exc_info()[0])
                        raise
                if(cursor is not None):
                	sql = "drop table if exists {0}".format(self.table)
			cursor.execute(sql)
			sql = "create table {0} (id int(10) unsigned NOT NULL AUTO_INCREMENT, group_id  varchar(30) COLLATE utf8mb4_bin DEFAULT NULL, feat  varchar(102) COLLATE utf8mb4_bin DEFAULT NULL, value int(11) DEFAULT NULL, group_norm double DEFAULT NULL, PRIMARY KEY (`id`), KEY `correl_field` (`group_id`),  KEY `feature` (`feat`))".format(self.table)
			cursor.execute(sql)
			counter = 0
			for index, row in self.properties.iterrows():
				print ( 'counter: ' , counter )
				counter +=1
				for i in range(len(columns)):
					if not  math.isnan(float(row[i])):
						sql = "insert into {0} (group_id, feat, value, group_norm) values ( {1}, \"{2}\" , {3}, {4} )".format(self.table, index , columns[i], row[i], row[i]) 
						cursor.execute(sql)

script = ColToRow()
script.retrieve_selected()
#script.store_train_to_db()
#script.handle_categories()
#script.retrieve_cols()
