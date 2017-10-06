import pandas as pd
import sys
import MySQLdb
import os


data_path = ''
user= 'mzamani'
password = ''
database = 'zillowChallenge'
host = 'localhost'
table = 'final_msgs_dedup'
filename = 'zillow_messages.tsv'

def connectMysqlDB(self):
        conn = MySQLdb.connect(host, user, password, database)
        c = conn.cursor()
        return c


def extract():
        try:
                cursor = connectMysqlDB()
        except:
                print("error while connecting to database:", sys.exc_info()[0])
                raise
        if(cursor is not None):
            sql='select {0}, {1} from {2}'.format('parcelid', 'message', table)
            cursor.execute(sql)
            res = cursor.fetchall()

            cols = ['parcelid', 'message']
            data = pd.DataFrame(data= res, columns= cols)

            data['train_test'] = 'train'

            data = data[['parcelid', 'train_test' , 'message']]


            data.to_csv(filename, sep='\t')


extract()
