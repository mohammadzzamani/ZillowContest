import MySQLdb
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import sys
import pandas as pd
import numpy as np
from functools import partial

# database connection info
database = 'zillow'
user= ''
password = ''
host = ''

# db table names
feature_table = 'feat$cat_msg_8to10_rpca_w$final_msgs_8to10$message_id$16to16'
msg_table = 'final_msgs_8to10'
housing_table = 'zillow_properties'

hid_rid_table = ''
rid_feat_table = ''


houses_df = None
features_df = None
msgs_df = None
distinct_features = None

# latitude longitude bounds
bound = {'lat': [33.7, 34.92] , 'lon':[-119.5 , -117.16]}
grid_size = 100  # for both width and height
num_of_sub_regions_in_region_square = 2  # for both width and height
lat_lon_adjustment = float(10**6)


def calc_row_number(latitude):
    return   int( (latitude - bound['lat'][0])/(bound['lat'][1] - bound['lat'][0]) * grid_size)

def calc_col_number(longitude):
    return int( (longitude - bound['lon'][0])/(bound['lon'][1] - bound['lon'][0]) * grid_size)





def connectToDB():
    # Create SQL engine
    myDB = URL(drivername='mysql', database=database, query={
        'read_default_file' : '/home/mzamani/.my.cnf' })
    engine = create_engine(name_or_url=myDB)
    engine1 = engine
    connection = engine.connect()
    return connection


def retrieve():
    print ('retrieve ...')
    try:
        cursor = connectToDB()
    except:
        print("error while connecting to database:", sys.exc_info()[0])
        raise

    features = None
    msgs = None
    houses = None

    if cursor is not None:
        # get houses
        sql='select {0}, {1}, {2} from {3}'.format('parcelid', 'latitude', 'longitude', housing_table)
        query = cursor.execute(sql)
        houses = query.fetchall()
        houses_df = pd.DataFrame(data= houses, columns = ['parcelid', 'latitude', 'longitude'])
        # get msgs
        sql='select {0}, {1}, {2} from {3}'.format('message_id', 'latitude', 'longitude', msg_table)
        query = cursor.execute(sql)
        msgs = query.fetchall()
        msgs_df = pd.DataFrame(data= msgs, columns = ['message_id', 'latitude', 'longitude'])
        # get features
        sql='select {0}, {1}, {2}, {3} from {4}'.format('group_id', 'feat', 'value', 'group_norm', feature_table)
        query = cursor.execute(sql)
        features = query.fetchall()
        features_df = pd.DataFrame(data= features, columns = ['message_id', 'feat', 'value', 'group_norm'])

        # get feature names
        sql='select {0} from {1} group by {0}'.format('feat', feature_table)
        query = cursor.execute(sql)
        distinct_features = query.fetchall()


        return houses_df, msgs_df, features_df, distinct_features



houses_df, msgs_df, features_df, distinct_features = retrieve()

print ('merging msgs and features')
msgs_features_df = pd.merge(msgs_df, features_df, how='left', on='message_id')

msgs_features_df['row']=0
msgs_features_df['col']=0

# for index, msg in msgs_features_df:
#     msgs_features_df.set_value(index,)


print ('calculating rows and cols')
# latitude_func = partial(calc_row_col_number(), lat_long = 'latitude')
msgs_features_df['row'] = msgs_features_df.latitude.map(calc_row_number)

# longitude_func = partial(calc_row_col_number(), lat_long = 'longitude')
msgs_features_df['col'] = msgs_features_df.longitude.map(calc_col_number)

print ('grouping on row and col')
msgs_features_df = msgs_features_df.groupby(['row', 'col']).sum()

print ('msgs_features_df.shape: ', msgs_features_df.shape)
print ('msgs_features_df: ', msgs_features_df)







