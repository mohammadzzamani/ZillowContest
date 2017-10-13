import pandas as pd
import sys
import MySQLdb
import os
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from happierfuntokenizing.happierfuntokenizing import Tokenizer

data_path = ''
user= ''
password = ''
#database = 'zillowChallenge'
database = 'mztwitter'
host = ''
table = 'final_msgs_dedup'
filename = 'zillow_messages.tsv'

def connectToDB():
                # Create SQL engine
                myDB = URL(drivername='mysql', database=database, query={
                    'read_default_file' : '/home/mzamani/.my.cnf' })
                engine = create_engine(name_or_url=myDB)
                engine1 = engine
                connection = engine.connect()
                return connection

def connectMysqlDB():
        conn = MySQLdb.connect(host, user, password, database)
        c = conn.cursor()
        return c


def load_data(fname):
    print 'load_data...'
    data = {"test": {}, "train": {}}
    #counter = 0
    #loaded_data = pd.read_csv(fname,)
    # with open(fname) as f:
    # for line in f:
    
    try:
            cursor = connectToDB()
    except:
            print("error while connecting to database:", sys.exc_info()[0])
            raise
    if(cursor is not None):
            sql='select {0}, {1} from {2}'.format('parcelid', 'message', table)
            query = cursor.execute(sql)
            res = query.fetchall()
	    
    #res = res[:2000]

    for r in res:
        #print r
	uid = r[0]
	tweet = r[1]        
	#uid, test_train, tweet = line.strip().split("\t")
        #test_train = test_train.lower()
	test_train = 'train'
        if uid not in data[test_train]:
            data[test_train][uid] = tweet
        else:
            data[test_train][uid] += " " + tweet


    train_tweets = []
    train_uids = []
    for uid in data["train"]:
        train_tweets.append(data["train"][uid])
        train_uids.append(uid)

    test_tweets = []
    test_uids = []
    for uid in data["test"]:
        test_tweets.append(data["test"][uid])
        test_uids.append(uid)
    return train_tweets, train_uids, test_tweets, test_uids


def run_tfidf(train_tweets, test_tweets):
    print 'run_tfidf...'
    tokenizer = Tokenizer()
    tfidf_vectorizer = TfidfVectorizer(input="content",
                                       strip_accents="ascii",
                                       decode_error="replace",
                                       analyzer="word",
                                       tokenizer=tokenizer.tokenize,
                                       ngram_range=(1, 3),
                                       stop_words="english",
                                       max_df=0.8,
                                       min_df=0.2,
                                       use_idf=True,
                                       max_features=200000)

    train_tfidf = tfidf_vectorizer.fit_transform(train_tweets)
    #test_tfidf = tfidf_vectorizer.transform(test_tweets)
    test_tfidf = []

    return train_tfidf, test_tfidf


def run_kmeans(train_tfidf, d):
    print 'run_kmeans...'
    km = KMeans(n_clusters=d, random_state=123, max_iter=1000,
                n_init=10)
    km.fit(train_tfidf)

    return km.cluster_centers_


def output_factors(train_uids, train_tfidf, test_uids, test_tfidf, cluster_centers,
                   output_f):
    print 'output_factors....'
    if len(test_uids) > 0:
    	uid_list = [train_uids, test_uids]
    	feat_list = [train_tfidf, test_tfidf]
    else:
	uid_list = train_uids
	feat_list = train_tfidf

    print 'len(uid_list): ' , len(uid_list)

    with open(output_f, "w") as f:
        header = ["UID"] + ["Factor {}".format(i + 1) for i in range(len(cluster_centers))]
        f.write("\t".join(header) + "\n")
        for i in range(len(uid_list)):
            us = uid_list[i]
	    if not isinstance(us,list):
		us = [us]
            #print 'us: ' , us
	    dists = euclidean_distances(feat_list[i], cluster_centers)
            #print 'dists: ' , dists
	    for j in range(len(us)):
                dist = list(dists[j])
                row = [us[j]] + [str(1 / (factor + .01)) for factor in dist]
                f.write("\t".join(row) + "\n")


def parse_cmd_input():
    print 'parse_cmd_input....'
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", metavar="F", type=str,
                        help="tab-separated input file where each line is in format USER_ID<tab>TEST/TRAIN<tab>TWEET")
    parser.add_argument("output_file", metavar="O", type=str,
                        help="name of output file to be produced")
    parser.add_argument("-d", metavar="d", type=int, default=5, nargs="?",
                        help="number of factors to produce")

    args = parser.parse_args()

    return args.input_file, args.output_file, args.d


def main():
    print 'main....'
    input_f, output_f, d = parse_cmd_input()
    train_tweets, train_uids, test_tweets, test_uids = load_data(input_f)
    print "Running tf-idf"
    train_tfidf, test_tfidf = run_tfidf(train_tweets, test_tweets)
    print "Running k-means"
    cluster_centers = run_kmeans(train_tfidf, d)



    output_factors(train_uids, train_tfidf, test_uids, test_tfidf, cluster_centers,
                   output_f)


if __name__ == '__main__':
    main()


def extract():
        try:
                cursor = connectToDB()
        except:
                print("error while connecting to database:", sys.exc_info()[0])
                raise
        if(cursor is not None):
            sql='select {0}, {1} from {2}'.format('parcelid', 'message', table)
            query = cursor.execute(sql)
            res = query.fetchall()

            cols = ['parcelid', 'message']
            data = pd.DataFrame(data= res, columns= cols)

            data['train_test'] = 'train'

            data = data[['parcelid', 'train_test' , 'message']]
	    data.set_index('parcelid', inplace=True)

	    
            data.to_csv(filename, sep='\t', header=None)


extract()
