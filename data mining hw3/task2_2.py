import json
import time
from itertools import combinations
from pyspark import SparkConf, SparkContext
from operator import add
import csv
import json
import os
import sys
import xgboost as xgb
import numpy as np



start_time = time.time()
folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

#folder_path = '/Users/ruichao/Desktop/all_data'
#test_file_name = '/Users/ruichao/Desktop/all_data/yelp_val.csv'
conf = SparkConf().setAppName("553hw3").setMaster('local[*]')
sc = SparkContext(conf=conf)

train_path = folder_path + '/yelp_train.csv'
train = sc.textFile(train_path)
header = train.first()
train = train.filter(lambda i: i != header).map(lambda i: i.split(","))

test = sc.textFile(test_file_name)
header_test = test.first()
test = test.filter(lambda i: i != header_test).map(lambda i: i.split(","))

user_path = folder_path + '/user.json'
user = sc.textFile(user_path)

business_path = folder_path + '/business.json'
business = sc.textFile(business_path)

user_json = user.map(json.loads).map(
    lambda i: ((i["user_id"], (i["useful"],i['compliment_hot'],i['fans'], i["review_count"], i["average_stars"])))).collectAsMap()
business_json = business.map(json.loads).map(lambda i: ((i["business_id"], (i["review_count"], i["stars"])))).collectAsMap()


def preprocess_user(data, user_json,default):
    if default == False:
        user_id = data[0]
        rating = data[2]
    else:
        user_id = data[0]
        rating = -1.0


    if user_id in user_json.keys():
        useful, reviewcount, averagestar = user_json[user_id]
        return [user_id, float(useful), float(reviewcount), float(averagestar), float(rating)]
    else:
        return [user_id, None, None, None, None]


def preprocess_business(data, business_json,default):
    if default == False:
        user_id = data[0]
        business_id = data[1]
        rating = data[2]
    else:
        user_id = data[0]
        business_id = data[1]
        rating = -1.0
    if business_id in business_json.keys():
        reviewcount, averagestar = business_json[business_id]
        return [user_id,business_id, float(reviewcount), float(averagestar),float(rating)]
    else:
        return [user_id,business_id, None, None, None]


def preprocess_data(data,user_json,business_json,default):
    user_id = data[0]
    business_id = data[1]
    if default == False:
        rating = data[2]
    else:
        rating = -1
    if user_id in user_json.keys() and business_id in business_json.keys():
        useful, compliment_hot,fans, reviewcount_user, averagestar_user = user_json[user_id]
        reviewcount_business, averagestar_business = business_json[business_id]
        useful, compliment_hot,fans,reviewcount_user, averagestar_user,reviewcount_business, averagestar_business,rating = float(useful),float(compliment_hot),float(fans), float(reviewcount_user), float(averagestar_user),float(reviewcount_business), float(averagestar_business),float(rating)

        return [user_id,business_id,useful,compliment_hot,fans, reviewcount_user, averagestar_user,reviewcount_business, averagestar_business,rating]
    else:
        return [user_id,business_id,None,None,None,None,None,None,None,None]
train_user = train.map(lambda i: preprocess_user(i, user_json,False))
train_business = train.map(lambda i: preprocess_business(i, business_json,False))


def create_train_or_test(data,tf):
    train_xgb = data.map(lambda i: preprocess_data(i, user_json,business_json,tf)).collect()
    train_xgb_array = np.array(train_xgb)
    len_cell = len(train_xgb_array[0])
    x = np.array(train_xgb_array[:,2:len_cell-1],dtype = 'float')
    y = np.array(train_xgb_array[:,-1],dtype = 'float')
    return x,y
def create_test_dataset(data,tf):
    test_xgb = data.map(lambda i: preprocess_data(i, user_json,business_json,tf)).collect()
    test_xgb_array = np.array(test_xgb)
    return test_xgb_array

x_train, y_train = create_train_or_test(train,False)
x_test, y_test = create_train_or_test(test,True)

xgb_used = xgb.XGBRegressor(learning_rate=0.25)
xgb_used.fit(x_train,y_train)

prediction = xgb_used.predict(x_test)
test_dataset = create_test_dataset(test,True)
results = np.c_[test_dataset[:,:2],prediction]

#print(results)

def csv_writing(path,input):
    file = open(path,mode='w')
    ww = csv.writer(file,delimiter=',',quoting=csv.QUOTE_MINIMAL)
    ww.writerow(['user_id','business_id','prediction'])
    for i in input:
        ww.writerow([str(i[0]), str(i[1]), i[2]])
    file.close()

csv_writing(output_file_name,results)
end_time = time.time()-start_time
print("Duration: "+str(end_time))

