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
sc.setLogLevel("WARN")

train_path = folder_path + '/yelp_train.csv'
train = sc.textFile(train_path)
header = train.first()
train = train.filter(lambda i: i != header).map(lambda i: i.split(","))

test = sc.textFile(test_file_name)
header_test = test.first()
test = test.filter(lambda i: i != header_test).map(lambda i: i.split(","))
test_use = test.sortBy(lambda i:((i[0]),(i[1]))).persist()


def user_dict_info(data):
    a = data.map(lambda x: ((x[0]), ((x[1]), float(x[2])))).groupByKey().sortByKey()
    b = a.mapValues(dict).collectAsMap()
    return b
def business_dict_info(data):
    a = data.map(lambda x: ((x[1]), ((x[0]), float(x[2])))).groupByKey().sortByKey()
    b = a.mapValues(dict).collectAsMap()
    return b

user_rating_info = user_dict_info(train)
business_rating_info = business_dict_info(train)
user_rating_avg_info = train.map(lambda i: (i[0],(float(i[2])))).combineByKey(lambda x:(x,1),lambda y,x:(y[0]+x,y[1]+1),lambda y,z:(y[0]+z[0],y[1]+z[1]))
user_rating_avg_info = user_rating_avg_info.mapValues(lambda i:i[0]/i[1])
user_rating_avg_info = {i:j for i, j in user_rating_avg_info.collect()}


business_rating_avg_info = train.map(lambda i: (i[1],(float(i[2])))).combineByKey(lambda x:(x,1),lambda y,x:(y[0]+x,y[1]+1),lambda y,z:(y[0]+z[0],y[1]+z[1]))
business_rating_avg_info = business_rating_avg_info.mapValues(lambda i:i[0]/i[1])
business_rating_avg_info = {i:j for i, j in business_rating_avg_info.collect()}

def pearson_calculation(rating_list_business,rating_list_neighbor,numerator,deno_business,deno_neighbour,avg_neighbor,avg_business):
    for i in range(len(rating_list_business)):
        numerator += (rating_list_business[i]-avg_business)*(rating_list_neighbor[i]-avg_neighbor)
        deno_business += (rating_list_business[i]-avg_business)**2
        deno_neighbour += (rating_list_neighbor[i]-avg_neighbor)**2

    return numerator, deno_business, deno_neighbour
def pearson(neighbour,user,business,business_avg):
    business_rating_lst = []
    neighbour_rating_lst = []
    business_rating_neighbour = business_rating_info.get(neighbour)
    avg_business_rating_neighbour = business_rating_avg_info.get(neighbour)
    for id in user:
        if business_rating_neighbour.get(id) == True:
            business_score = business.get(id)
            neighbour_score = business_rating_neighbour.get(id)
            business_rating_lst.append(business_score)
            neighbour_rating_lst.append(neighbour_score)

    if len(business_rating_lst)>0:
        numerator_default, business_denominator_default,neighbour_denominator_default = 0,0,0
        numerator_use, business_denominator, neighbour_denominator = pearson_calculation(business_rating_lst,neighbour_rating_lst,numerator_default,business_denominator_default,neighbour_denominator_default,avg_business_rating_neighbour,business_avg)
        denominator_overall = (business_denominator*neighbour_denominator)**0.5
        if numerator_use != 0 and denominator_overall !=0:
            pearson_coef = numerator_use/denominator_overall
        elif numerator_use == 0 and denominator_overall !=0:
            pearson_coef = 0
        elif numerator_use == 0 and denominator_overall ==0:
            pearson_coef = 1
        else:
            pearson_coef = -1
    else:
        pearson_coef = float(business_avg/avg_business_rating_neighbour)

    return pearson_coef


def prediction_coef_calculation(pearson_coef_lst):
    numerator = 0
    denominator = 0
    pearson_coef_lst = sorted(pearson_coef_lst,key=lambda x:-x[0])
    for i in range(len(pearson_coef_lst)):
        numerator += pearson_coef_lst[i][0]*pearson_coef_lst[i][1]
        denominator += abs(pearson_coef_lst[i][0])
    pred_value = numerator/denominator
    return pred_value


def final_calculation(rdd):
    user_info, business_info = rdd[0], rdd[1]
    if business_info in business_rating_info:
        avg_business_rating = business_rating_avg_info.get(business_info)
        user = list(business_rating_info.get(business_info))
        user_single = business_rating_info.get(business_info)
        if user_rating_info.get(user_info) is not None:
            user_rating_info_lst = list(user_rating_info.get(user_info))
            if len(user_rating_info_lst)>0:
                pearson_lst = []
                for i in user_rating_info_lst:
                    cur_neighbor_score = business_rating_info.get(i).get(user_info)
                    pearson_coef = pearson(i,user,user_single,avg_business_rating)
                    if pearson_coef > 0:
                        if pearson_coef>1:
                           pearson_coef = 1/pearson_coef
                        pearson_lst.append((pearson_coef,cur_neighbor_score))
                #ck(pearson_lst,user_info,user_rating_info_lst,business_rating_info,user,user_single,avg_business_rating)
                pred_coef = prediction_coef_calculation(pearson_lst)
                return pred_coef
            else:
                return avg_business_rating
        else:
            return avg_business_rating
    else:
        return str(user_rating_avg_info.get(user_info))

final_result = test_use.map(final_calculation).collect()
cf_coef = np.asarray(final_result,dtype='float')
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

prediction_coef = xgb_used.predict(x_test)
test_dataset = create_test_dataset(test,True)

combined_coef = 0.975*prediction_coef + (1-0.975)*cf_coef
results = np.c_[test_dataset[:,:2],combined_coef]

def csv_writing(path,input):
    file = open(path,mode='w')
    ww = csv.writer(file,delimiter=',',quoting=csv.QUOTE_MINIMAL)
    ww.writerow(['user_id','business_id','prediction'])
    for i in input:
        ww.writerow([str(i[0]), str(i[1]), float(i[2])])
    file.close()

csv_writing(output_file_name,results)
end_time = time.time()-start_time
print("Duration: "+str(end_time))





