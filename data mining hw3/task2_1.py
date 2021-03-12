import json
import time
from itertools import combinations
from pyspark import SparkConf, SparkContext
from operator import add
import csv
import math
import random
import sys
start_time = time.time()
train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]
conf = SparkConf().setAppName("553hw3").setMaster('local[*]')
sc = SparkContext(conf=conf)

train = sc.textFile(train_file_name)
header = train.first()
train = train.filter(lambda i: i!=header).map(lambda i:i.split(","))

val = sc.textFile(test_file_name)
header_val = val.first()
valid = val.filter(lambda i: i!=header_val).map(lambda i:i.split(",")).sortBy(lambda i:((i[0],i[1]))).persist()

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
#print(user_rating_avg_info)
#print(business_rating_avg_info)
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


def ck(given_lst,rdd_0,given_user,given_business,p1,p2,p3):
    for i in given_user:
        cur_neighbor_score = given_business.get(i).get(rdd_0)
        pearson_coef = pearson(i, p1, p2, p3)
        if pearson_coef > 0:
            if pearson_coef>1:
                pearson_coef = 1/pearson_coef
        given_lst.append((pearson_coef, cur_neighbor_score))
    return given_lst



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
                return user_info,business_info,pred_coef
            else:
                return user_info,business_info, avg_business_rating
        else:
            return user_info, business_info, avg_business_rating
    else:
        return user_info,business_info,str(user_rating_avg_info.get(user_info))


final_result = valid.map(final_calculation).collect()
#print(final_result)

def csv_writing(path,input):
    file = open(path,mode='w')
    write = csv.writer(file,delimiter=',',quoting=csv.QUOTE_MINIMAL)
    write.writerow(['user_id','business_id','prediction'])
    for i in input:
        write.writerow([str(i[0]), str(i[1]), i[2]])
    file.close()
csv_writing(output_file_name,final_result)
end_time = time.time()-start_time
print("Duration: "+str(end_time))






