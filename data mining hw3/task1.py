import json
import time
from itertools import combinations
from pyspark import SparkConf, SparkContext
from operator import add
import csv
import random
import sys
start_time = time.time()
input_file = sys.argv[1]
output_file = sys.argv[2]
conf = SparkConf().setAppName("553hw3").setMaster('local[*]')
sc = SparkContext(conf=conf)

def judge_prime_number(potential):
    for i in range(2,int(potential**0.5)+1):
        if potential % i==0:
            return False
    return True
#print(judge_prime_number(106))
def get_prime_number(number):
    for i in range(number+1,number+20000):
        if judge_prime_number(i) == True:
            return i

def create_hash(num):
    a = random.sample(range(1,1500),num)
    b = random.sample(range(1,1500),num)
    p = [get_prime_number(i) for i in random.sample(range(1500,21500),num)]
    hash_lst = []
    for i in range(num):
        hash_lst.append([a[i],b[i],p[i]])
    return hash_lst

def create_matrix(x,hash,m):
    matrix_hash_lst = []
    for x in x[1]:
        hash_value = ((hash[0]*x+hash[1])%hash[2]) % m
        matrix_hash_lst.append(hash_value)
    min_hash = min(matrix_hash_lst)
    return min_hash




train = sc.textFile(input_file)
header = train.first()
train = train.filter(lambda i: i!=header).map(lambda i:i.split(","))
#train = train.map
train_user_count = train.map(lambda i:i[0]).distinct().collect()
#train.sort()
user_count_lst = []

for i in sorted(train_user_count):
    user_count_lst.append(i)
#return len(user_count_lst)
user_count_num = len(user_count_lst)
#user_count_num = user_count(train_user_count)
#print(user_count_num)
#print(user_count_lst)
#matrix = train.map(lambda x: (x[1], user_count_lst)).groupByKey().map(lambda x: (x[0], list(x[1]))).sortBy(lambda x: x[0])
#print(matrix)

#def matrix_construction(rdd):
#    a = rdd.map(lambda i: (i[1], user_count_lst)).groupByKey()
#    b = a.map(lambda i:(i[0], list(i[1]))).sortBy(lambda x: x[0])
#    matrix = b.collectAsMap()
#    return matrix
#matrix_use = matrix_construction(train)

train_user_count.sort()
user_to_number_dict = {}
for index, ele in enumerate(train_user_count):
    user_to_number_dict[ele] = index

#print(len(user_to_number_dict))
user_count = len(user_to_number_dict)
def matrix_construction_jaccard(rdd):
    a = rdd.map(lambda i: (i[1], user_to_number_dict[i[0]])).groupByKey()
    b = a.map(lambda i:(i[0], list(i[1]))).sortBy(lambda x: x[0])
    matrix = b.collectAsMap()
    return matrix

def matrix_construction(rdd):
    a = rdd.map(lambda i: (i[1], user_to_number_dict[i[0]])).groupByKey()
    b = a.map(lambda i:(i[0], list(i[1]))).sortBy(lambda x: x[0])
    return b
#matrix = train.map(lambda x: (x[1], user_to_number_dict[x[0]])).groupByKey().map(lambda x: (x[0], list(x[1]))).sortBy(lambda x: x[0])
#matrix_use = matrix.collectAsMap()
matrix_use_jaccard = matrix_construction_jaccard(train)
matrix = matrix_construction(train)
#print(matrix_use)
Hash = create_hash(100)
bands = 50
rows = int(len(Hash)/bands)
print(rows)
#print(hash)
#user_count = get_specific_user(train)
def sing_matrix_construction(matrix,hash):
    return matrix.map(lambda x: (x[0], [create_matrix(x, i, user_count) for i in Hash]))
#print(matrix)
sing_matrix = sing_matrix_construction(matrix,hash)
#print(sing_matrix)

def local_sensitive_hashing(matrix_cell):
    band_lst = []
    business_value, minhashvalue = [matrix_cell[0]],matrix_cell[1]
    for i in range(bands):
        current_band = tuple(minhashvalue[i * rows:(i + 1) * rows])
        band_lst.append(((i,current_band),business_value))
        i+=1
    return band_lst

def get_pair_candidate(candidate):
    candidate_lst = list(candidate)
    candidate_pair = combinations(candidate_lst,2)
    return sorted(candidate_pair)

candidates = sing_matrix.flatMap(local_sensitive_hashing).reduceByKey(lambda x, y: x + y).filter(lambda x: len(x[1]) > 1).flatMap(lambda x: get_pair_candidate(x[1])).distinct()

#print(type(candidates))
def compute_jaccard(a1,a2):
    intersection = len((set(matrix_use_jaccard[a1]))&(set(matrix_use_jaccard[a2])))
    union = len((set(matrix_use_jaccard[a1]))|(set(matrix_use_jaccard[a2])))
    jaccard_similarity = intersection/union
    return a1,a2,jaccard_similarity
jaccard = candidates.map(lambda x:compute_jaccard(x[0],x[1])).filter(lambda x:x[2]>=0.5)
jaccard = jaccard.sortBy(lambda x: (x[0], x[1])).collect()
#print(jaccard)

#with open("/Users/ruichao/Desktop/task1.csv",'w') as fout:
#    fout.write("business_id_1, business_id_2, similarity\n")
#    for i in jaccard:
#        fout.write(str(jaccard[0]) + "," + str(jaccard[1]) + "," + str(jaccard[2]) + "\n")


def csv_writing(path,input):
    file = open(path,mode='w')
    write = csv.writer(file,delimiter=',',quoting=csv.QUOTE_MINIMAL)
    write.writerow(['business_id_1','business_id_2','similarity'])
    for i in input:
        write.writerow([str(i[0]),str(i[1]),i[2]])
    file.close()

csv_writing(output_file,jaccard)

total_time = time.time()-start_time

print("Duration: " + str(total_time))




















