
import os
from math import log

import pandas as pd
from pyspark.sql import SparkSession

from sklearn.datasets import load_iris

from operator import add



# data = load_iris()
# x = data.data
# y = data.target
# x_pd = pd.DataFrame(x,columns=range(x.shape[1]))
# y_pd = pd.DataFrame(y,columns=['label'])
# df_pf = pd.concat([x_pd,y_pd],axis=1)

def calc_entropy(c_counts,total):
    entropy = 0
    for c_name, c_count in c_counts.items():
        p_c = float(c_count) / float(total)
        entropy += p_c * log(p_c, 2)
    return entropy * -1

def calc_info_gain(dataset, home_path,spark):
    print "info gain on data " + dataset
    df_pd = pd.read_csv(os.path.join(home_path,dataset +".csv"))
    df_pd = df_pd.dropna()
    cols = list(df_pd.columns)
    cols[-1] = 'label'
    df_pd.columns = cols
    # for c in cols:
    #     df_pd[c].astype(str)
    df = spark.createDataFrame(df_pd)

    cols = spark.sparkContext.broadcast(df_pd.columns[:-1])

    #use map reduce to calculate the count of each label and column values
    value_counts = df.rdd.flatMap(lambda x: [((x['label'],c, x[c]),1) for c in cols.value]).cache().reduceByKey(lambda x, y: x + y).collectAsMap()

    #caclulate dictionary for info gain calculations
    label_counts = {}
    columns_counts = {}
    total_count = 0
    for value_k, value_count in value_counts.items():
        total_count += value_count

        if value_k[0] not in label_counts:
            label_counts[value_k[0]] = 0
        label_counts[value_k[0]] += value_count

        if value_k[1] not in columns_counts:
            columns_counts[value_k[1]] = {}

        if value_k[2] not in columns_counts[value_k[1]]:
            columns_counts[value_k[1]][value_k[2]] = {}
            columns_counts[value_k[1]][value_k[2]]['label'] = {}
            columns_counts[value_k[1]][value_k[2]]['total'] = 0
        columns_counts[value_k[1]][value_k[2]]['total'] += value_count

        if value_k[0] not in columns_counts[value_k[1]][value_k[2]]:
            columns_counts[value_k[1]][value_k[2]]['label'][value_k[0]] = 0

        columns_counts[value_k[1]][value_k[2]]['label'][value_k[0]] += value_count

    # caclulate info gain base on dictionary
    entropy_s = calc_entropy(label_counts,total_count)
    print "entropy before split: " + str(entropy_s)
    columns_entropy = {}
    for column_name, values_columns_counts in columns_counts.items():
        entropy_c = 0
        for value_name, value_counts in values_columns_counts.items():
            entropy_v = calc_entropy(value_counts['label'], value_counts['total'])
            entropy_c += entropy_v*(float(value_counts['total'])/float(total_count))
        entropy_c = entropy_s - entropy_c
        columns_entropy[column_name] = entropy_c

    i=1
    for key, value in sorted(columns_entropy.iteritems(), key=lambda (k, v): (v, k), reverse = True):
        if i > 10:
            break
        if i == 1:
            print "feature selected to split %s" % (key)
        print "top %i feature infogain: %s - %s" % (i,key, value)
        i +=1

    return columns_entropy


import numpy as np
os.environ['HADOOP_USER_NAME'] = 'lior'  # to avoid Permissio denied: user=root, access=WRITE, inode="/user":hdfs:supergroup:dr

spark = SparkSession \
    .builder \
    .appName('dstamp') \
    .config("spark.sql.warehouse.dir", "/user/hive/warehouse") \
    .config("spark.dynamicAllocation.maxExecutors", "6") \
    .enableHiveSupport() \
    .getOrCreate()

home_path = 'datasets'
datasets = ['cars', 'mushrooms', 'scale', 'specs', 'tictac']
for dataset in datasets:
    calc_info_gain(dataset,home_path,spark)