# Databricks notebook source
# ALWAYS IMPORT THESE PACKAGES AT THE BEGINNING
from __future__ import division, absolute_import
from pyspark.sql import Row
from pyspark.ml import regression
from pyspark.ml import feature
from pyspark.ml import Pipeline
from pyspark.sql import functions as fn

# COMMAND ----------

# MAGIC %md # Part 1. MapReduce

# COMMAND ----------

# MAGIC %md 1.1 (30 pts)
# MAGIC 
# MAGIC Sometimes, it is useful to summarize a set of numbers into binned frequencies. For example, to understand the distribution of age in a population, we would like to know how many people are between 0-20, how many between 20-40, how many between 40-60, and so on.
# MAGIC 
# MAGIC Create a map-reduce job with Spark that takes an RDD with each input element being a tuple (label, value). For each label, the map-reduce job must generate a list with the counts of how many values fall in each of the following ranges: `x < 0`, `0<= x <20`, `20 <= x < 40`, `40 <= x < 60`, and `60 <= x`. This is, the map-reduce job must generate a list of length 5 where the first element contains the frequency where `x < 0`, the second element contains the number of values where `0 <= x < 20`, and so on. You only need one map and one reduce operation. Use vanilla Python lists.
# MAGIC 
# MAGIC For example, for the following RDD
# MAGIC 
# MAGIC ```
# MAGIC rdd = sc.parallelize([('healthy', -100),
# MAGIC ('healthy', 0.),
# MAGIC ('healthy', 12.),
# MAGIC ('sick', 10.),
# MAGIC ('sick', 58.),
# MAGIC ('sick', 60),
# MAGIC ('sick', 100),
# MAGIC ])
# MAGIC ```
# MAGIC 
# MAGIC The result of `rdd.map(map_bin).reduceByKey(reduce_bin).collect()` should be:
# MAGIC 
# MAGIC ```
# MAGIC [('healthy', [1, 2, 0, 0, 0]), ('sick', [0, 1, 0, 1, 2])]
# MAGIC ```

# COMMAND ----------

# define map and reduce functions here
def map_bin(x):
  L=[0,0,0,0,0]
  
  if(x[0]<0):
    L[0]=L[0]+1
  elif(x[1]>=0 and x[1]<20):
    L[1]=L[1]+1
  elif(x[1]>=20 and x[1]<40):
    L[2]=L[2]+1
  elif(x[1]>=40 and x[1]<60):
    L[3]=L[3]+1
  elif(x[1]>=60):
    L[4]=L[4]+1
  return [x[0],L]

def reduce_bin(v1, v2):
  
  return [x+L for x,L in zip(v1,v2)]

# COMMAND ----------

# apply your map-reduce job to the following RDD
rdd = sc.parallelize([('low', 8), ('low', -2), ('low', -7), ('low', 11), ('low', 5), ('low', -8), ('low', 14), ('low', 9), ('low', 8), ('low', -6), ('low', 11), ('low', 8), ('low', 13), ('low', -10), ('low', 10), ('low', -4), ('low', -5), ('low', 6), ('low', 13), ('low', -3), ('unknown', 104), ('unknown', 130), ('unknown', 57), ('unknown', 50), ('unknown', 12), ('unknown', 110), ('unknown', 65), ('unknown', 66), ('unknown', 47), ('unknown', 96), ('high', 45), ('high', 44), ('high', 50), ('high', 45), ('high', 50), ('high', 44), ('high', 45), ('high', 46), ('high', 43), ('high', 52), ('high', 51), ('high', 46), ('high', 52), ('high', 53), ('high', 50), ('middle', 19), ('middle', 25), ('middle', 27), ('middle', 40), ('middle', 13), ('middle', 15), ('middle', 27), ('middle', 26), ('middle', 19), ('middle', 23)])

# COMMAND ----------

rdd.map(map_bin).reduceByKey(reduce_bin).collect()

# COMMAND ----------

# MAGIC %md # Part 2: Preprocess data and create dataframes

# COMMAND ----------

# MAGIC %md Sometimes, we must preprocess messy data through several steps. 
# MAGIC 
# MAGIC Consider the dataset in `/databricks-datasets/sample_logs`. This dataset contains access logs to an Apache webserver. Take for example the following line
# MAGIC 
# MAGIC `3.3.3.3 - user1 [21/Jun/2014:10:00:00 -0700] "GET /endpoint_27 HTTP/1.1" 200 21`
# MAGIC 
# MAGIC where the format is as follows:
# MAGIC 
# MAGIC 1. the IP
# MAGIC 2. user name if authenticated or `-` if not authenticated
# MAGIC 3. Time stamp of the access with time zone
# MAGIC 4. Method, endpoint, and protolcol
# MAGIC 5. Request status. See more here https://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html
# MAGIC 6. Size of the object returned to client in bytes
# MAGIC 
# MAGIC You can read more details about the structure of Apache logs here https://httpd.apache.org/docs/2.4/logs.html#accesslog

# COMMAND ----------

# we will read the logs into an RDD
rdd = sc.textFile('/databricks-datasets/sample_logs')

# COMMAND ----------

rdd.take(10)

# COMMAND ----------

# MAGIC %md 2.1 (10 pts) **Transform the RDD** Transform the original RDD from line into list of length 8 with the following structure:
# MAGIC 
# MAGIC 1. Index 0: String with the IP
# MAGIC 2. Index 1: A number 1 if there is a username and 0 if there no username (i.e., `-`)
# MAGIC 3. Index 2: A number representing the month, starting from `Jan` = 0, `Feb` = 1, ..., `Dec` = 11
# MAGIC 4. Index 3: A number 1 if the method is `GET` and 0 otherwise.
# MAGIC 5. Index 4: String with the endpoint (e.g., `"/endpoint_27"`)
# MAGIC 6. Index 5: String with the protocol (e.g., `"HTTP/1.1"`)
# MAGIC 7. Index 6: Integer with the response (e.g., `200`)
# MAGIC 8. Index 7: Integer with response size (e.g., `21`)
# MAGIC 
# MAGIC Hints: You can transform a string into an integer by applying the function `int`. For example, `int("21")` becomes `21`. Also, remember that to concatenate strings you can use the `+` operator

# COMMAND ----------

def split_line(line):
  L1=[]
  L2=[0,0,0,0,0,0,0,0]
  month_data=dict(Jan=0,Feb=1,Mar=2,Apr=3,May=4,Jun=5,Jul=6,Aug=7,Sep=8,Oct=9,Nov=10,Dec=11)
  
  L1=line.split(" ")
  
  L2[0]=L1[0]
  if(L1[2]=='-'):
    L2[1]=0
  else:
    L2[1]=1
    
  month=L1[3].split("/")[1]
  L2[2]=month_data[month]
  
  if(L1[5]=="\"GET"):
    L2[3]=1
  else:
    L2[3]=0
    
  L2[4]=L1[6]
  L2[5]=L1[7]
  L2[6]=int(L1[8])
  L2[7]=int(L1[9])
  return L2

# COMMAND ----------

rdd.take(10)

# COMMAND ----------

rdd.map(split_line).take(10)

# COMMAND ----------

# MAGIC %md 2.2 (10 pts) **Exploratory analysis** Sometimes, we want to understand where most errors happen. We will define an "error" as any response code greater or equal to 300 and not error as everything else. Using map-reduce, estimate the frequency of errors per month and user presence, with month and user presence encoded as in question 2.1. *Hint*: The key should be a tuple `(has_user, n_month)` where `has_user` is 0 or 1 and `n_month` is in `[0, ..., 11]`. You should use the function from question 2.1.

# COMMAND ----------

def map_explore(l):
  L1=0
  if(l[6]>=300):
    L1=1
  
  
  return ((l[1],l[2]),L1) 

def reduce_explore(v1, v2):
  return v1+v2

# COMMAND ----------

rdd.map(split_line).map(map_explore).reduceByKey(reduce_explore).collect()

# COMMAND ----------

# MAGIC %md 2.3 (10 pts) **From RDD to DataFrame** From the RDD created in question 2.1, create a DataFrame by using the `toDF()` method of an RDD. This RDD should have a set of `Row` objects with the following names for the fields:
# MAGIC 
# MAGIC 1. Index 0: ip
# MAGIC 2. Index 1: has_user
# MAGIC 3. Index 2: n_month
# MAGIC 4. Index 3: is_get
# MAGIC 5. Index 4: endpoint
# MAGIC 6. Index 5: protocol
# MAGIC 7. Index 6: response
# MAGIC 8. Index 7: size

# COMMAND ----------

from pyspark.sql import Row

# COMMAND ----------

df = rdd.map(split_line).map(lambda x:Row(IP=x[0],has_user=x[1],n_month=x[2],is_get=x[3],endpoint=x[4],protocol=x[5],response=x[6],size=x[7])).toDF()

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md 2.4 (10 pts) **Linear regression** Use the dataframe created in question 2.3 to build a model that uses `has_user`, `n_month`, and `is_get` to predict `size`. Further, print the parameters of the model and estimate the mean squared error of the model on the training data. *Hint:* Use a `Pipeline` of `VectorAssembler` and a `LinearRegression`.

# COMMAND ----------

va = feature.VectorAssembler(inputCols=['has_user','n_month','is_get'], outputCol='features')
lr = regression.LinearRegression(featuresCol='features', labelCol='size')
pipe = Pipeline(stages=[va, lr])

# COMMAND ----------

pipe_model = pipe.fit(df)

# COMMAND ----------

# define symbolic MSE expression using fn package
mse = fn.avg((fn.col('size') - fn.col('prediction'))**2)

# COMMAND ----------

pipe_model.transform(df).select(mse).show()

# COMMAND ----------

# MAGIC %md ## Part 3: Dataframe manipulation and model comparison

# COMMAND ----------

# MAGIC %md For this question, we will use the TPC-H schema, which is a standard schema used for benchmarking big data SQL engines. The full schema is displayed below. An arrow indicates how columns of a table are related to other tables. For example, the column `ORDERKEY` in the dataframe `LINEITEM` refers to the dataframe `ORDERS`. This means that for each line item we can get the order details by joining `LINEITEM` with `ORDERS` by using `ORDERKEY` as the join key.

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://docs.snowflake.net/manuals/_images/sample-data-tpch-schema.png)

# COMMAND ----------

# MAGIC %md We will only create the DataFrames for customer (`customer_df`), orders (`order_df`), and line item (`lineitem_df`):

# COMMAND ----------

customer_df = spark.read.csv('dbfs:/databricks-datasets/tpch/data-001/customer/', sep='|').\
  selectExpr('_c0 as CUSTKEY', 
             '_c1 as NAME',
             '_c2 as ADDRESS',
             'cast(_c3 as float) as NATIONKEY',
             '_c4 as PHONE',
             'cast(_c5 as float) as ACCTBAL',
             '_c6 as MKTSEGMENT',
             '_c7 as COMMMENT')

order_df = spark.read.csv('dbfs:/databricks-datasets/tpch/data-001/orders/', sep='|').\
  selectExpr('_c0 as ORDERKEY',
             '_c1 as CUSTKEY',
             '_c2 as ORDERSTATUS',
             'cast(_c3 as float) as TOTALPRICE',
             '_c4 as ORDERDATE',
             '_c5 as ORDER_PRIORITY',
             '_c6 as CLERK',
             '_c7 as SHIP_PRIORITY',
             '_c8 as COMMENT')

lineitem_df = spark.read.csv('dbfs:/databricks-datasets/tpch/data-001/lineitem/', sep='|').\
  selectExpr('_c0 as ORDERKEY',
             '_c1 as PARTKEY',
             '_c2 as SUPPKEY',
             '_c3 as LINENUMBER',
             '_c4 as QUANTITY',
             '_c5 as EXTENDEDPRICE',
             '_c6 as DISCOUNT',
             '_c7 as TAX',
             '_c8 as RETURNFLAG',
             '_c9 as LINESTATUS',
             '_c10 as SHIPDATE',
             '_c11 as COMMITDATE',
             '_c12 as RECEIPTDATE',
             '_c13 as SHIPINSTRUCT',
             '_c14 as SHIPMODE',
             '_c15 as COMMENT')

# COMMAND ----------

# MAGIC %md 3.1 (5 pts) **Outer joins** Compute how many customers do not have orders. *Hint*: Join the customer and order dataframes and use the appropriate `how` option. Then, select the cases where the `orderkey` is null using the appropriate function from the package `fn` (i.e., no matching order is available). Count the number of rows in that resulting dataframe.

# COMMAND ----------

customer_df.join(order_df,on='CUSTKEY',how='left').filter(fn.isnull('ORDERKEY')).select(fn.count('CUSTKEY')).show()

# COMMAND ----------

# MAGIC %md 3.2 (5 pts) **Summary stats** Estimate how much a customer pay in taxes on average. *Hint:* Link customer, order, and lineitem dataframes.

# COMMAND ----------

customer_df.join(order_df,on='CUSTKEY',how='inner').join(lineitem_df,on='ORDERKEY',how='left').groupBy('CUSTKEY').agg(fn.avg('TAX')).show()

# COMMAND ----------

# MAGIC %md 3.3 (20 pts) **Assessing model accuracy** Use training, validation, and testing splits to build a model that predicts the order's total price (`TOTALPRICE`) using the following models:
# MAGIC 
# MAGIC 1. No features (only intercept)
# MAGIC 2. Customer account balance (`ACCTBAL`)
# MAGIC 3. Customer account balance (`ACCTBAL`) and nation key (`NATIONKEY`)
# MAGIC 
# MAGIC Report the **root mean squared error** (RMSE) of each model for validation (15 pts) and the RMSE of the best model on testing (5 pts). The RMSE is simply the square root of the MSE.
# MAGIC 
# MAGIC *Hint*: You need to build three pipelines and use different  vector assemblers and linear regressions accordingly. Use the `randomSplit` dataframe method to use 60% for training, 30% for validation, and 10% for testing.

# COMMAND ----------

data = customer_df.join(order_df,on='CUSTKEY')

# COMMAND ----------

training_df, validation_df, testing_df = data.randomSplit([0.6, 0.3, 0.1])

# COMMAND ----------

va1=feature.VectorAssembler(inputCols=[], outputCol='features')
va2=feature.VectorAssembler(inputCols=['ACCTBAL'], outputCol='features')
va3=feature.VectorAssembler(inputCols=['ACCTBAL','NATIONKEY'], outputCol='features')

lr = regression.LinearRegression(featuresCol='features', labelCol='TOTALPRICE')

# COMMAND ----------

pipe1=Pipeline(stages=[va1,lr])
pipe2=Pipeline(stages=[va2,lr])
pipe3=Pipeline(stages=[va3,lr])

# COMMAND ----------



# COMMAND ----------

model1 = pipe1.fit(training_df)
model2 = pipe2.fit(training_df)
model3 = pipe3.fit(training_df)

# COMMAND ----------



# COMMAND ----------

# symbolically define RMSE
rmse = fn.sqrt(fn.avg((fn.col('TOTALPRICE') - fn.col('prediction'))**2))

# COMMAND ----------

# model selection
model1.transform(validation_df).select(rmse).show()

# COMMAND ----------

model2.transform(validation_df).select(rmse).show()

# COMMAND ----------

model3.transform(validation_df).select(rmse).show()

# COMMAND ----------

# estimate generalization error
model1.transform(testing_df).select(rmse).show()
