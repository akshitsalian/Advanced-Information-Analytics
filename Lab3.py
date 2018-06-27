# Databricks notebook source
# MAGIC %md # Homework 3: Sarcasm prediction
# MAGIC 
# MAGIC In this homework, we are going to detect sarcasm.

# COMMAND ----------

from pyspark.ml import feature
from pyspark.ml import classification
from pyspark.ml import Pipeline
from pyspark.ml import evaluation
from pyspark.sql import functions as fn
import pandas as pd

import requests
from pyspark.sql import Row
rdd_raw = sc.parallelize(requests.get('https://raw.githubusercontent.com/daniel-acuna/python_data_science_intro/master/data/sarcasm-dataset.txt').content.split('\n'))
sarcasm_df = rdd_raw.filter(lambda x: len(x)>1).map(lambda x: Row(text=x[:-2], sarcastic=int(x[-1]))).toDF()

# COMMAND ----------

display(sarcasm_df)

# COMMAND ----------

# the data is almost balanced
sarcasm_df.groupBy('sarcastic').count().show()

# COMMAND ----------

# use these splits throught the notebook
training, validation, testing = sarcasm_df.randomSplit([0.6, 0.3, 0.1])
subtraining1, subtraining2 = training.randomSplit([0.6, 0.4])

# COMMAND ----------

# MAGIC %md 
# MAGIC ** Question 1 (30 pts)** This dataset is based on tweets and therefore sarcasm is sometimes represented by the hashtag "#not". Build a simple classifier that predicts sarcasm when the text contains such hashtag. Estimate the accuracy of the classifier (you don't need to split the data into training because there is no training!) **Hint**: This can be solved using the function `fn.instr` to check if a string is inside another. This function returns 0 if nothing is found.

# COMMAND ----------

a=fn.instr(sarcasm_df['text'],'#not')

# COMMAND ----------

display(sarcasm_df)

# COMMAND ----------

predicted_sarcasm_df

# COMMAND ----------

# your code here
predicted_sarcasm_df=sarcasm_df.\
withColumn("predicted",fn.when(fn.instr(sarcasm_df['text'],'#not' )!=0,1).otherwise(0))
predicted_sarcasm_df.show(10)

#predicting accuracy
predicted_sarcasm_df.select(fn.expr('int(sarcastic = predicted)').alias('correct')).select(fn.avg('correct')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ** Question 2 (40 pts)** Build and evaluate the performance of a classifiers of sarcasm using elastic net regularized logistic regression on the TFIDF representation of the text. Compare three models with \\( \alpha \in (0, 0.05, 0.1 ) \\) and fixed \\( \lambda = 0.1 \\) Use accuracy to evaluate generalization performance. Using the best model, show the words that have the highest and lowest weights on the prediction (don't show words with weights 0) *Hint*: Follow the steps of the sentiment analysis notebook

# COMMAND ----------

# your code here
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.classification import LogisticRegression
#splitting the words in  a review
tokenizer = RegexTokenizer().setGaps(False)\
  .setPattern("\\p{L}+")\
  .setInputCol("text")\
  .setOutputCol("words")
#obtaining stop words
import requests
stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()
#filtering the words by removing stop words
from pyspark.ml.feature import StopWordsRemover
sw_filter = StopWordsRemover()\
  .setStopWords(stop_words)\
  .setCaseSensitive(False)\
  .setInputCol("words")\
  .setOutputCol("filtered")
#remove words that appear in 5 docs or less
cv = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17)\
  .setInputCol("filtered")\
  .setOutputCol("tf")
#creating a pipelined transformer
cv_pipeline = Pipeline(stages=[tokenizer, sw_filter, cv]).fit(sarcasm_df)
#making the transformation between the raw text and the counts
cv_pipeline.transform(sarcasm_df).show(5)
#building a pipeline to take the output of the previous pipeline and lower the terms of documents that are very common
idf = IDF().\
    setInputCol('tf').\
    setOutputCol('tfidf')
idf_pipeline = Pipeline(stages=[cv_pipeline, idf]).fit(sarcasm_df)
idf_pipeline.transform(sarcasm_df).show(5)
tfidf_df = idf_pipeline.transform(sarcasm_df)



# COMMAND ----------

#creating a logistic model with a= 0, lambda= 0.1 for model 1
lr1 = LogisticRegression().\
    setLabelCol('sarcastic').\
    setFeaturesCol('tfidf').\
    setRegParam(0.1).\
    setMaxIter(100).\
    setElasticNetParam(0.)
#creating a pipeline for model 1
lr_pipeline = Pipeline(stages=[idf_pipeline, lr1]).fit(training)
#estimating accuracy of model 1
lr_pipeline.transform(validation).\
    select(fn.expr('float(prediction = sarcastic)').alias('correct')).\
    select(fn.avg('correct')).show()

# COMMAND ----------

lr_pipeline = Pipeline(stages=[idf_pipeline, lr1]).fit(validation)

# COMMAND ----------

display(lr_pipeline.stages[-1])

# COMMAND ----------

#creating a logistic model with a= 0.05, lambda= 0.1 for model 2
lr2 = LogisticRegression().\
    setLabelCol('sarcastic').\
    setFeaturesCol('tfidf').\
    setRegParam(0.1).\
    setMaxIter(100).\
    setElasticNetParam(0.05)
#creating a pipeline for model 2
lr_pipeline = Pipeline(stages=[idf_pipeline, lr2]).fit(training)
#estimating accuracy of model 2
lr_pipeline.transform(validation).\
    select(fn.expr('float(prediction = sarcastic)').alias('correct')).\
    select(fn.avg('correct')).show()

# COMMAND ----------

#creating a logistic model with a= 0.1, lambda= 0.1 for model 3
lr3 = LogisticRegression().\
    setLabelCol('sarcastic').\
    setFeaturesCol('tfidf').\
    setRegParam(0.1).\
    setMaxIter(100).\
    setElasticNetParam(0.1)
#creating a pipeline for model 3
lr_pipeline = Pipeline(stages=[idf_pipeline, lr3]).fit(training)
#estimating accuracy of model 3
lr_pipeline.transform(validation).\
    select(fn.expr('float(prediction = sarcastic)').alias('correct')).\
    select(fn.avg('correct')).show()

# COMMAND ----------

#showing weights of best model 3 in elastic net regularztaion
# show weights
import pandas as pd
vocabulary = idf_pipeline.stages[0].stages[-1].vocabulary
weights = lr_pipeline.stages[-1].coefficients.toArray()
coeffs_df = pd.DataFrame({'word': vocabulary, 'weight': weights})
#lowest weighted words
coeffs_df.sort_values('weight').head(5)
#highest weighted words
coeffs_df.sort_values('weight', ascending=False).head(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC **Question 3** (10 pts) Using the same features on the same training, validation, and testing splits from the previous question, fit a random forest classifier with 500 trees and evaluate its performance using accuracy. Is it better or worse than elatic net logistic regression?

# COMMAND ----------

# your code here
from pyspark.ml.classification import RandomForestClassifier
#creating random forest classifier
rf = RandomForestClassifier(numTrees=500).\
setLabelCol('sarcastic').\
    setFeaturesCol('tfidf')
rf_pipeline = Pipeline(stages=[idf_pipeline, rf]).fit(training)
#calculating accuracy
rf_pipeline.transform(validation).\
    select(fn.expr('float(prediction = sarcastic)').alias('correct')).\
    select(fn.avg('correct')).show()
#random forest classifier is  better than elastic net regression since it has an  increase in accuracy

# COMMAND ----------

# MAGIC %md 
# MAGIC **Question 4** (20 pts) Suppose that you want to catch every sarcastic tweet out here and therefore you care about high _recall_ rather than accuracy. Re-run the models from questions 1, 2 and 3 and estimate their recall on validation. Is the recall criteria changing which model is the best compared to the accuracy criteria?

# COMMAND ----------

# your code
def recall(df):
  
    tp = df[(df.sarcastic == 1) & (df.predicted == 1)].count()
    fn = df[(df.sarcastic == 1) & (df.predicted == 0)].count()
    recall= float(tp)/(tp+fn)
    print recall

# COMMAND ----------

#recall of model in question 1
print(recall(predicted_sarcasm_df))

# COMMAND ----------

def recall(df):
  
    tp = df[(df.sarcastic == 1) & (df.prediction== 1)].count()
    fn = df[(df.sarcastic == 1) & (df.prediction == 0)].count()
    recall= float(tp)/(tp+fn)
    print recall

# COMMAND ----------

#recall of best model for  elatsic net regularization
recall(lr_pipeline.transform(validation))

# COMMAND ----------

#recall of random forest model
recall(rf_pipeline.transform(validation))
#recall on validation is changing with models, the best model compared to accuracy is model 3 of elastic net regularization
