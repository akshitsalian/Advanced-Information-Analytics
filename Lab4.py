# Databricks notebook source
# MAGIC %md # Detecting happiness

# COMMAND ----------

from pyspark.sql import functions as fn
from pyspark.ml import classification, evaluation, Pipeline, feature
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %%sh
# MAGIC # download and load dataset
# MAGIC wget https://github.com/daniel-acuna/python_data_science_intro/raw/master/data/emotions.parquet.zip -nv
# MAGIC unzip emotions.parquet.zip

# COMMAND ----------

# read data
# pixels: 48x48 pixel gray values (between 0 and 255) 
emotions = spark.read.parquet('file:///databricks/driver/emotions.parquet')

# COMMAND ----------

# utility function to display the first element of a Spark dataframe as an image
def display_first_as_img(df):
    plt.figure()
    plt.imshow(df.first().pixels.reshape([48,48]), 'gray');
    display()

# COMMAND ----------

# show random faces
display_first_as_img(emotions.where('is_happy=0').orderBy(fn.rand()))

# COMMAND ----------

# we will first balance the data so that we have 50% of faces is_happy = 1 and 50% faces is_happy = 0
n_min = min(emotions.where('is_happy == 1').count(), emotions.where('is_happy == 0').count())
balanced_data = emotions.where('is_happy == 1').limit(n_min).\
  union(
  emotions.where('is_happy == 0').limit(n_min)
  )


# COMMAND ----------

# check that it is balanced
display(balanced_data.groupBy('is_happy').count())

# COMMAND ----------

# use these splits throughout the homework
training, validation, testing = balanced_data.randomSplit([0.6, 0.3, 0.1])

# COMMAND ----------

# MAGIC %md 
# MAGIC **Question 1 (40 pts):** Choose the best model based on accuracy between multilayer perceptrons predicting `is_happy` based on `pixels`. Compare the following architectures:
# MAGIC 
# MAGIC  - No hidden layers
# MAGIC  - One hidden layer with 10 neurons
# MAGIC  - Two Hidden layers with 10 neurons each
# MAGIC 
# MAGIC Fit both models to `training` and estimate `accuracy` on validation. You don't need to build a Pipeline because the features needed are in the column `pixels`. Pick the best one based on validation performance. The input dimension is 2304 (=48\*48) and the output is 2

# COMMAND ----------

# model definitions
mlp = classification.MultilayerPerceptronClassifier(seed=0).\
    setStepSize(0.2).\
    setMaxIter(200).\
    setFeaturesCol('pixels').\
    setLabelCol('is_happy').setLayers([48*48, 2])

# COMMAND ----------

# fitting
mlp1_model = mlp.fit(training)

# COMMAND ----------

# evaluations
mlp1_model.transform(validation).select(fn.expr('avg(float(is_happy=prediction))').alias('accuracy')).show()

# COMMAND ----------

# model definitions
mlp = classification.MultilayerPerceptronClassifier(seed=0).\
    setStepSize(0.2).\
    setMaxIter(200).\
    setFeaturesCol('pixels').\
    setLabelCol('is_happy').\
    setLayers([48*48,10, 2])


# COMMAND ----------

# fitting
mlp_simple_model = mlp.fit(training)

# COMMAND ----------

# evaluations
mlp_simple_model.transform(validation).select(fn.expr('avg(float(is_happy=prediction))').alias('accuracy')).show()

# COMMAND ----------

# model definitions
mlp = classification.MultilayerPerceptronClassifier(seed=0).\
    setStepSize(0.2).\
    setMaxIter(200).\
    setFeaturesCol('pixels').\
    setLabelCol('is_happy').\
    setLayers([48*48,10,10, 2])


# COMMAND ----------

# fitting
mlp_simple_model = mlp.fit(training)

# COMMAND ----------

# evaluations
mlp_simple_model.transform(validation).select(fn.expr('avg(float(is_happy=prediction))').alias('accuracy')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC BEST MODEL IS ONE WITHOUT ANY LAYERS

# COMMAND ----------

# MAGIC %md **Question 2 (30 pts):** Using the boilerplate code provided below, find four images of people's faces online (the image URLs) one for each of case corresponding to true positive, true negative, false positive, and false negative. Use the best model from Question 1. Remember that we are predicting whether or not someone is smiling, and therefore true positive is "the model predicts is_happy and the person looks happy", false negative is "the model predict is_happy = 0 but the person looks happy". The images that you find must be fair in that they must be similar to the ones in the training dataset: a frontal face image of people smiling and neutral. The professor provides one true positive URL and one true negative URL as examples

# COMMAND ----------

# BOILER PLATE CODE - DONT MODIFY
from PIL import Image
import requests
from io import BytesIO
from pyspark.ml.linalg import Vectors
import numpy as np

def get_image(url):
  # face
  response = requests.get(url)
  img = Image.open(BytesIO(response.content))
  shrinked_img = np.array(img.resize([48, 48]).convert('P'))
  return shrinked_img

def display_image(url):
  plt.figure()
  plt.imshow(get_image(url), 'gray')
  display()
  
def predict_image(model, url):
  new_image = get_image(url).flatten()
  new_img_df = spark.createDataFrame([[Vectors.dense(new_image)]], ['pixels'])
  return model.transform(new_img_df)

# COMMAND ----------

# POSITIVE
smiling_person = 'https://blog.zoom.us/wordpress/wp-content/uploads/2013/09/78813113-1024x682.jpg'
display_image(smiling_person)

# COMMAND ----------

# TRUE POSITIVE 
predict_image(mlp1_model, smiling_person).show()

# COMMAND ----------

# NEGATIVE
neutral_face = "http://i2.wp.com/detourphotography.ca/wp-content/uploads/2015/11/PPK_0907-Edit.jpg?resize=296%2C446"
display_image(neutral_face)

# COMMAND ----------

# TRUE NEGATIVE
predict_image(mlp1_model, neutral_face).show()

# COMMAND ----------

# Find (e.g., Google Images) URLs for 1 true positive, 1 true negative, 1 false positive, and 1 false negative.

# COMMAND ----------

#True Positive
true_positive = 'https://i.pinimg.com/736x/20/05/48/200548541b488399cd12fcc1bd0c7edb--smile-face-a-smile.jpg'
display_image(true_positive)

# COMMAND ----------

predict_image(mlp1_model, true_positive).show()

# COMMAND ----------

#True Negative
true_negative = 'https://i.ytimg.com/vi/lClsYZebzyw/hqdefault.jpg'
display_image(true_negative)

# COMMAND ----------

predict_image(mlp1_model, true_negative).show()

# COMMAND ----------

#False Positve
false_positive = 'https://assets.lookbookspro.com/atelier-management/gs_5988cb0e-7890-48e1-96f5-34d7ac110004.jpg'
display_image(false_positive)

# COMMAND ----------

predict_image(mlp1_model, false_positive).show()

# COMMAND ----------

#False Negative
false_negative = 'https://thumbs.dreamstime.com/b/baby-girl-smiling-portrait-little-black-white-square-67277455.jpg'
display_image(false_negative)

# COMMAND ----------

predict_image(mlp1_model, false_negative).show()

# COMMAND ----------

# MAGIC %md **Question 3 (30 pts)**: Study neural network architectures to fit the infamous concentric circles dataset. There is boilerplate code to do the data generation, plotting and evaluation. Play with number of hidden layers and with the number of neurons per hidden layer. The input dimension is 2 and the output dimension is 2, but how many hidden layers and hidden neurons are needed to achieve more than 95% accuracy?

# COMMAND ----------

# BOILERPLATE CODE
from sklearn import manifold, datasets
from pyspark.sql import Row
X, y = datasets.make_circles(n_samples=300, factor=.6, noise=.1, random_state=0)
data = spark.createDataFrame(  [Row(x=float(x[0]), y=float(x[1]), label=int(label)) for x, label in zip(X, y)])

plotting_data = spark.range(100).selectExpr("(id/100)*3 - 1.5 as x").\
  crossJoin(spark.range(100).selectExpr("(id/100)*3 - 1.5 as y"))
  
def fit_and_plot(estimator):
  """Plot the data and decision surface of estimator"""
  va= feature.VectorAssembler(inputCols=['x', 'y'], outputCol='features')
  df = va.transform(plotting_data)
  model = estimator.fit(va.transform(data))
  pp = model.transform(df).select('x', 'y', 'prediction').toPandas()
  
  fig, ax = plt.subplots(figsize=(5,5))
  plt.contourf(pp.x.unique(), pp.y.unique(), pp.prediction.reshape(pp.x.unique().shape[0], pp.x.unique().shape[0]), alpha=0.5)
  colors=['blue', 'red']
  for i, grp in data.toPandas().groupby('label'):
    grp.plot(x='x', y='y', kind='scatter', ax=ax,color=colors[i])  
  
  acc = model.transform(va.transform(data)).selectExpr('avg(CAST((label = prediction) AS FLOAT)) AS avg').first().avg
  plt.title('Accuracy = {}'.format(acc))
  display()

# COMMAND ----------

# this is an example with a logistic regression model
fit_and_plot(classification.LogisticRegression())

# COMMAND ----------

# try with multilayer perceptron
fit_and_plot(classification.MultilayerPerceptronClassifier(layers=[2,4,4,2]))

# COMMAND ----------

# MAGIC %md # **Extra credit (10 pts) ** 
# MAGIC 
# MAGIC ### beat the professor!
# MAGIC Starting from the following Tensorflow Playground [here](http://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.00001&regularizationRate=0&noise=0&networkShape=4,2&seed=0.00846&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false),
# MAGIC 
# MAGIC change any of the following parameters:
# MAGIC 
# MAGIC - Learning rate
# MAGIC - Activation
# MAGIC - Regularization
# MAGIC - Regularization rate
# MAGIC - Number of hidden layers
# MAGIC - Number of neurons per hidden layer
# MAGIC 
# MAGIC Usually, after you change any of the allowable parameters, you will need to restart the learning by pressing the *reset network* button and then pressing the *play* button.
# MAGIC 
# MAGIC But DO NOT CHANGE any of the following:
# MAGIC 
# MAGIC - Data
# MAGIC - Ratio of training to test data
# MAGIC - Noise
# MAGIC - Batch size
# MAGIC - Features
# MAGIC 
# MAGIC Report the parameters you use to achieve less than **0.01 test loss before 20,000 epochs**

# COMMAND ----------

# parameters used to beat the professor: Test Loss = 0.007 before 3000 epochs.
# - Learning rate: 0.01
# - Activation: ReLU
# - Regularization: None
# - Regularization rate: 
# - Number of hidden layers: 2
# - Number of neurons in hidden layers: [8,8]

# COMMAND ----------


