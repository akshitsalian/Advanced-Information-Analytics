# Databricks notebook source
# MAGIC %md # Homework 1

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part 1. Math
# MAGIC 
# MAGIC **You can submit Part 1 of the homework as a separate document in Blackboard (e.g., as scanned pages or PDF Latex of your derivation) **
# MAGIC 
# MAGIC Assume the following model for predicting the age of a person:
# MAGIC 
# MAGIC $$f() = b$$
# MAGIC 
# MAGIC where b is the only parameter of your model. This is, the model does not use any features and on avereage predicts the same age for everybody. Assume further that the loss function when predicting \\( n\\) data points \\( \\{\text{age}_1, \text{age}_2, ..., \text{age}_n \\} \\) is 
# MAGIC 
# MAGIC $$l(b) = \sum_{i=1}^n (f() - \text{age}_i)^2 + \lambda b^2$$
# MAGIC 
# MAGIC where \\( \lambda \\) is a constant greater than 0.
# MAGIC 
# MAGIC 
# MAGIC ** Question 1.1 (5 pts) ** Calculate the derivative of the loss function with respect to \\( b \\), \\( \frac{d l(b)}{db} \\)
# MAGIC 
# MAGIC ** Question 1.2 (5 pts) ** Estimate the best value for \\( b \\) that minimizes the loss function by solving \\( \frac{d l(b)}{db} = 0 \\)
# MAGIC 
# MAGIC ** Question 1.3 (5 pts) ** Interpret the best value from Question 1.2. This is, (2 pts) how is the best value \\( b \\) different from just taking the average age? and (3 pts) what is the effect of the constant \\( \lambda \\) as it goes to infinity?

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part 2. Basic Python programming

# COMMAND ----------

# MAGIC %md ** Question 2.1 (10 pts) ** Create a function called "cleanse" which checks that the types of all elements in a list are the same type of the first element. If an element is of a different type, it discards it. For example, `cleanse([2, "hi", 4, None, "yes"])` should return `[2, 4]` because 4 is the only element that coincides with the type of the first element. But `cleanse(["hi", 2, 4, None, "yes"])` should return `["hi", "yes"]` because `"yes"`'s type coincides with `"hi"`'s type. To get the type of an expression use the function `type`. You are free to solve this problem with list comprehensions or for-loops. Assume that the input list has one or more elements.

# COMMAND ----------

def cleanse(L):
  t=[]
  for i in range (len(L)):
    if type(L[0])== type(L[i]):
      t.append(L[i])
  return t

# COMMAND ----------

cleanse([2,4,5])==[2,4,5]

# COMMAND ----------

# check your work
cleanse([2, "hi", 4, None, "yes"]) == [2, 4]

# COMMAND ----------

cleanse(["hi", 2, 4, None, "yes"]) == ["hi", "yes"]

# COMMAND ----------

# MAGIC %md **Question 2.2 (10 pts)** Reverse the words in a sentence. Implement a function `scramble_sentence` that takes a sentence as a string and returns a string where the words have been reversed. For example, `scramble_sentence("Hello World")` should return `"olleH dlroW"`. Consider that strings are just like any other sequence such as lists and tuples and therefore can be accessed with the slice notation (e.g., `s[::-1]`). Also you can use the method `slice` of string object to split a sentence into its individual words and `join` to put a list of strings inside another string. For example, `" ".join(["hello", "world"])` will create `"hello world"`. You can use list comprehensions or for-loops to solve this problem.

# COMMAND ----------

def scramble_sentence(s):
  return " ".join(rev[::-1] for rev in s.split())

# COMMAND ----------

# check your work
scramble_sentence("Hello World") == "olleH dlroW"

# COMMAND ----------

scramble_sentence("Data Science") == "ataD ecneicS"

# COMMAND ----------

# MAGIC %md **Question 2.3 (10 pts)** Basic statistics. Create a function `statistic` which receives two parameters: a list of floating point numbers and a statistic. The statistic will be a string containing either `"max"`, `"min"`, or `"average"`. The function will return the provided statistic of the list. For example, `statistic([1.0, 3.5, 6.0], "max")` should return `6.0`, `statistic([1.0, 3.5, 6.0], "min")` should return `1.0`, and `statistic([1.0, 3.5, 6.0], "average")` should return `3.5`. If the statistic provided is not part of min, max, or average, print an error message. Don't assume that the list is a numpy array but rather a vanilla Python list.

# COMMAND ----------

def statistic(L, op):
  if(op=="max"):
    return max(L)
  if(op=="min"):
    return min(L)
  if(op=="average"):
    return sum(L)/len(L)
  else:
    return "error"

# COMMAND ----------

# check your results
statistic([1.0, 3.5, 6.0], "max") == 6.0

# COMMAND ----------

statistic([1.0, 3.5, 6.0], "min") == 1.0

# COMMAND ----------

statistic([1.0, 3.5, 6.0], "average") == 3.5

# COMMAND ----------

statistic([1.0, 3.5, 6.0], "mode")

# COMMAND ----------

# MAGIC %md ## Part 3: Numpy

# COMMAND ----------

# MAGIC %md **Question 3.1 (15 pts)** Estimate the value of \\( \sqrt{2} \\) by simulation. In class, we saw an example of how to estimate \\( \pi \\) using number random generation. In this question, you will estimate the square root of 2 using the same idea. 
# MAGIC 
# MAGIC You can generate uniform random numbers between 0 and 1 using the method `numpy.random.random` in the package `numpy`. For example:

# COMMAND ----------

import numpy.random
x = numpy.random.random(size=5)
print(x)

# COMMAND ----------

# MAGIC %md
# MAGIC `x` has an numpy array of 5 random numbers between 0 and 1. Notice that you can create a list of random numbers between 0 and 2 by simply multiplying `x` by 2:

# COMMAND ----------

x*2

# COMMAND ----------

# MAGIC %md 
# MAGIC We will use this fact to estimate the side *r* of a square whose area is 2. It is easy to notice that \\( r \leq 2 \\) because \\( r^2 = 2 \\). Therefore, we can make this estimation by generating random squares with sides between 0 and 2 with `numpy`. Then we will estimate the ration of times *p* these random squares have an area equal or less to 2. This ratio will represent approximately the ratio \\( \sqrt{2} \\) of 2 (why is this?). Mathematically, \\( p \approx \frac{\sqrt{2}}{2} \\). Use this fact to estimate \\( \sqrt{2} \\)

# COMMAND ----------

# generate random sides between 0 and 2
s = numpy.random.random(5000)*2

# COMMAND ----------

# estimate p
p = (s**2<= 2).mean()

# COMMAND ----------

# use p to estimate sqrt(2)
value=p*2
value

# COMMAND ----------

# MAGIC %md ## Part 4: Matplotlib

# COMMAND ----------

# import needed packages
import matplotlib.pyplot as plt
import numpy as np

# COMMAND ----------

# MAGIC %md Use the following dataset

# COMMAND ----------

# create the data
group = np.array(['young', 'old', 'young', 'young', 'young', 'old', 'young', 'old', 'old', 'young', 'young', 'old', 'young', 'young', 'old', 'old', 'old', 'old', 'old', 'young', 'old', 'old', 'young', 'old', 'old', 'old', 'old', 'old', 'old', 'old', 'young', 'young', 'young', 'old', 'young', 'old', 'old', 'old', 'young', 'young', 'old', 'old', 'young', 'old', 'old', 'young', 'old', 'young', 'old', 'young', 'young', 'young', 'old', 'young', 'old', 'old', 'young', 'young', 'old', 'old', 'young', 'old', 'old', 'young', 'young', 'young', 'young', 'young', 'old', 'old', 'old', 'old', 'young', 'old', 'old', 'young', 'old', 'old', 'old', 'old', 'old', 'old', 'young', 'young', 'old', 'young', 'old', 'old', 'old', 'old', 'young', 'young', 'young', 'old', 'old', 'young', 'old', 'old', 'young', 'young'])
age = np.array([37.0, 41.0, 42.0, 41.0, 36.0, 46.0, 31.0, 43.0, 46.0, 36.0, 36.0, 43.0, 28.0, 35.0, 46.0, 51.0, 41.0, 51.0, 55.0, 38.0, 50.0, 52.0, 44.0, 43.0, 49.0, 52.0, 54.0, 42.0, 44.0, 51.0, 33.0, 38.0, 33.0, 47.0, 35.0, 47.0, 46.0, 40.0, 36.0, 35.0, 44.0, 44.0, 31.0, 51.0, 44.0, 34.0, 52.0, 36.0, 51.0, 38.0, 39.0, 29.0, 38.0, 42.0, 47.0, 44.0, 39.0, 24.0, 51.0, 41.0, 36.0, 41.0, 46.0, 28.0, 24.0, 32.0, 36.0, 36.0, 46.0, 55.0, 55.0, 38.0, 36.0, 47.0, 37.0, 29.0, 45.0, 44.0, 45.0, 44.0, 47.0, 44.0, 39.0, 35.0, 42.0, 35.0, 39.0, 47.0, 45.0, 41.0, 35.0, 41.0, 34.0, 41.0, 39.0, 34.0, 43.0, 42.0, 36.0, 32.0])
income = np.array([38202.0, 44883.0, 42011.0, 35934.0, 35561.0, 42219.0, 35113.0, 42141.0, 41041.0, 38442.0, 37445.0, 40634.0, 32318.0, 37991.0, 43268.0, 45893.0, 39100.0, 48929.0, 47271.0, 36575.0, 44893.0, 49479.0, 37809.0, 41565.0, 43805.0, 43887.0, 49753.0, 41668.0, 38260.0, 47663.0, 35522.0, 37105.0, 34757.0, 41890.0, 40052.0, 42313.0, 40720.0, 37984.0, 40259.0, 33736.0, 45272.0, 42000.0, 31468.0, 51204.0, 40887.0, 38974.0, 46151.0, 35729.0, 48820.0, 42052.0, 35463.0, 32899.0, 42328.0, 44504.0, 45697.0, 42009.0, 41934.0, 31368.0, 47346.0, 39064.0, 35646.0, 41512.0, 46011.0, 30096.0, 27235.0, 32728.0, 39859.0, 40774.0, 46112.0, 49337.0, 51348.0, 36289.0, 40332.0, 47470.0, 36637.0, 31849.0, 40644.0, 44750.0, 47441.0, 40280.0, 44322.0, 43532.0, 39243.0, 34646.0, 39483.0, 38488.0, 38063.0, 43645.0, 40608.0, 37451.0, 33825.0, 38936.0, 36828.0, 40781.0, 40228.0, 34174.0, 38186.0, 41781.0, 31930.0, 34096.0])

# COMMAND ----------

# MAGIC %md **Question 4.1: (5 pts)** Plot the histogram of the income of the "young" population. Use the `group` variable to select from `income` the appropriate datapoints.

# COMMAND ----------

#plt.figure();
newIncome=income[np.where(group == 'young')]
plt.hist(newIncome)
display()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md **Question 4.2: ** (10 pts) Produce a scatter plot using `plt.scatter` to depic the relationship between `age` and `income`.  However, put the scatter plot of the 'young' group in green and the 'old' group in 'blue'. The plot must have a legend describing the groups, a title describing the relationship plotted, and labels for both axes.

# COMMAND ----------

col= np.where(group[:] =='young','g','b')
colors=['g','b']
a=plt.scatter(age,income,marker='o',color=colors[0])
b=plt.scatter(age,income,marker='o',color=colors[1])
plt.legend((a,b),('Young','Old'),scatterpoints=1)
plt.scatter(age,income,c=col)
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Relationship')
display()

# COMMAND ----------

# MAGIC %md # Part 5: Pandas

# COMMAND ----------

# MAGIC %md using a modified dataset from above

# COMMAND ----------

# import needed packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# create the data
group2 = np.array(['young', 'old', 'young', 'young', 'young', 'old', 'young', 'old', 'old', 'young', 'young', 'old', 'young', 'young', 'old', 'old', 'old', 'old', 'old', 'young', 'old', 'old', 'young', 'old', 'old', 'old', 'old', 'old', 'old', 'old', 'young', 'young', 'young', 'old', 'young', 'old', 'old', 'old', 'young', 'young', 'old', 'old', 'young', 'old', 'old', 'young', 'old', 'young', 'old', 'young', 'young', 'young', 'old', 'young', 'old', 'old', 'young', 'young', 'old', 'old', 'young', 'old', 'old', 'young', 'young', 'young', 'young', 'young', 'old', 'old', 'old', 'old', 'young', 'old', 'old', 'young', 'old', 'old', 'old', 'old', 'old', 'old', 'young', 'young', 'old', 'young', 'old', 'old', 'old', 'old', 'young', 'young', 'young', 'old', 'old', 'young', 'old', 'old', 'young', 'young'])
age2 = np.array([37.0, 41.0, 42.0, 41.0, 36.0, 46.0, 31.0, 43.0, 46.0, 36.0, 36.0, None, 28.0, 35.0, 46.0, 51.0, 41.0, 51.0, 55.0, 38.0, 50.0, 52.0, 44.0, 43.0, 49.0, 52.0, 54.0, 42.0, 44.0, 51.0, 33.0, 38.0, 33.0, 47.0, 35.0, 47.0, 46.0, 40.0, 36.0, 35.0, 44.0, 44.0, None, 51.0, 44.0, 34.0, 52.0, 36.0, 51.0, 38.0, 39.0, 29.0, 38.0, 42.0, 47.0, None, 39.0, 24.0, 51.0, 41.0, 36.0, 41.0, 46.0, 28.0, 24.0, 32.0, 36.0, 36.0, 46.0, 55.0, 55.0, 38.0, 36.0, 47.0, 37.0, 29.0, 45.0, 44.0, 45.0, 44.0, 47.0, 44.0, 39.0, 35.0, 42.0, 35.0, 39.0, 47.0, 45.0, 41.0, 35.0, 41.0, 34.0, 41.0, 39.0, 34.0, 43.0, 42.0, 36.0, 32.0], float)
income2 = np.array([38202.0, 44883.0, 42011.0, 35934.0, 35561.0, 42219.0, 35113.0, 42141.0, 41041.0, 38442.0, 37445.0, 40634.0, 32318.0, 37991.0, 43268.0, 45893.0, 39100.0, 48929.0, 47271.0, 36575.0, 44893.0, 49479.0, 37809.0, 41565.0, 43805.0, 43887.0, 49753.0, 41668.0, 38260.0, 47663.0, 35522.0, 37105.0, 34757.0, 41890.0, 40052.0, 42313.0, 40720.0, 37984.0, 40259.0, None, 45272.0, 42000.0, 31468.0, 51204.0, 40887.0, 38974.0, 46151.0, 35729.0, 48820.0, 42052.0, 35463.0, 32899.0, 42328.0, 44504.0, 45697.0, 42009.0, 41934.0, 31368.0, 47346.0, 39064.0, 35646.0, 41512.0, 46011.0, 30096.0, 27235.0, 32728.0, 39859.0, 40774.0, 46112.0, 49337.0, 51348.0, 36289.0, None, 47470.0, 36637.0, 31849.0, 40644.0, 44750.0, 47441.0, 40280.0, 44322.0, 43532.0, 39243.0, 34646.0, 39483.0, 38488.0, 38063.0, 43645.0, 40608.0, 37451.0, 33825.0, 38936.0, 36828.0, 40781.0, 40228.0, 34174.0, 38186.0, 41781.0, None, 34096.0], float)

# COMMAND ----------

# MAGIC %md **Question 5.1:** (5 pts): Create a dataframe `df` that contains three columns describing the group, age, and income, respectively, based on the variables `group2`, `age2`, and `income2` created in the cell above. Use the Pandas DataFrame functionality to compute the mean age and income of each group.

# COMMAND ----------

df = pd.DataFrame({"group":group2,'age':age2,"income":income2})
df

# COMMAND ----------

# estimte the means per group
df.groupby("group").mean()

# COMMAND ----------

# MAGIC %md **Question 5.2 : ** (10 pts) Standardize features. Standardized the age and income in the dataframe from Question 5.1. Standardization is the process of subtracting the mean of a feature and dividing the result by the standard deviation of the feature. Use the `apply` method of a dataframe to apply a function to the age and income series to achieve this result. Remember that `apply` will apply the function to all series so it will not work with the `group` series because it contains strings. Therefore, you must select the appropriate columns first.

# COMMAND ----------

df1= df.loc[:,['income','age']]
df1
def stand(x):
  return (x-x.mean())/x.std()
newdf=df1.apply(stand)
newdf

# COMMAND ----------

# MAGIC %md **Question 5.3:** (10 pts) Plot the standardized features using Pandas. Plot age vs income in their standardized format (Question 5.2) as a scatterplot using the functionality of Pandas. 

# COMMAND ----------

plt.figure();
newdf.plot.scatter(x='age',y='income')
display();
