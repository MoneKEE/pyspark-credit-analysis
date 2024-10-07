#!pip install pyspark

#set current working directory to the dataset's location
import os
os.chdir("C:\\Users\\mwill\\OneDrive\\Documents\\Datasets")

#Create the spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()

#Read in the data
file_path = "UCI_Credit_Card.csv"
data = spark.read.csv(file_path,header=True)

#Show the data
data.show()

#Machine Learning Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

data = data.withColumn("label", lit(0))

#Cast the columns to integers
data = data.withColumn("LIMIT_BAL", data.LIMIT_BAL.cast("integer"))
data = data.withColumn("SEX", data.SEX.cast("integer"))
data = data.withColumn("EDUCATION", data.EDUCATION.cast("integer"))
data = data.withColumn("MARRIAGE", data.MARRIAGE.cast("integer"))
data = data.withColumn("label", data.label.cast("integer"))

#Remove missing values
data = data.filter("LIMIT_BAL is not NULL and SEX is not NULL and EDUCATION is not NULL and MARRIAGE is not NULL")

#Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["LIMIT_BAL","SEX","EDUCATION","MARRIAGE"],outputCol="features")

#Pipeline is a class that combines all the Estimators and Transformers allowing for reuse of the same
#modelling process by wrapping it in an object.
pipeline = Pipeline(stages=[vec_assembler])

'''
Data Encoding

The first step to encoding the categorical feature is the create a StringIndexer.  Members of this class are Estimators
that take a Dataframe with a column of strings and map each unique string to a number.  Then the estimator returns a 
Transformer that takes a DataFrame, attaches the mapping to it as metadata, and returns a new DataFrame with a numeric
column corresponding to the string column.

The second step is to encode this numeric column as a one-hot vector using a OneHotEncoder.  This works exactly the same way 
as the StringIndexer by creating an Estimator and then a Transformer.  The end result is a column that oncodes your categorical
feature as a vector that's suitable for machine learning routines.
'''

from pyspark.ml.feature import StringIndexer, OneHotEncoder

#Create a StringIndexer
string_indexer = StringIndexer(inputCol="inputCol", outputCol="outputCol")

#Create OneHotEncoder
one_encoder = OneHotEncoder(inputCol="inputCol", outputCol="outputCol")

#Fit and transform the data
piped_data = pipeline.fit(data).transform(data)

#Split the data into training and test sets
training, test = piped_data.randomSplit([0.8,0.2])

'''
Now we fit the model and then test it on the test set.
'''

from pyspark.ml.classification import LogisticRegression
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune

#Create a LogisticRegression Estimator
lr = LogisticRegression()

#Fit model
best_lr = lr.fit(training)

#Make predictions
predictions = best_lr.transform(test)

predictions.show()

predictions.select('label','features','rawPrediction','probability','prediction').show()