# PySpark Example - Credit Card Data Analysis

This script demonstrates the use of PySpark for data processing and machine learning. It uses the UCI Credit Card dataset to build a machine learning pipeline, which includes data reading, preprocessing, and training a logistic regression model.

## Features

- **Data Reading**: Reads a CSV file into a Spark DataFrame.
- **Data Preprocessing**: Casts columns to integers, removes missing values, and creates feature vectors.
- **Machine Learning Pipeline**: Uses `VectorAssembler`, `StringIndexer`, and `OneHotEncoder` for data preparation.
- **Model Training**: Trains a logistic regression model.
- **Model Evaluation**: Splits the data into training and test sets, makes predictions, and evaluates the model.

## Requirements

Ensure you have PySpark installed. You can install it using:
```bash
pip install pyspark
```

## How to Use

1. **Set Up Environment**:
   - Change the working directory to the dataset's location:
     ```python
     import os
     os.chdir("C:\\Users\\mwill\\OneDrive\\Documents\\Datasets")
     ```

2. **Create Spark Session**:
   ```python
   from pyspark.sql import SparkSession
   spark = SparkSession.builder.master("local[*]").getOrCreate()
   ```

3. **Read Data**:
   ```python
   file_path = "UCI_Credit_Card.csv"
   data = spark.read.csv(file_path, header=True)
   data.show()
   ```

4. **Preprocess Data**:
   - Add a label column and cast columns to integers:
     ```python
     from pyspark.sql.functions import lit
     from pyspark.ml.feature import VectorAssembler

     data = data.withColumn("label", lit(0))
     data = data.withColumn("LIMIT_BAL", data.LIMIT_BAL.cast("integer"))
     data = data.withColumn("SEX", data.SEX.cast("integer"))
     data = data.withColumn("EDUCATION", data.EDUCATION.cast("integer"))
     data = data.withColumn("MARRIAGE", data.MARRIAGE.cast("integer"))
     data = data.withColumn("label", data.label.cast("integer"))
     ```

   - Remove missing values:
     ```python
     data = data.filter("LIMIT_BAL is not NULL and SEX is not NULL and EDUCATION is not NULL and MARRIAGE is not NULL")
     ```

5. **Create Pipeline**:
   ```python
   vec_assembler = VectorAssembler(inputCols=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE"], outputCol="features")
   from pyspark.ml import Pipeline
   pipeline = Pipeline(stages=[vec_assembler])
   ```

6. **Data Encoding**:
   - Use `StringIndexer` and `OneHotEncoder` for encoding categorical features:
     ```python
     from pyspark.ml.feature import StringIndexer, OneHotEncoder

     string_indexer = StringIndexer(inputCol="inputCol", outputCol="outputCol")
     one_encoder = OneHotEncoder(inputCol="inputCol", outputCol="outputCol")
     ```

7. **Fit and Transform Data**:
   ```python
   piped_data = pipeline.fit(data).transform(data)
   ```

8. **Split Data**:
   ```python
   training, test = piped_data.randomSplit([0.8, 0.2])
   ```

9. **Train and Evaluate Model**:
   ```python
   from pyspark.ml.classification import LogisticRegression

   lr = LogisticRegression()
   best_lr = lr.fit(training)
   predictions = best_lr.transform(test)
   predictions.show()
   predictions.select('label', 'features', 'rawPrediction', 'probability', 'prediction').show()
   ```

## Recent Changes

The recent changes for [Medium_pyspark_example1.py](https://github.com/MoneKEE/Python-Code-Samples/blob/master/Medium_pyspark_example1.py) are:
- [4a81b16](https://github.com/MoneKEE/Python-Code-Samples/commit/4a81b167d8f700aaf35595053b846be0b052c730): "Add files via upload"

In summary, this script was added to the repository to demonstrate the use of PySpark for data analysis and machine learning on the UCI Credit Card dataset.
