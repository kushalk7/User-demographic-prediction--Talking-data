import sys
import numpy as np

from pyspark import SparkContext, SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import VectorAssembler

sc = SparkContext()
sqlContext = SQLContext(sc)

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('/media/sf_TalkingData/TalkingData/events-gender-brand.csv')
df = df.drop('timestamp')
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in list(set(df.columns)) ]

pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(df).transform(df)
df_r = df_r.drop("event_id").drop("device_id").drop("longitude").drop("latitude").drop("gender").drop("phone_brand").drop("device_model")
df_r.show()

assembler = VectorAssembler(
inputCols=["event_id_index", "age_index","longitude_index","device_model_index","latitude_index","phone_brand_index"],
outputCol="features")
res = assembler.transform(df_r)
res = res.drop("event_id_index").drop("age_index").drop("longitude_index").drop("device_model_index").drop("device_id_index").drop("phone_brand_index")
res.show()

scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")

# Compute summary statistics and generate MaxAbsScalerModel
scalerModel = scaler.fit(res)

# rescale each feature to range [-1, 1].
scaledData = scalerModel.transform(res)
scaledData.show()
scaledData.select("scaledFeatures").show()
res = scaledData.limit(100000)

(trainingData, testingData) = res.randomSplit([0.7,0.3])
rf = RandomForestClassifier(labelCol="group_index", featuresCol="scaledFeatures")
rfModel = rf.fit(trainingData)
predictions = rfModel.transform(testingData)
predictions.show()

evaluator = MulticlassClassificationEvaluator(
    labelCol="group_index", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("accuracy = %g" % (accuracy*100))