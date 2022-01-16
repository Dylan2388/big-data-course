import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder.getOrCreate()

schema = StructType(\
    [StructField("track_id", StringType(), True),\
     StructField("duration", FloatType(), True)])

data = spark.read.json("/user/s2733226/project/msd_lastfm_tags_columns_preprocessed/part-00000*.json")
data.printSchema()
selected_column = ["song_id", 
                   "scaled_duration_vector", "scaled_loudness_vector", "scaled_tempo_vector",
                   "categorical_key_vector", "categorical_mode_vector", "categorical_time_signature_vector",
                   "dance", "electronic", "indie", "instrumental", "jazz", "loudness", "metal", "pop", "rock", "soul"]
data = data.select(selected_column)


## Convert Struct to Vector
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
data = data.withColumn("value", F.expr("substring(value, 2, length(value) - 2)"))

parse = udf(lambda s: Vectors.parse(s), VectorUDT())
data_test = data.select(parse("scaled_duration_vector"))

data_test.show(5)

data.select("categorical_time_signature_vector").show(5, truncate=False)


## Preparedata
vectorized_column = ["scaled_duration_vector", "scaled_loudness_vector", "scaled_tempo_vector",
                   "categorical_key_vector", "categorical_mode_vector", "categorical_time_signature_vector"]
assembler = VectorAssembler().setInputCols(vectorized_column).setOutputCol('features') 
selected_column = ["song_id",
                   "features",
                   "dance", "electronic", "indie", "instrumental", "jazz", "loudness", "metal", "pop", "rock", "soul"]
data = assembler.transform(data).select[selected_column]
train, validate, test = data.randomSplit([0.7, 0.2, 0.1], seed=42)



###### LOGISTIC REGRESSION #######
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

