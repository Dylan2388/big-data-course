from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.getOrCreate()

df = spark.read.format("libsvm").option("numFeatures", "780").load("data/mllib/sample_libsvm_data.txt")