from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.getOrCreate()

df = (spark.read
          .format("csv")
          .option('header', 'true')
          .load("train.csv"))