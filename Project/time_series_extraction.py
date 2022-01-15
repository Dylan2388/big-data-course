from ast import expr
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()

df_section_start = spark.read.csv("/user/s2733226/project/column_data/sections_start/part-0*.csv")
df_section_start = df_section_start.selectExpr("_c0 as name", "_c1 as value")
df_section_start = df_section_start.distinct()


from pyspark.sql.types import ArrayType, IntegerType

df_section_start = df_section_start.withColumn("value", df_section_start["value"].substr(1,len(df_section_start["value"])))
df_section_start_array = df_section_start.withColumn("value",F.split(F.col("value"), ",\s*").cast(ArrayType(IntegerType())).alias("value"))