# Import packages and initiate session
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

# 1. Time Signature
# Read the text file
df_timesig = spark.read.csv("/user/s2733226/project/column_data/time_signature/part-0*.csv")
# Showing the data
df_timesig.show()
# Rename the column names
df_timesig = df_timesig.selectExpr("_c0 as name", "_c1 as value")
# Convert the data types to integer
df_timesig = df_timesig.withColumn("value",df_timesig["value"].cast(IntegerType()))
# Count the whole data
df_timesig.distinct().count()
# Showing the distinct categorical data
df_timesig.select('value').distinct().show()
# Count the data that is zero
df_timesig.select('name').where(df_timesig.value==0).distinct().count()
# 619
# Time signature range : 0-7 (ignore 0)

# 2. Key 
# Read the text file
df_key = spark.read.csv("/user/s2733226/project/column_data/key/part-0*.csv")
# Rename the column names
df_key = df_key.selectExpr("_c0 as name", "_c1 as value")
# Showing the data
df_key.show()
# Convert the data types to integer
df_key = df_key.withColumn("value",df_key["value"].cast(IntegerType()))
# Count the whole data
df_key.distinct().count()
# Count the data that is zero
df_key.select('name').where(df_key.value==0).distinct().count()
# 122237
# Showing the distinct categorical data
df_key.select('value').distinct().show()
# Key range : 0-12

# 3. Mode
# Read the text file
df_mode = spark.read.csv("/user/s2733226/project/column_data/mode/part-0*.csv")
# Rename the column names
df_mode = df_mode.selectExpr("_c0 as name", "_c1 as value")
# Showing the data
df_mode.show()
# Convert the data types to integer
df_mode = df_mode.withColumn("value",df_mode["value"].cast(IntegerType()))
# Count the whole data
df_mode.distinct().count()
# Count the data that is zero
df_mode.select('name').where(df_mode.value==0).distinct().count()
# 122237
# Showing the distinct categorical data
df_mode.select('value').distinct().show()
# mode range : 0-12
