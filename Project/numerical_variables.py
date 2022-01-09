# Import packages and initiate session
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

# 1. Danceability
# Read the text file
dfnew = spark.read.csv("/user/s2733226/project/column_data/danceability/part-0*.csv")
# Showing the data
dfnew.show()
# Rename the column names
dfnew = dfnew.selectExpr("_c0 as name", "_c1 as danceability")
# Convert the data types to integer
dfnew = dfnew.withColumn("danceability",dfnew["danceability"].cast(IntegerType()))
# Count the whole data
dfnew.count()
# 3000000
# Count the data that is zero
dfnew.select('name').where(dfnew.danceability==0).count()
# 3000000
# Count the data that is not zero
dfnew.select('name').where(dfnew.danceability!=0).count()

# 2. Duration
# Read the text file
df_duration = spark.read.csv("/user/s2733226/project/column_data/duration/part-0*.csv")
# Rename the column names
df_duration = df_duration.selectExpr("_c0 as name", "_c1 as value")
# Showing the data
df_duration.show()
# Convert the data types to integer
df_duration = df_duration.withColumn("value",df_duration["value"].cast(IntegerType()))
# Count the whole data
df_duration.count()
# Count the data that is zero
df_duration.select('name').where(df_duration.value==0).count()
# 255
# Count the data that is not zero
df_duration.select('name').where(df_duration.value!=0).count()
# 2999745

# 3. Energy
# Read the text file
df_energy = spark.read.csv("/user/s2733226/project/column_data/duration/part-0*.csv")
# Showing the data
df_energy.show()
# Rename the column names
df_energy = df_energy.selectExpr("_c0 as name", "_c1 as value")
# Convert the data types to integer
df_energy = df_energy.withColumn("value",df_energy["value"].cast(IntegerType()))
# Count the whole data
df_energy.count()
# 3000000
# Count the data that is zero
df_energy.select('name').where(df_energy.value==0).count()
# 255 
# Count the data that is not zero
df_energy.select('name').where(df_energy.value!=0).count()
# 2999745

# 4. Loudness
# Read the text file
df_loudness = spark.read.csv("/user/s2733226/project/column_data/loudness/part-0*.csv")
# Rename the column names
df_loudness = df_loudness.selectExpr("_c0 as name", "_c1 as value")
# Showing the data
df_loudness.show()
# Convert the data types to integer
df_loudness = df_loudness.withColumn("value",df_loudness["value"].cast(IntegerType()))
# Count the whole data
df_loudness.count()
# Count the data that is zero
df_loudness.select('name').where(df_loudness.value==0).count()
# 1824
# Count the data that is not zero
df_loudness.select('name').where(df_loudness.value!=0).count()
# 2998176

# 5. Song Hotness
# Read the text file
df_hotness = spark.read.csv("/user/s2733226/project/column_data/song_hotttnesss/part-0*.csv")
# Showing the data
df_hotness.show()
# Rename the column names
df_hotness = df_hotness.selectExpr("_c0 as name", "_c1 as value")
# Convert the data types to integer
df_hotness = df_hotness.withColumn("value",df_hotness["value"].cast(IntegerType()))
# Count the whole data
df_hotness.count()
# Count the data that is zero
df_hotness.select('name').where(df_hotness.value==0).count()
# 1745379
# Count the data that is not zero
df_hotness.select('name').where(df_hotness.value!=0).count()
# 516
# 1255137
# there are nan and 0 data in hotness

# 6. Tempo
# Read the text file
df_tempo = spark.read.csv("/user/s2733226/project/column_data/tempo/part-0*.csv")
# Rename the column names
df_tempo = df_tempo.selectExpr("_c0 as name", "_c1 as value")
# Showing the data
df_tempo.show()
# Convert the data types to integer
df_tempo = df_tempo.withColumn("value",df_tempo["value"].cast(IntegerType()))
# Count the whole data
# 3000000
df_tempo.distinct().count()
# Count the data that is zero
df_tempo.select('name').where(df_tempo.value==0).distinct().count()
# 9690
# Count the data that is not zero
df_tempo.select('name').where(df_tempo.value!=0).count()
# 2990310

# Tempo =  beat/minutes
# There are triple duplicates but it should be 1000000 datasets