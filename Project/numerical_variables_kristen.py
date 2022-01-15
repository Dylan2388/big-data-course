# Run command:
# time spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=10 --executor-memory 2G numerical_variables_kristen.py > logfile_numerical_variables_kristen.txt 2>&1 /dev/null

# Import packages and initiate session
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.sql import functions as f
from pyspark.sql.functions import col
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import MinMaxScaler



spark = SparkSession.builder.getOrCreate()


# Feature: Duration
# Create schema
schema = StructType(\
    [StructField("track_id", StringType(), True),\
     StructField("duration", FloatType(), True)])
# Read the text file
df_duration = spark.read.csv("/user/s2733226/project/column_data/duration/part*.csv", header = 'False', schema = schema)
# Remove duplicates: w/ duplicates df_duration.count() = 3.000.000; w/o duplicates df_duration.count() = 1.000.000
df_duration = df_duration.distinct()
# Remove null's
df_duration = df_duration.filter(col("track_id").isNotNull() & col("duration").isNotNull())
# Remove zero's
df_duration = df_duration.filter(col('duration') != 0.0)
# Remove outliners using StandardScaler
assembler = VectorAssembler().setInputCols(["duration"]).setOutputCol("duration_vector") # convert duration from float to vector
scaler = StandardScaler(inputCol="duration_vector", outputCol="duration_std_vector", withStd=True, withMean=True)
scaler_model = scaler.fit(assembler.transform(df_duration))
df_duration = scaler_model.transform(assembler.transform(df_duration)) # calculate duration_std_vector
convert_float = f.udf(lambda v: float(v[0]), FloatType()) # transform duration_std_vector to duration_std float
df_duration = df_duration.withColumn("duration_std", convert_float("duration_std_vector"))
# df_duration.sort(col('duration_std'), ascending=True).show()
lower_std = -1.5 # min std = -1.9740827; max std = 22.066174|
upper_std = 1.5
df_duration = df_duration.filter((col("duration_std") > lower_std) & (col("duration_std") < upper_std)) # keep only songs that fall between lower_std and upper_std
# Normalize duration using Normalizer
# normalizer = Normalizer(inputCol="duration_vector", outputCol="duration_normalized_vector", p=2.0)
# df_duration = normalizer.transform(df_duration)
# df_duration = df_duration.withColumn("duration_normalized", convert_float("duration_normalized_vector"))
# df_duration.sort(col('duration_normalized'), ascending=False).show()
# Apply MinMaxScaler
scaler = MinMaxScaler(inputCol="duration_vector", outputCol="scaled_duration_vector")
scalerModel = scaler.fit(df_duration)
df_duration = scalerModel.transform(df_duration)
#### REMEMBER TO SELECT THE "scaled_duration_vector" ONLY




# Attribute: Loudness
# Create schema
schema = StructType(\
    [StructField("track_id", StringType(), True),\
     StructField("loudness", FloatType(), True)])
# Read the text file
df_loudness = spark.read.csv("/user/s2733226/project/column_data/loudness/part*.csv", header = 'False', schema = schema)
# Remove duplicates
df_loudness = df_loudness.distinct()
# Remove null's
df_loudness = df_loudness.filter(col("track_id").isNotNull() & col("loudness").isNotNull())
# Remove zero's
df_loudness = df_loudness.filter(col('loudness') != 0.0)
# Remove outliners using StandardScaler
assembler = VectorAssembler().setInputCols(['loudness']).setOutputCol('loudness_vector') # convert duration from float to vector
scaler = StandardScaler(inputCol="loudness_vector", outputCol="loudness_std_vector", withStd=True, withMean=True)
scaler_model = scaler.fit(assembler.transform(df_loudness))
df_loudness = scaler_model.transform(assembler.transform(df_loudness)) # calculate duration_std_vector
convert_float = f.udf(lambda v: float(v[0]), FloatType()) # transform duration_std_vector to duration_std float
df_loudness = df_loudness.withColumn("loudness_std", convert_float("loudness_std_vector"))
lower_std = -1.5 # min std = -9.246046; max std = 2.7787876
upper_std = 1.5
df_loudness = df_loudness.filter((col("loudness_std") > lower_std) & (col("loudness_std") < upper_std)) # keep only songs that fall between lower_std and upper_std
# Apply MinMaxScaler
scaler = MinMaxScaler(inputCol="loudness_vector", outputCol="scaled_loudness_vector")
scalerModel = scaler.fit(df_loudness)
df_loudness = scalerModel.transform(df_loudness)
#### REMEMBER TO SELECT THE "scaled_loudness_vector" ONLY


# Attribute: Tempo (beat/minute)
# Create schema
schema = StructType(\
    [StructField("track_id", StringType(), True),\
     StructField("tempo", FloatType(), True)])
# Read the text file
df_tempo = spark.read.csv("/user/s2733226/project/column_data/tempo/part*.csv", header = 'False', schema = schema)
# Remove duplicates
df_tempo = df_tempo.distinct()
# Remove null's
df_tempo = df_tempo.filter(col("track_id").isNotNull() & col("tempo").isNotNull())
# Remove zero's
df_tempo = df_tempo.filter(col('tempo') != 0.0)
# Remove outliners using StandardScaler
assembler = VectorAssembler().setInputCols(['tempo']).setOutputCol('tempo_vector') # convert duration from float to vector
scaler = StandardScaler(inputCol="tempo_vector", outputCol="tempo_std_vector", withStd=True, withMean=True)
scaler_model = scaler.fit(assembler.transform(df_tempo))
df_tempo = scaler_model.transform(assembler.transform(df_tempo)) # calculate duration_std_vector
convert_float = f.udf(lambda v: float(v[0]), FloatType()) # transform duration_std_vector to duration_std float
df_tempo = df_tempo.withColumn("tempo_std", convert_float("tempo_std_vector"))
# df_tempo.sort(col('tempo_std'), ascending=False).show()
lower_std = -1.5 # min std = -3.5340395; max std = 5.08931
upper_std = 1.5
df_tempo = df_tempo.filter((col("tempo_std") > lower_std) & (col("tempo_std") < upper_std)) # keep only songs that fall between lower_std and upper_std
# Apply MinMaxScaler
scaler = MinMaxScaler(inputCol="tempo_vector", outputCol="scaled_tempo_vector")
scalerModel = scaler.fit(df_tempo)
df_tempo = scalerModel.transform(df_tempo)
#### REMEMBER TO SELECT THE "scaled_tempo_vector" ONLY



################### DONT USE EVERYTHING FROM HERE ########################


# Attribute: Song Hotness
# Create schema
schema = StructType(\
    [StructField("track_id", StringType(), False),\
     StructField("song_hotttnesss", FloatType(), False)])
# Read the text file
df_song_hotttnesss = spark.read.csv("/user/s2733226/project/column_data/song_hotttnesss/part*.csv", header = 'False', schema = schema)
# Remove duplicates
df_song_hotttnesss = df_song_hotttnesss.distinct()
# Remove null's
df_song_hotttnesss = df_song_hotttnesss.filter(col("track_id").isNotNull() & col("song_hotttnesss").isNotNull())
# Remove zero's
df_song_hotttnesss = df_song_hotttnesss.filter(col('song_hotttnesss') != 0.0)
# df_song_hotttnesss.sort(col('song_hotttnesss'), ascending=False).show()
#### REMEMBER TO SELECT THE "song_hotttnesss" ONLY


# Attribute: Energy; Dancability - disregard because all energy/ danceability data == 0.0
# Create schema
schema = StructType(\
    [StructField("track_id", StringType(), True),\
     StructField("energy", FloatType(), True)])
# Read the text file
df_energy = spark.read.csv("/user/s2733226/project/column_data/energy/part*.csv", header = 'False', schema = schema)
# Remove duplicates
df_energy = df_energy.distinct()
# Confirm that all energy data == 0.0
df_energy.groupBy().agg(f.sum("energy")).collect()


