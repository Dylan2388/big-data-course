#Command: time spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=50 --executor-memory 6G  tags_columns_preprocess.py > logfile_tags_columns_preprocess.txt 2>&1 /dev/null
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

#Read exported MSD + lastfm joined file
df = spark.read.json("/user/s2733226/project/msd_lastfm_tags_1/*.json")

from pyspark.sql.functions import col, concat_ws, when, explode

#Converting tags column from an array to a string 
df2 = df.withColumn("tags",concat_ws(",",col("tags")))

#Selecting required columns and creating columns for each genre (rock, alternative, pop, indie, electronic, jazz, dance, instrumental, soul, metal)
df3 = df2.select(col('track_id'),col('artist_mbid'),col('artist_name'),col('artist_playmeid'),col('artist_terms'),col('artist_terms_freq'),col('artist_terms_weight'),col('audio_md5'),col('bars_confidence'),col('bars_start'),col('beats_confidence'),col('beats_start'),col('danceability'),col('duration'),col('end_of_fade_in'),col('energy'),col('key'),col('key_confidence'),col('loudness'),col('mode'),col('mode_confidence'),col('release'),col('release_7digitalid'),col('sections_confidence'),col('sections_start'),col('segments_confidence'),col('segments_loudness_max'),col('segments_loudness_max_time'),col('segments_loudness_start'),col('segments_pitches'),col('segments_start'),col('segments_timbre'),col('similar_artists'),col('song_hotttnesss'),col('song_id'),col('start_of_fade_out'),col('tatums_confidence'),col('tatums_start'),col('tempo'),col('time_signature'),col('time_signature_confidence'),col('title'),col('track_7digitalid'),col('year'),col('similars'),col('tags'), \
	when (col('tags').contains('rock'),1).otherwise(0).alias('rock'), \
	when (col('tags').contains('alternative'),1).otherwise(0).alias('alternative'), \
	when (col('tags').contains('pop'),1).otherwise(0).alias('pop'), \
	when (col('tags').contains('indie'),1).otherwise(0).alias('indie'), \
	when (col('tags').contains('electronic'),1).otherwise(0).alias('electronic'), \
	when (col('tags').contains('jazz'),1).otherwise(0).alias('jazz'), \
	when (col('tags').contains('dance'),1).otherwise(0).alias('dance'), \
	when (col('tags').contains('instrumental'),1).otherwise(0).alias('instrumental'), \
	when (col('tags').contains('soul'),1).otherwise(0).alias('soul'), \
	when (col('tags').contains('metal'),1).otherwise(0).alias('metal'))


#####################Numerical data pre-processing
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.sql import functions as f
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import MinMaxScaler

#Removing bad rows for Duration, Loudness, Tempo
#Removing nulls
df3 = df3.filter(col("track_id").isNotNull() & col("duration").isNotNull() & col("tempo").isNotNull() & col("loudness").isNotNull())

#Removing zeros
df3 = df3.filter((col('duration') != 0.0) & (col('loudness') != 0.0) & (col('tempo') != 0.0))


#Remove outliers using StandardScaler
#Convert to float
df3 = df3.withColumn('duration',df3['duration'].cast("float").alias('duration'))
df3 = df3.withColumn('loudness',df3['loudness'].cast("float").alias('loudness'))
df3 = df3.withColumn('tempo',df3['tempo'].cast("float").alias('tempo'))

#Duration
assembler = VectorAssembler().setInputCols(["duration"]).setOutputCol("duration_vector") # convert duration from float to vector
scaler = StandardScaler(inputCol="duration_vector", outputCol="duration_std_vector", withStd=True, withMean=True)
scaler_model = scaler.fit(assembler.transform(df3))
df3 = scaler_model.transform(assembler.transform(df3)) # calculate duration_std_vector
convert_float = f.udf(lambda v: float(v[0]), FloatType()) # transform duration_std_vector to duration_std float
df3 = df3.withColumn("duration_std", convert_float("duration_std_vector"))
lower_std = -1.5 # min std = -1.9740827; max std = 22.066174|
upper_std = 1.5
df3 = df3.filter((col("duration_std") > lower_std) & (col("duration_std") < upper_std)) # keep only songs that fall between lower_std and upper_std
scaler = MinMaxScaler(inputCol="duration_vector", outputCol="scaled_duration_vector")
scalerModel = scaler.fit(df3)
df3 = scalerModel.transform(df3)

#Loudness
assembler = VectorAssembler().setInputCols(['loudness']).setOutputCol('loudness_vector') # convert loudness from float to vector
scaler = StandardScaler(inputCol="loudness_vector", outputCol="loudness_std_vector", withStd=True, withMean=True)
scaler_model = scaler.fit(assembler.transform(df3))
df3 = scaler_model.transform(assembler.transform(df3)) # calculate loudness_std_vector
df3 = df3.withColumn("loudness_std", convert_float("loudness_std_vector"))
#lower_std = -1.5 # min std = -9.246046; max std = 2.7787876
#upper_std = 1.5
df3 = df3.filter((col("loudness_std") > lower_std) & (col("loudness_std") < upper_std)) # keep only songs that fall between lower_std and upper_std
# Apply MinMaxScaler
scaler = MinMaxScaler(inputCol="loudness_vector", outputCol="scaled_loudness_vector")
scalerModel = scaler.fit(df3)
df3 = scalerModel.transform(df3)

#Tempo
assembler = VectorAssembler().setInputCols(['tempo']).setOutputCol('tempo_vector') # convert loudness from float to vector
scaler = StandardScaler(inputCol="tempo_vector", outputCol="tempo_std_vector", withStd=True, withMean=True)
scaler_model = scaler.fit(assembler.transform(df3))
df3 = scaler_model.transform(assembler.transform(df3)) # calculate loudness_std_vector
#convert_float = f.udf(lambda v: float(v[0]), FloatType()) # transform loudness_std_vector to loudness_std float
df3 = df3.withColumn("tempo_std", convert_float("tempo_std_vector"))
#lower_std = -1.5 # min std = -3.5340395; max std = 5.08931
#upper_std = 1.5
df3 = df3.filter((col("tempo_std") > lower_std) & (col("tempo_std") < upper_std)) # keep only songs that fall between lower_std and upper_std
# Apply MinMaxScaler
scaler = MinMaxScaler(inputCol="tempo_vector", outputCol="scaled_tempo_vector")
scalerModel = scaler.fit(df3)
df3 = scalerModel.transform(df3)

#Drop columns
cols = ("duration_vector","duration_std_vector","duration_std","loudness_vector","loudness_std_vector","loudness_std","tempo_vector","tempo_std_vector","tempo_std")
df3 = df3.drop(*cols)

'''
#Categorical Data Pre-Processing
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator

# 1. Time Signature
stringIndexer = StringIndexer(inputCol="time_signature", outputCol="categorical_time_signature").fit(df3)
df3 = stringIndexer.transform(df3)
# One hot encoder
encoder = OneHotEncoderEstimator(inputCols=["categorical_time_signature"], outputCols=["categorical_time_signature_vector"])
model = encoder.fit(df3)
df3 = model.transform(df3)

# 2. key
# First step of transformation
stringIndexer = StringIndexer(inputCol="key", outputCol="categorical_key").fit(df3)
df3 = stringIndexer.transform(df3)
# One hot encoder
encoder = OneHotEncoderEstimator(inputCols=["categorical_key"], outputCols=["categorical_key_vector"])
model = encoder.fit(df3)
df3 = model.transform(df3)

# 3. Mode
# First step of transformation
stringIndexer = StringIndexer(inputCol="mode", outputCol="categorical_mode").fit(df3)
df3 = stringIndexer.transform(df3)
# One hot encoder
encoder = OneHotEncoderEstimator(inputCols=["categorical_mode"], outputCols=["categorical_mode_vector"])
model = encoder.fit(df3)
df3 = model.transform(df3)

#Drop columns
cols = ("categorical_time_signature","categorical_key","categorical_mode")
df3 = df3.drop(*cols)
'''
#Taking distinct values
df3 = df3.distinct()
	
#Write as json
df3.write.format('json').save('/user/s2733226/project/msd_lastfm_tags_columns_preprocessed')
