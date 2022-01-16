from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from pyspark.sql.functions import col, explode

df = spark.read.json('/user/s2733226/project/segments_tatums_preprocessed/*.json')

from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
import pyspark.sql.functions as f	
	
#CATEGORICAL DATA
#categorical_key_vector
#convert struct to vectors
#df1 = df.select("track_id","categorical_key_vector.*")
df1 = df.select('alternative', 'artist_mbid', 'artist_name', 'categorical_key_vector.*', 'categorical_mode_vector', 'categorical_time_signature_vector', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year')
#split struct column into individual columns for each element
df1 = df1.withColumn("indices",concat_ws(",",col("indices")))
df1 = df1.withColumn("values",concat_ws(",",col("values")))
#Convert to float
df1 = df1.withColumn('indices',df1['indices'].cast("float").alias('indices'))
df1 = df1.withColumn('size',df1['size'].cast("float").alias('size'))
df1 = df1.withColumn('type',df1['type'].cast("float").alias('type'))
df1 = df1.withColumn('values',df1['values'].cast("float").alias('values'))
#combine all elements into an array
columns = [f.col("indices"),f.col("size"),f.col("type"),f.col("values")]
df2 = df1.withColumn("categorical_key_vector", f.array(columns)).drop("indices","size","type","values")
#convert to vector
to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
#df_columns = ('alternative', 'artist_mbid', 'artist_name', 'categorical_key_vector', 'categorical_mode_vector', 'categorical_time_signature_vector', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year')
df = df2.select('alternative', 'artist_mbid', 'artist_name', 'categorical_mode_vector', 'categorical_time_signature_vector', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year', to_vector("categorical_key_vector").alias('categorical_key_vector'))

#categorical_mode_vector
#convert struct to vectors
df1 = df.select('alternative', 'artist_mbid', 'artist_name', 'categorical_key_vector', 'categorical_mode_vector.*', 'categorical_time_signature_vector', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year')
#split struct column into individual columns for each element
df1 = df1.withColumn("indices",concat_ws(",",col("indices")))
df1 = df1.withColumn("values",concat_ws(",",col("values")))
#Convert to float
df1 = df1.withColumn('indices',df1['indices'].cast("float").alias('indices'))
df1 = df1.withColumn('size',df1['size'].cast("float").alias('size'))
df1 = df1.withColumn('type',df1['type'].cast("float").alias('type'))
df1 = df1.withColumn('values',df1['values'].cast("float").alias('values'))
#combine all elements into an array
columns = [f.col("indices"),f.col("size"),f.col("type"),f.col("values")]
df2 = df1.withColumn("categorical_mode_vector", f.array(columns)).drop("indices","size","type","values")
#convert to vector
#to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
df = df2.select('alternative', 'artist_mbid', 'artist_name', 'categorical_key_vector', 'categorical_time_signature_vector', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year', to_vector("categorical_mode_vector").alias('categorical_mode_vector'))

#categorical_time_signature_vector
#convert struct to vectors
df1 = df.select('alternative', 'artist_mbid', 'artist_name', 'categorical_key_vector', 'categorical_mode_vector', 'categorical_time_signature_vector.*', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year')
#split struct column into individual columns for each element
df1 = df1.withColumn("indices",concat_ws(",",col("indices")))
df1 = df1.withColumn("values",concat_ws(",",col("values")))
#Convert to float
df1 = df1.withColumn('indices',df1['indices'].cast("float").alias('indices'))
df1 = df1.withColumn('size',df1['size'].cast("float").alias('size'))
df1 = df1.withColumn('type',df1['type'].cast("float").alias('type'))
df1 = df1.withColumn('values',df1['values'].cast("float").alias('values'))
#combine all elements into an array
columns = [f.col("indices"),f.col("size"),f.col("type"),f.col("values")]
df2 = df1.withColumn("categorical_time_signature_vector", f.array(columns)).drop("indices","size","type","values")
#convert to vector
df = df2.select('alternative', 'artist_mbid', 'artist_name', 'categorical_key_vector', 'categorical_mode_vector', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year', to_vector("categorical_time_signature_vector").alias('categorical_time_signature_vector'))

#####NUMERICAL DATA
#scaled_duration_vector
#convert struct to vectors
df1 = df.select('alternative', 'artist_mbid', 'artist_name', 'categorical_key_vector', 'categorical_mode_vector', 'categorical_time_signature_vector', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector.*', 'scaled_loudness_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year')
#split struct column into individual columns for each element
#df1 = df1.withColumn("indices",concat_ws(",",col("indices")))
df1 = df1.withColumn("values",concat_ws(",",col("values")))
#Convert to float
#df1 = df1.withColumn('indices',df1['indices'].cast("float").alias('indices'))
#df1 = df1.withColumn('size',df1['size'].cast("float").alias('size'))
df1 = df1.withColumn('type',df1['type'].cast("float").alias('type'))
df1 = df1.withColumn('values',df1['values'].cast("float").alias('values'))
#combine all elements into an array
columns = [f.col("type"),f.col("values")]
df2 = df1.withColumn("scaled_duration_vector", f.array(columns)).drop("type","values")
#convert to vector
df = df2.select('alternative', 'artist_mbid', 'artist_name', 'categorical_key_vector', 'categorical_mode_vector', 'categorical_time_signature_vector', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_loudness_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year', to_vector("scaled_duration_vector").alias('scaled_duration_vector'))

#scaled_loudness_vector
#convert struct to vectors
df1 = df.select('alternative', 'artist_mbid', 'artist_name', 'categorical_key_vector', 'categorical_mode_vector', 'categorical_time_signature_vector', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector.*', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year')
#split struct column into individual columns for each element
#df1 = df1.withColumn("indices",concat_ws(",",col("indices")))
df1 = df1.withColumn("values",concat_ws(",",col("values")))
#Convert to float
#df1 = df1.withColumn('indices',df1['indices'].cast("float").alias('indices'))
#df1 = df1.withColumn('size',df1['size'].cast("float").alias('size'))
df1 = df1.withColumn('type',df1['type'].cast("float").alias('type'))
df1 = df1.withColumn('values',df1['values'].cast("float").alias('values'))
#combine all elements into an array
columns = [f.col("type"),f.col("values")]
df2 = df1.withColumn("scaled_loudness_vector", f.array(columns)).drop("type","values")
#convert to vector
df = df2.select('alternative', 'artist_mbid', 'artist_name', 'categorical_key_vector', 'categorical_mode_vector', 'categorical_time_signature_vector', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year', to_vector("scaled_loudness_vector").alias('scaled_loudness_vector'))

#scaled_tempo_vector
#convert struct to vectors
df1 = df.select('alternative', 'artist_mbid', 'artist_name', 'categorical_key_vector', 'categorical_mode_vector', 'categorical_time_signature_vector', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector', 'scaled_tempo_vector.*', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year')
#split struct column into individual columns for each element
#df1 = df1.withColumn("indices",concat_ws(",",col("indices")))
df1 = df1.withColumn("values",concat_ws(",",col("values")))
#Convert to float
#df1 = df1.withColumn('indices',df1['indices'].cast("float").alias('indices'))
#df1 = df1.withColumn('size',df1['size'].cast("float").alias('size'))
df1 = df1.withColumn('type',df1['type'].cast("float").alias('type'))
df1 = df1.withColumn('values',df1['values'].cast("float").alias('values'))
#combine all elements into an array
columns = [f.col("type"),f.col("values")]
df2 = df1.withColumn("scaled_tempo_vector", f.array(columns)).drop("type","values")
#convert to vector
df = df2.select('alternative', 'artist_mbid', 'artist_name', 'categorical_key_vector', 'categorical_mode_vector', 'categorical_time_signature_vector', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year', to_vector("scaled_tempo_vector").alias('scaled_tempo_vector'))

#df.write.format('json').save('/user/s2733226/project/preprocessed_and_vectorized')