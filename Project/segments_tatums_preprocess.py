#Command: time spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=50 --executor-memory 6G  segments_tatums_preprocess.py > logfile_segments_tatums_preprocess.txt 2>&1 /dev/null
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

#Read preprocessed file
df3 = spark.read.json("/user/s2733226/project/msd_lastfm_tags_columns_preprocessed/*.json")

import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType
df4 = df3.select(col('track_id'),col('segments_start'))
#exploding
#remove the array brackets at the start and end of the string
df5 = df4.withColumn("segments_start", F.expr("substring(segments_start, 2, length(segments_start) - 2)"))
#convert string to float
df6 = df5.withColumn("segments_start",F.split(F.col("segments_start"), ",\s*").cast(ArrayType(FloatType())).alias("segments_start"))
#Explode
df6 = df6.select(col('track_id'),explode(col('segments_start')).alias('segments_start'))
# create bucketizer
from pyspark.ml.feature import Bucketizer
splits = [0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360,375,390,405,420,435,450]
splits_dict = {i:splits[i] for i in range(len(splits))}
bucketizer = Bucketizer(splits=splits, inputCol="segments_start",outputCol="segments")
# bucketed dataframe
bucketed = bucketizer.setHandleInvalid('skip').transform(df6)
bucketed = bucketed.replace(to_replace=splits_dict, subset=['result'])

bucketed_2 = bucketed.groupBy(col('track_id'),col('segments')).count()
df7 = bucketed_2.groupBy('track_id').pivot('segments').sum()

#drop columns
cols = ("0.0_sum(segments)","1.0_sum(segments)","2.0_sum(segments)","3.0_sum(segments)","4.0_sum(segments)","5.0_sum(segments)","6.0_sum(segments)","7.0_sum(segments)","8.0_sum(segments)","9.0_sum(segments)","10.0_sum(segments)","11.0_sum(segments)","12.0_sum(segments)","13.0_sum(segments)","14.0_sum(segments)","15.0_sum(segments)","16.0_sum(segments)","17.0_sum(segments)","18.0_sum(segments)","19.0_sum(segments)","20.0_sum(segments)","21.0_sum(segments)","22.0_sum(segments)","23.0_sum(segments)","24.0_sum(segments)","25.0_sum(segments)","26.0_sum(segments)","27.0_sum(segments)")
df7 = df7.drop(*cols)
df7 = df7.fillna(0)

#Rename columns
df8 = df7.select(col('track_id'),col('`0.0_sum(count)`').alias('segment_0_sum'),col('`1.0_sum(count)`').alias('segment_1_sum'),col('`2.0_sum(count)`').alias('segment_2_sum'),col('`3.0_sum(count)`').alias('segment_3_sum'),col('`4.0_sum(count)`').alias('segment_4_sum'),col('`5.0_sum(count)`').alias('segment_5_sum'),col('`6.0_sum(count)`').alias('segment_6_sum'),col('`7.0_sum(count)`').alias('segment_7_sum'),col('`8.0_sum(count)`').alias('segment_8_sum'),col('`9.0_sum(count)`').alias('segment_9_sum'),col('`10.0_sum(count)`').alias('segment_10_sum'),col('`11.0_sum(count)`').alias('segment_11_sum'),col('`12.0_sum(count)`').alias('segment_12_sum'),col('`13.0_sum(count)`').alias('segment_13_sum'),col('`14.0_sum(count)`').alias('segment_14_sum'),col('`15.0_sum(count)`').alias('segment_15_sum'),col('`16.0_sum(count)`').alias('segment_16_sum'),col('`17.0_sum(count)`').alias('segment_17_sum'),col('`18.0_sum(count)`').alias('segment_18_sum'),col('`19.0_sum(count)`').alias('segment_19_sum'),col('`20.0_sum(count)`').alias('segment_20_sum'),col('`21.0_sum(count)`').alias('segment_21_sum'),col('`22.0_sum(count)`').alias('segment_22_sum'),col('`23.0_sum(count)`').alias('segment_23_sum'),col('`24.0_sum(count)`').alias('segment_24_sum'),col('`25.0_sum(count)`').alias('segment_25_sum'),col('`26.0_sum(count)`').alias('segment_26_sum'),col('`27.0_sum(count)`').alias('segment_27_sum'))

#join with main table
df_joined = df3.join(df8, how = 'inner', on = 'track_id')


#tatums_start
df4 = df3.select(col('track_id'),col('tatums_start'))
#exploding
#remove the array brackets at the start and end of the string
df5 = df4.withColumn("tatums_start", F.expr("substring(tatums_start, 2, length(tatums_start) - 2)"))
#convert string to float
df6 = df5.withColumn("tatums_start",F.split(F.col("tatums_start"), ",\s*").cast(ArrayType(FloatType())).alias("tatums_start"))
#Explode
df6 = df6.select(col('track_id'),explode(col('tatums_start')).alias('tatums_start'))
# create bucketizer
bucketizer = Bucketizer(splits=splits, inputCol="tatums_start",outputCol="tatums")
# bucketed dataframe
bucketed = bucketizer.setHandleInvalid('skip').transform(df6)
bucketed = bucketed.replace(to_replace=splits_dict, subset=['result'])

bucketed_2 = bucketed.groupBy(col('track_id'),col('tatums')).count()
df7 = bucketed_2.groupBy('track_id').pivot('tatums').sum()

#drop columns
cols = ("0.0_sum(tatums)","1.0_sum(tatums)","2.0_sum(tatums)","3.0_sum(tatums)","4.0_sum(tatums)","5.0_sum(tatums)","6.0_sum(tatums)","7.0_sum(tatums)","8.0_sum(tatums)","9.0_sum(tatums)","10.0_sum(tatums)","11.0_sum(tatums)","12.0_sum(tatums)","13.0_sum(tatums)","14.0_sum(tatums)","15.0_sum(tatums)","16.0_sum(tatums)","17.0_sum(tatums)","18.0_sum(tatums)","19.0_sum(tatums)","20.0_sum(tatums)","21.0_sum(tatums)","22.0_sum(tatums)","23.0_sum(tatums)","24.0_sum(tatums)","25.0_sum(tatums)","26.0_sum(tatums)","27.0_sum(tatums)")
df7 = df7.drop(*cols)
df7 = df7.fillna(0)

#Rename columns
df8 = df7.select(col('track_id'),col('`0.0_sum(count)`').alias('tatum_0_sum'),col('`1.0_sum(count)`').alias('tatum_1_sum'),col('`2.0_sum(count)`').alias('tatum_2_sum'),col('`3.0_sum(count)`').alias('tatum_3_sum'),col('`4.0_sum(count)`').alias('tatum_4_sum'),col('`5.0_sum(count)`').alias('tatum_5_sum'),col('`6.0_sum(count)`').alias('tatum_6_sum'),col('`7.0_sum(count)`').alias('tatum_7_sum'),col('`8.0_sum(count)`').alias('tatum_8_sum'),col('`9.0_sum(count)`').alias('tatum_9_sum'),col('`10.0_sum(count)`').alias('tatum_10_sum'),col('`11.0_sum(count)`').alias('tatum_11_sum'),col('`12.0_sum(count)`').alias('tatum_12_sum'),col('`13.0_sum(count)`').alias('tatum_13_sum'),col('`14.0_sum(count)`').alias('tatum_14_sum'),col('`15.0_sum(count)`').alias('tatum_15_sum'),col('`16.0_sum(count)`').alias('tatum_16_sum'),col('`17.0_sum(count)`').alias('tatum_17_sum'),col('`18.0_sum(count)`').alias('tatum_18_sum'),col('`19.0_sum(count)`').alias('tatum_19_sum'),col('`20.0_sum(count)`').alias('tatum_20_sum'),col('`21.0_sum(count)`').alias('tatum_21_sum'),col('`22.0_sum(count)`').alias('tatum_22_sum'),col('`23.0_sum(count)`').alias('tatum_23_sum'),col('`24.0_sum(count)`').alias('tatum_24_sum'),col('`25.0_sum(count)`').alias('tatum_25_sum'),col('`26.0_sum(count)`').alias('tatum_26_sum'),col('`27.0_sum(count)`').alias('tatum_27_sum'))

#join with main table
df_joined_2 = df_joined.join(df8, how = 'inner', on = 'track_id')

#Drop final columns
cols = ("artist_playmeid","artist_terms","artist_terms_freq","artist_terms_weight","audio_md5","bars_confidence","bars_start","beats_confidence","beats_start","danceability","key_confidence","energy","mode_confidence","release","release_7digitalid","sections_confidence","segments_confidence","segments_loudness_max","segments_loudness_max_time","segments_loudness_start","segments_pitches","segments_start","segments_timbre","similar_artists","song_hotttnesss","start_of_fade_out","tatums_confidence","tatums_start","time_signature","time_signature_confidence","track_7digitalid","duration_vector","duration_std_vector","duration_std","loudness_vector","loudness_std_vector","loudness_std","end_of_fade_in","sections_start")
df_final = df_joined_2.drop(*cols)

df_final.write.format('json').save('/user/s2733226/project/segments_tatums_preprocessed')