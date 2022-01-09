#Command: time spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=50 --executor-memory 6G  msd_tagged.py > logfile_msd_tagged.txt 2>&1 /dev/null
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from pyspark.sql.functions import col

#Read all MSD data
df1 = spark.read.option("header",True).option("quote", "\"").option("escape", "\"").csv("/data/doina/OSCD-MillionSongDataset/*.csv")

#Read all lastfm data
lastfm = spark.read.json("/user/s2733226/project/lastfm_preprocessed/*.json")

#Join tables
df_joined = df1.join(lastfm, 'track_id')

#Remove duplicates, select required columns
df_joined = df_joined.select(col('track_id'),col('artist_mbid'),col('artist_name'),col('artist_playmeid'),col('artist_terms'),col('artist_terms_freq'),col('artist_terms_weight'),col('audio_md5'),col('bars_confidence'),col('bars_start'),col('beats_confidence'),col('beats_start'),col('danceability'),col('duration'),col('end_of_fade_in'),col('energy'),col('key'),col('key_confidence'),col('loudness'),col('mode'),col('mode_confidence'),col('release'),col('release_7digitalid'),col('sections_confidence'),col('sections_start'),col('segments_confidence'),col('segments_loudness_max'),col('segments_loudness_max_time'),col('segments_loudness_start'),col('segments_pitches'),col('segments_start'),col('segments_timbre'),col('similar_artists'),col('song_hotttnesss'),col('song_id'),col('start_of_fade_out'),col('tatums_confidence'),col('tatums_start'),col('tempo'),col('time_signature'),col('time_signature_confidence'),col('title'),col('track_7digitalid'),col('year'),col('similars'),col('tags')).distinct().coalesce(50)

#Write as json
df_joined.write.format('json').save('/user/s2733226/project/msd_lastfm_tags')