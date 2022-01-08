from pyspark import SparkContext
sc = SparkContext(appName="hashtags")
sc.setLogLevel("ERROR")

from pyspark.sql.functions import col

#Read
df1 = spark.read.option("header",True).option("quote", "\"").option("escape", "\"").csv("/data/doina/OSCD-MillionSongDataset/*.csv")

#Select
df_artist_mbid = df1.select(col('track_id'),col('artist_mbid'))
df_artist_name = df1.select(col('track_id'),col('artist_name'))
df_artist_playmeid = df1.select(col('track_id'),col('artist_playmeid'))
df_artist_terms = df1.select(col('track_id'),col('artist_terms'))
df_artist_terms_freq = df1.select(col('track_id'),col('artist_terms_freq'))
df_artist_terms_weight = df1.select(col('track_id'),col('artist_terms_weight'))
df_audio_md5 = df1.select(col('track_id'),col('audio_md5'))
df_bars_confidence = df1.select(col('track_id'),col('bars_confidence'))
df_bars_start = df1.select(col('track_id'),col('bars_start'))
df_beats_confidence = df1.select(col('track_id'),col('beats_confidence'))
df_beats_start = df1.select(col('track_id'),col('beats_start'))
df_end_of_fade_in = df1.select(col('track_id'),col('end_of_fade_in'))
df_duration = df1.select(col('track_id'),col('duration'))
df_energy = df1.select(col('track_id'),col('energy'))
df_key = df1.select(col('track_id'),col('key'))
df_key_confidence = df1.select(col('track_id'),col('key_confidence'))
df_loudness = df1.select(col('track_id'),col('loudness'))
df_mode = df1.select(col('track_id'),col('mode'))
df_mode_confidence = df1.select(col('track_id'),col('mode_confidence'))
df_release = df1.select(col('track_id'),col('release'))
df_release_7digitalid = df1.select(col('track_id'),col('release_7digitalid'))
df_sections_confidence = df1.select(col('track_id'),col('sections_confidence'))
df_sections_start = df1.select(col('track_id'),col('sections_start'))
df_segments_confidence = df1.select(col('track_id'),col('segments_confidence'))
df_segments_loudness_max = df1.select(col('track_id'),col('segments_loudness_max'))
df_segments_loudness_max_time = df1.select(col('track_id'),col('segments_loudness_max_time'))
df_segments_loudness_start = df1.select(col('track_id'),col('segments_loudness_start'))
df_segments_pitches = df1.select(col('track_id'),col('segments_pitches'))
df_segments_start = df1.select(col('track_id'),col('segments_start'))
df_segments_timbre = df1.select(col('track_id'),col('segments_timbre'))
df_similar_artists = df1.select(col('track_id'),col('similar_artists'))
df_song_hotttnesss = df1.select(col('track_id'),col('song_hotttnesss'))
df_song_id = df1.select(col('track_id'),col('song_id'))
df_start_of_fade_out = df1.select(col('track_id'),col('start_of_fade_out'))
df_tatums_confidence = df1.select(col('track_id'),col('tatums_confidence'))
df_tatums_start = df1.select(col('track_id'),col('tatums_start'))
df_tempo = df1.select(col('track_id'),col('tempo'))
df_time_signature = df1.select(col('track_id'),col('time_signature'))
df_time_signature_confidence = df1.select(col('track_id'),col('time_signature_confidence'))
df_title = df1.select(col('track_id'),col('title'))
df_track_7digitalid = df1.select(col('track_id'),col('track_7digitalid'))
df_year = df1.select(col('track_id'),col('year'))

#Writing
df_artist_mbid.write.csv("/user/s2733226/project/column_data/artist_mbid")
df_artist_name.write.csv("/user/s2733226/project/column_data/artist_name")
df_artist_playmeid.write.csv("/user/s2733226/project/column_data/artist_playmeid")
df_artist_terms.write.csv("/user/s2733226/project/column_data/artist_terms")
df_artist_terms_freq.write.csv("/user/s2733226/project/column_data/artist_terms_freq")
df_artist_terms_weight.write.csv("/user/s2733226/project/column_data/artist_terms_weight")
df_audio_md5.write.csv("/user/s2733226/project/column_data/audio_md5")
df_bars_confidence.write.csv("/user/s2733226/project/column_data/bars_confidence")
df_bars_start.write.csv("/user/s2733226/project/column_data/bars_start")
df_beats_confidence.write.csv("/user/s2733226/project/column_data/beats_confidence")
df_beats_start.write.csv("/user/s2733226/project/column_data/beats_start")
df_end_of_fade_in.write.csv("/user/s2733226/project/column_data/end_of_fade_in")
df_duration.write.csv("/user/s2733226/project/column_data/duration")
df_energy.write.csv("/user/s2733226/project/column_data/energy")
df_key.write.csv("/user/s2733226/project/column_data/key")
df_key_confidence.write.csv("/user/s2733226/project/column_data/key_confidence")
df_loudness.write.csv("/user/s2733226/project/column_data/loudness")
df_mode.write.csv("/user/s2733226/project/column_data/mode")
df_mode_confidence.write.csv("/user/s2733226/project/column_data/mode_confidence")
df_release.write.csv("/user/s2733226/project/column_data/release")
df_release_7digitalid.write.csv("/user/s2733226/project/column_data/release_7digitalid")
df_sections_confidence.write.csv("/user/s2733226/project/column_data/sections_confidence")
df_sections_start.write.csv("/user/s2733226/project/column_data/sections_start")
df_segments_confidence.write.csv("/user/s2733226/project/column_data/segments_confidence")
df_segments_loudness_max.write.csv("/user/s2733226/project/column_data/segments_loudness_max")
df_segments_loudness_max_time.write.csv("/user/s2733226/project/column_data/segments_loudness_max_time")
df_segments_loudness_start.write.csv("/user/s2733226/project/column_data/segments_loudness_start")
df_segments_pitches.write.csv("/user/s2733226/project/column_data/segments_pitches")
df_segments_start.write.csv("/user/s2733226/project/column_data/segments_start")
df_segments_timbre.write.csv("/user/s2733226/project/column_data/segments_timbre")
df_similar_artists.write.csv("/user/s2733226/project/column_data/similar_artists")
df_song_hotttnesss.write.csv("/user/s2733226/project/column_data/song_hotttnesss")
df_song_id.write.csv("/user/s2733226/project/column_data/song_id")
df_start_of_fade_out.write.csv("/user/s2733226/project/column_data/start_of_fade_out")
df_tatums_confidence.write.csv("/user/s2733226/project/column_data/tatums_confidence")
df_tatums_start.write.csv("/user/s2733226/project/column_data/tatums_start")
df_tempo.write.csv("/user/s2733226/project/column_data/tempo")
df_time_signature.write.csv("/user/s2733226/project/column_data/time_signature")
df_time_signature_confidence.write.csv("/user/s2733226/project/column_data/time_signature_confidence")
df_title.write.csv("/user/s2733226/project/column_data/title")
df_track_7digitalid.write.csv("/user/s2733226/project/column_data/track_7digitalid")
df_year.write.csv("/user/s2733226/project/column_data/year")