#Command: time spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=20 --executor-memory 4G lastfm_preprocessing.py > logfile_lastfm_preprocessing.txt 2>&1 /dev/null
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from pyspark.sql.functions import col, explode

#Read all tags
lastfm = spark.read.json("/user/s2733226/project/lastfm_data.json/*.json")
#Explode tags column
lastfm_1 = lastfm.select(col('artist'),col('similars'),explode(col('tags')).alias('tags'),col('title'),col('track_id'))
#Select confidence higher than 20
lastfm_2 = lastfm_1.filter(col('tags')[1]>20)
#Separate tags and tags_confidence
lastfm_3 = lastfm_2.select(col('artist'),col('similars'),col('tags')[0].alias('tags'),col('tags')[1].alias('tags_confidence'),col('title'),col('track_id'))
#Count number of tags
lastfm_tagcount = lastfm_3.groupBy('tags').count()
#Assign each tag with its count, and remove tags with count less than 10,000
lastfm_4 = lastfm_3.join(lastfm_tagcount,'tags', how='left').filter(col('count') > 1000)

#Combine tags and tags_confidence into a single column
import pyspark.sql.functions as f
lastfm_5 = lastfm_4.select(col('track_id'),col('artist'),col('similars'),f.array(col('tags'),col('tags_confidence')).alias('tags'),col('title'),col('count'))

#Grouping all tags for unique track_id
from pyspark.sql.functions import collect_list
grouped_tags = lastfm_5.groupBy(col('track_id'),col('artist'),col('similars'),col('title')).agg(collect_list(col('tags')).alias('tags'))

#Taking distinct and writing to file
grouped_tags.write.format('json').save('/user/s2733226/project/lastfm_preprocessed')
