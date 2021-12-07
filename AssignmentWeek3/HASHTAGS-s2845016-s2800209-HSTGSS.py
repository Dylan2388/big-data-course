"""
Group 17
Pham Nguyen Hoang Dung - s2845016
Silvi Fitria - s2800209
Run time: time spark-submit HASHTAGS-s2845016-s2800209-HSTGSS.py 2> /dev/null
real	0m16.212s
user	1m25.549s
sys	    0m2.483s
"""

# Import packages
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sc = SparkContext()
sc.setLogLevel("ERROR")
spark = SparkSession.builder.getOrCreate()

# Load data
df = spark.read.json("/data/doina/Twitter-Archive.org/2020-01/01/00/0*.json.bz2")
# Filter to hashtag twitter only
df1 = df.filter(col('text').contains('#')).select(col('entities')['hashtags'].alias('hashtags'),col('text'))
# Flatmap the hastag into each row
df2 = df1.selectExpr("explode(hashtags) as tag1").selectExpr("tag1.*")
df3 = df2.select(explode(split(col('text'),",")).alias('tags'))
df4 = df3.select(col('tags'))
# Count each hashtags
text = [row.tags for row in df4.collect()]
rdd = sc.parallelize(text)
words = rdd.map(lambda x: (x,1))
wordcount = words.reduceByKey(lambda a, b: a+b).toDF()
result = wordcount.selectExpr("_1 as hashtags", "_2 as count")
# Output 
final = result.orderBy(col("count").desc())
final.show()
