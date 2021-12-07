"""
Group 17
Pham Nguyen Hoang Dung - s2845016
Silvi Fitria - s2800209

Runs with Python3:
pyspark --conf "spark.pyspark.python=/usr/bin/python3.6" --conf "spark.pyspark.driver.python=/usr/bin/python3.6"

Run time: time spark-submit --conf "spark.pyspark.python=/usr/bin/python3.6" --conf "spark.pyspark.driver.python=/usr/bin/python3.6" GAMES3-s2845016-s2800209-GMTSMG.py > logfile.txt 2>&1 /dev/null
real	0m17.383s
user	0m51.084s
sys	    0m2.547s

It seems like the product with exact ASIN ID is no longer on Amazon anymore, since I couldn't find it.
"""

## Import packages and initiate session
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import json
spark = SparkSession.builder.appName('GAMES').getOrCreate()
    
## Read data from json file
df = spark.read.json("/data/doina/UCSD-Amazon-Data/meta_Video_Games.json.gz")
## Get "also_brought" column to a new dataframe
df2 = df.select(F.col('related')['also_bought'].alias('item')).where(F.col('item').isNotNull())
## Explode new dataframe
df3 = df2.select(F.explode(df2['item']).alias('item'))
## Group dataframe by its id, then count
df4 = df3.groupby('item').count()
## Sort by number of appearance 
df5 = df4.sort('count', ascending=False)
## Get the first id
record_id = df5.take(1)[0]['item']
## Get the record of related id
record = df.filter(df['asin'] == record_id)
## Jsonify for pretty print
record_json = json.loads(record.toJSON().collect()[0])
print(json.dumps(record_json, indent=4))




