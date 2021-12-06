"""
Group 17
Pham Nguyen Hoang Dung - s2845016
Silvi Fitria - s2800209

Runs with Python3:
pyspark --conf "spark.pyspark.python=/usr/bin/python3.6" --conf "spark.pyspark.driver.python=/usr/bin/python3.6"

Run time: time spark-submit --conf "spark.pyspark.python=/usr/bin/python3.6" --conf "spark.pyspark.driver.python=/usr/bin/python3.6" DSORT-s2845016-s2800209-DSKDRT.py > logfile.txt 2>&1 /dev/null
real	0m17.383s
user	0m51.084s
sys	    0m2.547s

Result HDFS path: /user/s2845016/DSORT
Remove folder: hdfs dfs -rm -r /user/s2845016/DSORT
"""
## Import packages and initiate session
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('DSORT').getOrCreate()

df = spark.read.text("/data/doina/integers.txt")
df2 = df.select(F.split(df['value'], ",").alias('value'))
df3 = df2.select(F.explode(df2['value']).alias('value'))
df4 = df3.select(df3['value'].cast("int")).sort('value', ascending=True)
df5 = df4.repartitionByRange(10, "value")
rdd = df5.rdd.map(lambda x: x[0])
rdd.saveAsTextFile("/user/s2845016/DSORT")
print(rdd.take(1)[0])


