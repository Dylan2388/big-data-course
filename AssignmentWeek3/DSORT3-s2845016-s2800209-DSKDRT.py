"""
Group 17
Pham Nguyen Hoang Dung - s2845016
Silvi Fitria - s2800209

Runs with Python3:
pyspark --conf "spark.pyspark.python=/usr/bin/python3.6" --conf "spark.pyspark.driver.python=/usr/bin/python3.6"

Run time: time spark-submit --conf "spark.pyspark.python=/usr/bin/python3.6" --conf "spark.pyspark.driver.python=/usr/bin/python3.6" DSORT3-s2845016-s2800209-DSKDRT.py > logfile.txt 2>&1 /dev/null
SPARK SQL:
real	0m14.214s
user	0m56.497s
sys	    0m2.591s

RDD:

Result HDFS path: /user/s2845016/DSORT
Remove folder: hdfs dfs -rm -r /user/s2845016/DSORT
Check folder exist: hdfs dfs -ls /user/s2845016
"""
## Import packages and initiate session
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('DSORT').getOrCreate()

## Read data from text file
df = spark.read.text("/data/doina/integers.txt")
## Split string into array
df2 = df.select(F.split(df['value'], ",").alias('value'))
## Explode all array into 1 column
df3 = df2.select(F.explode(df2['value']).alias('value'))
## Sort the column (using spark sql)
# df4 = df3.select(df3['value'].cast("int")).sort('value', ascending=True)
# ## Repartition base on column
# df5 = df4.repartitionByRange(10, "value")
# ## Map back to rdd then save to files
# rdd = df5.rdd.map(lambda x: x[0])
# rdd.saveAsTextFile("/user/s2845016/DSORT")


### SORT USING RDD
rdd_1 = df3.rdd.map(lambda x: int(x[0])).repartition(10)
rdd_2 = rdd_1.sortBy(lambda x: x)
rdd_2.saveAsTextFile("/user/s2845016/DSORT_RDD")


