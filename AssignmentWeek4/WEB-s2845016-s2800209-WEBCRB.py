"""
Group 17
Pham Nguyen Hoang Dung - s2845016
Silvi Fitria - s2800209
Run time: time spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=10 --num-executors 2 --executor-cores 4 --executor-memory 4G  WEB-s2845016-s2800209-WEBCRB.py > logfile_WEB.txt 2>&1 /dev/null
--num-executors 2 --executor-cores 2 --executor-memory 2G
real	2m44.510s
user	0m11.648s
sys	    0m2.388s

--num-executors 2 --executor-cores 4 --executor-memory 4G
real	2m35.079s
user	0m12.188s
sys	    0m2.343s

DASHBOARD at http://ctit048.ewi.utwente.nl:8088/cluster .

Result HDFS path: /user/s2845016/WEB
Remove folder: hdfs dfs -rm -r /user/s2845016/WEB
Check folder exist: hdfs dfs -ls /user/s2845016
"""

# Import packages and initiate session
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("WEB").getOrCreate()

### Small input
small_input_1 = "/data/doina/WebInsight/2020-07-13/1M.2020-07-13-aa.gz"
small_input_2 = "/data/doina/WebInsight/2020-09-14/1M.2020-09-14-aa.gz"
### Big input
big_input_1 = "/data/doina/WebInsight/2020-07-13/*.gz"
big_input_2 = "/data/doina/WebInsight/2020-09-14/*.gz"

### Read data from input
df1_1 = spark.read.json(big_input_1)
df2_1 = spark.read.json(big_input_2)

### Select url and textSize column
df1_2 = df1_1.select([df1_1["url"], df1_1["fetch"]["textSize"].alias("textSize1")])
df2_2 = df2_1.select([df2_1["url"], df2_1["fetch"]["textSize"].alias("textSize2")])

### Innner Join (Filter urls that appear in both dataframe)
df = df1_2.join(df2_2, on="url", how="inner")
### Filter urls that have at least 1 textSize different than 0
df1 = df.filter((F.col("textSize1") != 0) | (F.col("textSize2") != 0))
### Compute size diff
df2 = df1.select([F.col("url"), 
                  (F.col("textSize1") - F.col("textSize2")).alias("sizediff")])
### Sort and redistribute
df3 = df2.sort('sizediff', ascending=True).coalesce(5)
### Save to disk
df3.write.json("/user/s2845016/WEB")

