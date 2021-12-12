"""
Group 17
Pham Nguyen Hoang Dung - s2845016
Silvi Fitria - s2800209
Run time: time spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=10 WEB-s2845016-s2800209-WEBCRB.py > logfile_WEB.txt 2>&1 /dev/null



Result HDFS path: /user/s2845016/WEB
Remove folder: hdfs dfs -rm -r /user/s2845016/WEB
Check folder exist: hdfs dfs -ls /user/s2845016
"""

# Import packages and initiate session
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("WEB").getOrCreate()
sc = SparkContext(appName="WEB")
sc.setLogLevel("ERROR")



### Small input
small_input_1 = "/data/doina/WebInsight/2020-07-13/1M.2020-07-13-aa.gz"
small_input_2 = "/data/doina/WebInsight/2020-09-14/1M.2020-09-14-aa.gz"
### Big input
big_input_1 = "/data/doina/WebInsight/2020-07-13/*.gz"
big_input_2 = "/data/doina/WebInsight/2020-09-14/*.gz"

df1_1 = spark.read.json(big_input_1)
df2_1 = spark.read.json(big_input_2)

df1_2 = df1_1.select([df1_1["url"], df1_1["fetch"]["textSize"].alias("textSize1")])
df2_2 = df2_1.select([df2_1["url"], df2_1["fetch"]["textSize"].alias("textSize2")])

df = df1_2.join(df2_2, on="url", how="inner")
df1 = df.filter((F.col("textSize1") != 0) | (F.col("textSize2") != 0))
df2 = df1.select([F.col("url"), 
                  (F.col("textSize1") - F.col("textSize2")).alias("sizediff")])
df3 = df2.sort('sizediff', ascending=True).coalesce(5)
df3.write.json("/user/s2845016/WEB")

