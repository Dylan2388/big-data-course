"""
Group 17
Pham Nguyen Hoang Dung - s2845016
Silvi Fitria - s2800209
Run time: time spark-submit WEB-s2845016-w2800209-WEBCRB.py 2> /dev/null

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
big_input_1 = "/data/doina/WebInsight/2020-07-13/"
big_input_2 = "/data/doina/WebInsight/2020-09-14"

df1_1 = spark.read.json(small_input_1)
df2_1 = spark.read.json(small_input_2)


df1_2 = df1_1.select([df1_1["url"], df1_1["fetch"]["textSize"].alias["textSize"]])
df2_2 = df2_1.select([df2_1["url"], df2_1["fetch"]["textSize"].alias["textSize"]])

