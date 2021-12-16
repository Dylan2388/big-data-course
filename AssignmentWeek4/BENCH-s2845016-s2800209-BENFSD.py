"""
Group 17
Pham Nguyen Hoang Dung - s2845016
Silvi Fitria - s2800209

Run time: time spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=10 --executor-memory 2G  BENCH-s2845016-s2800209-BENFSD.py > logfile_BENCH.txt 2>&1 /dev/null

Variable of interest:
1. How much data you put in? (MB, or number of files)
2. How much compute you use? (number of executors) (2, 5, 10)
3. Runtime of the program? (Dash board: http://ctit048.ewi.utwente.nl:8088/cluster) 

Fix (1) or (2)
Vary the other
Measure (3)

Requirements:
1. At least 5 measurement times
2. Increase the data size or executors (big data size). What size match the number of executor?

Output: A graphical plot showing 2 variables of either
1. Size of the input data
2. Number of executors
3. Runtime of the program


"""

### Code taken from Tweet_Selection
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

### Fix number of executors: 10, executor memory: 2G. Overall, the total capacity of executors is 20GB. (Actual allocated memory: 32768MB)
### A day of tweets is ~1.6 GB compressed. We assume after extracting, the amount of data is double or tripled (assume 3-4.5GB)
### 1-day data: 3-4.5GB  
PATH_1_DAY = "/data/doina/Twitter-Archive.org/2017-01/1[0-0]/*/*.json.bz2"
### 2-day data: 6-9GB
PATH_2_DAY = "/data/doina/Twitter-Archive.org/2017-01/1[0-1]/*/*.json.bz2"
### 3-day data: 9-13.5GB
PATH_3_DAY = "/data/doina/Twitter-Archive.org/2017-01/1[0-2]/*/*.json.bz2"
### 4-day data: 12-18GB
PATH_4_DAY = "/data/doina/Twitter-Archive.org/2017-01/1[0-3]/*/*.json.bz2"
### 5-day data: 15-22.5GB (Assumption: The plot should be bumped at this point or onward)
PATH_5_DAY = "/data/doina/Twitter-Archive.org/2017-01/1[0-4]/*/*.json.bz2"
### 6-day data: 18-27GB 
PATH_6_DAY = "/data/doina/Twitter-Archive.org/2017-01/1[0-5]/*/*.json.bz2"
### 7-day data: 21-31.5GB 
PATH_7_DAY = "/data/doina/Twitter-Archive.org/2017-01/1[0-6]/*/*.json.bz2"
### 8-day data: 24-36GB
PATH_8_DAY = "/data/doina/Twitter-Archive.org/2017-01/1[0-7]/*/*.json.bz2"
### 9-day data: 27-40.5GB
PATH_9_DAY = "/data/doina/Twitter-Archive.org/2017-01/1[0-8]/*/*.json.bz2"
### 10-day data: 30-45GB
PATH_10_DAY = "/data/doina/Twitter-Archive.org/2017-01/1[0-9]/*/*.json.bz2"

######## Change the date: E.g: to test the system

# a regexp, case-insensitive matching; this was a topic of interest on those dates (Jan 2017)
KEYWORDS = "(inauguration)|(whitehouse)|(washington)|(president)|(obama)|(trump)"

# always creates a new folder to write in (clean up your old folders though)
now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

tweets = spark.read.json(PATH_10_DAY) \
    .filter(col("text").isNotNull()) \
    .select(col("text")) \
    .filter(col("text").rlike(KEYWORDS).alias("text")) \
    .write.text("tweet_selection-"+now)
    

