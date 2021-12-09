"""
DISTRIBUTED mode: Run me from ctithead1.ewi.utwente.nl (the master) with:
    time spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=10 tweet_selection.py

(No 2>/dev/null because the messages here are useful: get the applicationId !)

DASHBOARD at http://ctit048.ewi.utwente.nl:8088/cluster .

You can raise maxExecutors above 10 IF the cluster is NOT in heavy use (but even then, up to 50, because the 
dynamic allocation gives <= 4 cores to each executor for some stages, so 1/3 of the 660 cores on the cluster). 
The memory allocation per executor is capped at 8 GB.

The Yarn logs are where you can understand any error:
    yarn logs --applicationId application_1508921708233_00...

Any files in output are saved in parts (one per partition! written out in parallel) in a folder in the 
user's HDFS home directory under /user/. You may get many output files. If you add coalesce(small_number) 
before saving the output, you get few files, BUT your program will be slower. 
"""

from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

# Runtime for 1 hour of tweets: 0m54.753s, 
#         for 1 day  of tweets: 2m19.612s,
#         for 2 days of tweets: 3m30.032s, etc.
# A day of tweets is ~1.44 GB compressed (in 1440 files).
PATH = "/data/doina/Twitter-Archive.org/2017-01/0[1-2]/*/*.json.bz2"

# a regexp, case-insensitive matching; this was a topic of interest on those dates (Jan 2017)
KEYWORDS = "(inauguration)|(whitehouse)|(washington)|(president)|(obama)|(trump)"

# always creates a new folder to write in (clean up your old folders though)
now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

tweets = spark.read.json(PATH) \
    .filter(col("text").isNotNull()) \
    .select(col("text")) \
    .filter(col("text").rlike(KEYWORDS).alias("text")) \
    .write.text("tweet_selection-"+now)
