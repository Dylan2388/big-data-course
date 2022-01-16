# Import packages
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import *
from ast import expr

# reading the data
df = spark.read.json("/user/s2733226/project/msd_lastfm_tags_1/part-00049-c3ff1628-3da5-42c1-ac31-46eadf3f68a9-c000.json")
# filtering only three columns
sections_ts = df.select("sections_confidence", "sections_start", "duration")
# removing and substract []
df_section_start = sections_ts.withColumn("sections_start", F.expr("substring(sections_start, 2, length(sections_start) - 2)"))
# converting string to array and changing the datatypes to float types
df_section_start_array = df_section_start.withColumn("sections_start",F.split(F.col("sections_start"), ",\s*").cast(ArrayType(FloatType())).alias("sections_start"))
df_section_start_array = df_section_start_array.withColumn("duration",F.split(F.col("duration"), ",\s*").cast(ArrayType(FloatType())).alias("duration"))
# joined the two columns
joined = df_section_start_array.withColumn("start_joined", F.concat(col("sections_start"), col("duration")))
# counting the length of start sections
res = joined.withColumn("length_startsections", size(joined.sections_start))
# calculate the min-max 
res = res.withColumn("start_joined_substring", F.expr("slice(start_joined, 2, SIZE(start_joined))"))
res = res.withColumn("min-max", expr("transform(arrays_zip(start_joined_substring, start_joined), x -> x.start_joined_substring - x.start_joined)"))
# calculate the average
def normalise(a, dist):
    return [dist for element in a /  element]
a = res.withColumn('average', F.udf(normalise, ArrayType(FloatType()))(res['duration'], res['length_startsections']))

