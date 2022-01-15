import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator


spark = SparkSession.builder.getOrCreate()

# 1. Time Signature
# Read the text file
df_timesig = spark.read.csv("/user/s2733226/project/column_data/time_signature/part*.csv")
# Showing the data
# df_timesig.show()
# Rename the column names
df_timesig = df_timesig.selectExpr("_c0 as name", "_c1 as value")
# First step of transformation
stringIndexer = StringIndexer(inputCol="value", outputCol="categorical").fit(df_timesig)
df_timesig = stringIndexer.transform(df_timesig)
# df_timesig.drop("bar").show()
# One hot encoder
encoder = OneHotEncoderEstimator(inputCols=["categorical"], outputCols=["categorical_vector"])
model = encoder.fit(df_timesig)
df_timesig = model.transform(df_timesig)
# df_timesig.show()

# 2. Key 
# Read the text file
df_key = spark.read.csv("/user/s2733226/project/column_data/key/part*.csv")
# Rename the column names
df_key = df_key.selectExpr("_c0 as name", "_c1 as value")
# Showing the data
df_key.show()
# First step of transformation
stringIndexer = StringIndexer(inputCol="value", outputCol="categorical").fit(df_key)
df_key = stringIndexer.transform(df_key)
df_key.drop("bar").show()
# One hot encoder
encoder = OneHotEncoderEstimator(inputCols=["categorical"], outputCols=["categorical_vector"])
model = encoder.fit(df_key)
df_key = model.transform(df_key)
df_key.show()

# 3. Mode
# Read the text file
df_mode = spark.read.csv("/user/s2733226/project/column_data/mode/part*.csv")
# Rename the column names
df_mode = df_mode.selectExpr("_c0 as name", "_c1 as value")
# Showing the data
df_mode.show()
# First step of transformation
stringIndexer = StringIndexer(inputCol="value", outputCol="categorical").fit(df_mode)
df_mode = stringIndexer.transform(df_mode)
df_mode.drop("bar").show()
# One hot encoder
encoder = OneHotEncoderEstimator(inputCols=["categorical"], outputCols=["categorical_vector"])
model = encoder.fit(df_mode)
df_mode = model.transform(df_mode)
df_mode.show()