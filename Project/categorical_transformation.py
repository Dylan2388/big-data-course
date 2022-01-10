# 1. Time Signature
# Read the text file
df_timesig = spark.read.csv("/user/s2733226/project/column_data/time_signature/part-0*.csv")
# Showing the data
df_timesig.show()
# Rename the column names
df_timesig = df_timesig.selectExpr("_c0 as name", "_c1 as value")
# First step of transformation
stringIndexer = StringIndexer(inputCol="value", outputCol="categorical").fit(df_timesig)
indexed_df = stringIndexer.transform(df_timesig)
indexed_df.drop("bar").show()
# One hot encoder
encoder = OneHotEncoderEstimator(inputCols=["categorical"], outputCols=["categorical_vector"])
model = encoder.fit(indexed_df)
encoded = model.transform(indexed_df)
encoded.show()

# 2. Key 
# Read the text file
df_key = spark.read.csv("/user/s2733226/project/column_data/key/part-0*.csv")
# Rename the column names
df_key = df_key.selectExpr("_c0 as name", "_c1 as value")
# Showing the data
df_key.show()
# First step of transformation
stringIndexer = StringIndexer(inputCol="value", outputCol="categorical").fit(df_key)
indexed_df = stringIndexer.transform(df_key)
indexed_df.drop("bar").show()
# One hot encoder
encoder = OneHotEncoderEstimator(inputCols=["categorical"], outputCols=["categorical_vector"])
model = encoder.fit(indexed_df)
encoded = model.transform(indexed_df)
encoded.show()

# 3. Mode
# Import packages for transformation
from pyspark.ml.feature import IndexToString
from pyspark.ml.feature import OneHotEncoderEstimator

# Read the text file
df_mode = spark.read.csv("/user/s2733226/project/column_data/mode/part-0*.csv")
# Rename the column names
df_mode = df_mode.selectExpr("_c0 as name", "_c1 as value")
# Showing the data
df_mode.show()
# First step of transformation
stringIndexer = StringIndexer(inputCol="value", outputCol="categorical").fit(df_mode)
indexed_df = stringIndexer.transform(df_mode)
indexed_df.drop("bar").show()
# One hot encoder
encoder = OneHotEncoderEstimator(inputCols=["categorical"], outputCols=["categorical_vector"])
model = encoder.fit(indexed_df)
encoded = model.transform(indexed_df)
encoded.show()