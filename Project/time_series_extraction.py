import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import PCA


spark = SparkSession.builder.getOrCreate()

df = spark.read.csv("/user/s2733226/project/column_data/sections_start/part-0*.csv")
df = df.selectExpr("_c0 as name", "_c1 as value")

df_section_start = df.withColumn("value", F.expr("substring(value, 2, length(value) - 2)"))
df_section_start_array = df_section_start.withColumn("value",F.split(F.col("value"), ",\s*").cast(ArrayType(FloatType())).alias("value"))

#### Vectorize data from array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
df_section_start_vector = df_section_start_array.select(
    df_section_start_array["value"], 
    list_to_vector_udf(df_section_start_array["value"]).alias("value_vector")
)

# PCA algorithm
pca = PCA(k=3, inputCol="value_vector", outputCol="value_vector_pca")
model = pca.fit(df_section_start_vector)

result = model.transform(df_section_start_vector).select("value_vector_pca")
result.show(truncate=False)


df_section_start_vector.show(5, truncate=False)