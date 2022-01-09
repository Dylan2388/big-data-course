"""[summary]
pyspark --conf "spark.pyspark.python=/usr/bin/python3.6" --conf "spark.pyspark.driver.python=/usr/bin/python3.6"
"""


from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import MultilayerPerceptronClassifier

spark = SparkSession.builder.getOrCreate()

df = spark.createDataFrame([
    (0.0, Vectors.dense([0.0, 0.0])),
    (1.0, Vectors.dense([0.0, 1.0])),
    (1.0, Vectors.dense([1.0, 0.0])),
    (0.0, Vectors.dense([1.0, 1.0]))], ["label", "features"])
mlp = MultilayerPerceptronClassifier(layers=[2, 2, 2], seed=123)