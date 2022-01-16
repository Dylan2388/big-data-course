from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from pyspark.sql.functions import col, explode, udf, concat_ws

df = spark.read.json('/user/s2733226/project/segments_tatums_preprocessed/*.json')

from pyspark.ml.linalg import Vectors, VectorUDT
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator


#convert to vector
to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
#####NUMERICAL DATA
#scaled_duration_vector
#convert struct to vectors
df1 = df.select('alternative', 'artist_mbid', 'artist_name', 'key', 'mode', 'time_signature', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector.*', 'scaled_loudness_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year')
#split struct column into individual columns for each element
df1 = df1.withColumn("values",concat_ws(",",col("values")))
#Convert to float
df1 = df1.withColumn('values',df1['values'].cast("float").alias('values'))
#combine all elements into an array
columns = [F.col("values")]
df2 = df1.withColumn("scaled_duration_vector", F.array(columns)).drop("type","values")
#convert to vector
df = df2.select('alternative', 'artist_mbid', 'artist_name', 'key', 'mode', 'time_signature', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_loudness_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year', to_vector("scaled_duration_vector").alias('scaled_duration_vector'))

#scaled_loudness_vector
#convert struct to vectors
df1 = df.select('alternative', 'artist_mbid', 'artist_name', 'key', 'mode', 'time_signature', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector.*', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year')
#split struct column into individual columns for each element
df1 = df1.withColumn("values",concat_ws(",",col("values")))
#Convert to float
df1 = df1.withColumn('values',df1['values'].cast("float").alias('values'))
#combine all elements into an array
columns = [F.col("values")]
df2 = df1.withColumn("scaled_loudness_vector", F.array(columns)).drop("type","values")
#convert to vector
df = df2.select('alternative', 'artist_mbid', 'artist_name', 'key', 'mode', 'time_signature', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_tempo_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year', to_vector("scaled_loudness_vector").alias('scaled_loudness_vector'))

#scaled_tempo_vector
#convert struct to vectors
df1 = df.select('alternative', 'artist_mbid', 'artist_name', 'key', 'mode', 'time_signature', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector', 'scaled_tempo_vector.*', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year')
#split struct column into individual columns for each element
df1 = df1.withColumn("values",concat_ws(",",col("values")))
#Convert to float
df1 = df1.withColumn('values',df1['values'].cast("float").alias('values'))
#combine all elements into an array
columns = [F.col("values")]
df2 = df1.withColumn("scaled_tempo_vector", F.array(columns)).drop("type","values")
#convert to vector
df = df2.select('alternative', 'artist_mbid', 'artist_name', 'key', 'mode', 'time_signature', 'dance', 'duration', 'electronic', 'indie', 'instrumental', 'jazz', 'key', 'loudness', 'metal', 'mode', 'pop', 'rock', 'scaled_duration_vector', 'scaled_loudness_vector', 'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum', 'similars', 'song_id', 'soul', 'tags', 'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum', 'tempo', 'title', 'track_id', 'year', to_vector("scaled_tempo_vector").alias('scaled_tempo_vector'))


 
######CATEGORICAL DATA
# 1. Time Signature
stringIndexer = StringIndexer(inputCol="time_signature", outputCol="time_signature_index").fit(df)
df = stringIndexer.transform(df)
# One hot encoder
encoder = OneHotEncoderEstimator(inputCols=["time_signature_index"], outputCols=["time_signature_vector"])
model = encoder.fit(df)
df = model.transform(df)

# 2. Key
# First step of transformation
stringIndexer = StringIndexer(inputCol="key", outputCol="key_index").fit(df)
df = stringIndexer.transform(df)
# One hot encoder
encoder = OneHotEncoderEstimator(inputCols=["key_index"], outputCols=["key_vector"])
model = encoder.fit(df)
df = model.transform(df)

# 3. Mode
# First step of transformation
stringIndexer = StringIndexer(inputCol="mode", outputCol="mode_index").fit(df)
df = stringIndexer.transform(df)
# One hot encoder
encoder = OneHotEncoderEstimator(inputCols=["mode_index"], outputCols=["mode_vector"])
model = encoder.fit(df)
df = model.transform(df)
 

################ TRAINING MODEL ####################
### Vectorize Data
from pyspark.ml.feature import VectorAssembler
vectorized_column = ['scaled_duration_vector', 'scaled_loudness_vector', 'scaled_tempo_vector',
                    'key_vector', 'mode_vector', 'time_signature_vector',
                    'segment_0_sum', 'segment_10_sum', 'segment_11_sum', 'segment_12_sum', 'segment_13_sum', 'segment_14_sum', 'segment_15_sum', 
                    'segment_16_sum', 'segment_17_sum', 'segment_18_sum', 'segment_19_sum', 'segment_1_sum', 'segment_20_sum', 'segment_21_sum', 
                    'segment_22_sum', 'segment_23_sum', 'segment_24_sum', 'segment_25_sum', 'segment_26_sum', 'segment_27_sum', 'segment_2_sum', 
                    'segment_3_sum', 'segment_4_sum', 'segment_5_sum', 'segment_6_sum', 'segment_7_sum', 'segment_8_sum', 'segment_9_sum',
                    'tatum_0_sum', 'tatum_10_sum', 'tatum_11_sum', 'tatum_12_sum', 'tatum_13_sum', 'tatum_14_sum', 'tatum_15_sum', 'tatum_16_sum', 
                    'tatum_17_sum', 'tatum_18_sum', 'tatum_19_sum', 'tatum_1_sum', 'tatum_20_sum', 'tatum_21_sum', 'tatum_22_sum', 'tatum_23_sum', 
                    'tatum_24_sum', 'tatum_25_sum', 'tatum_26_sum', 'tatum_27_sum', 'tatum_2_sum', 'tatum_3_sum', 'tatum_4_sum', 'tatum_5_sum', 
                    'tatum_6_sum', 'tatum_7_sum', 'tatum_8_sum', 'tatum_9_sum']


assembler = VectorAssembler().setInputCols(vectorized_column).setOutputCol('features')
alternative_column = ["alternative", "features"]
dance_column = ["dance", "features"]
electronic_column = ["electronic", "features"]
indie_column = ["indie", "features"]
instrumental_column = ["instrumental", "features"]
jazz_column = ["jazz", "features"]
metal_column = ["metal", "features"]
pop_column = ["pop", "features"]
rock_column = ["rock", "features"]
soul_column = ["soul", "features"]
data = assembler.transform(df)



# ############### LOGISTIC REGRESSION - ALTENATIVE #####################
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(alternative_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# lr = LogisticRegression(featuresCol='features', labelCol='alternative')
# model = lr.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "alternative")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='alternative')
# print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


# ############### LOGISTIC REGRESSION - DANCE #####################
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(dance_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# lr = LogisticRegression(featuresCol='features', labelCol='dance')
# model = lr.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "dance")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='dance')
# print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


# ############### LOGISTIC REGRESSION - ELECTRONIC #####################
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(electronic_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# lr = LogisticRegression(featuresCol='features', labelCol='electronic')
# model = lr.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "electronic")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='electronic')
# print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


# ############### LOGISTIC REGRESSION - INDIE #####################
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(indie_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# lr = LogisticRegression(featuresCol='features', labelCol='indie')
# model = lr.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "indie")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='indie')
# print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


# ############### LOGISTIC REGRESSION - INSTRUMENTAL #####################
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(instrumental_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# lr = LogisticRegression(featuresCol='features', labelCol='instrumental')
# model = lr.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "instrumental")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='instrumental')
# print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

# ############### LOGISTIC REGRESSION - JAZZ #####################
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(jazz_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# lr = LogisticRegression(featuresCol='features', labelCol='jazz')
# model = lr.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "jazz")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='jazz')
# print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

# ############### LOGISTIC REGRESSION - METAL #####################
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(metal_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# lr = LogisticRegression(featuresCol='features', labelCol='metal')
# model = lr.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "metal")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='metal')
# print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

# ############### LOGISTIC REGRESSION - POP #####################
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(pop_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# lr = LogisticRegression(featuresCol='features', labelCol='pop')
# model = lr.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "pop")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='pop')
# print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


# ############### LOGISTIC REGRESSION - ROCK #####################
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# data_rock = data.select(rock_column)
# train, validate, test = data_rock.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# lr = LogisticRegression(featuresCol='features', labelCol='rock')
# model = lr.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "rock")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='rock')
# print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


# ############### LOGISTIC REGRESSION - SOUL #####################
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(soul_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# lr = LogisticRegression(featuresCol='features', labelCol='soul')
# model = lr.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "soul")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='soul')
# print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))















############### MULTILAYER PERCEPTION - ROCK #####################
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(rock_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
layers = [64, 32, 16, 8]
mlp = MultilayerPerceptronClassifier(layers=layers, seed=42, featuresCol='features', labelCol='rock')
model = mlp.fit(train)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "rock")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='rock')
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
