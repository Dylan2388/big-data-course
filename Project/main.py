'''
time spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=10 --executor-memory 6G  main.py > logfile_main.txt 2>&1 /dev/null
'''
# import fractions
from random import seed
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



############### LOGISTIC REGRESSION - ALTENATIVE #####################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(alternative_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
lr = LogisticRegression(featuresCol='features', labelCol='alternative')
model = lr.fit(train)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "alternative")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='alternative')
print(" LOGISTIC REGRESSION - ALTENATIVE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### LOGISTIC REGRESSION - DANCE #####################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(dance_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Training
lr = LogisticRegression(featuresCol='features', labelCol='dance')
model = lr.fit(train)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "dance")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='dance')
print("LOGISTIC REGRESSION - DANCE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### LOGISTIC REGRESSION - ELECTRONIC #####################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(electronic_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Training
lr = LogisticRegression(featuresCol='features', labelCol='electronic')
model = lr.fit(train)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "electronic")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='electronic')
print("LOGISTIC REGRESSION - ELECTRONIC Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### LOGISTIC REGRESSION - INDIE #####################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(indie_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
lr = LogisticRegression(featuresCol='features', labelCol='indie')
model = lr.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "indie")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='indie')
print("LOGISTIC REGRESSION - INDIE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### LOGISTIC REGRESSION - INSTRUMENTAL #####################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(instrumental_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
lr = LogisticRegression(featuresCol='features', labelCol='instrumental')
model = lr.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "instrumental")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='instrumental')
print("LOGISTIC REGRESSION - INSTRUMENTAL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

############### LOGISTIC REGRESSION - JAZZ #####################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(jazz_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
lr = LogisticRegression(featuresCol='features', labelCol='jazz')
model = lr.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "jazz")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='jazz')
print("LOGISTIC REGRESSION - JAZZ Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

############### LOGISTIC REGRESSION - METAL #####################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(metal_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
lr = LogisticRegression(featuresCol='features', labelCol='metal')
model = lr.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "metal")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='metal')
print("LOGISTIC REGRESSION - METAL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

############## LOGISTIC REGRESSION - POP #####################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(pop_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
train1 = train.filter(train['pop'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['pop'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Training
lr = LogisticRegression(featuresCol='features', labelCol='pop')
model = lr.fit(train)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "pop")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='pop')
print("LOGISTIC REGRESSION - POP Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### LOGISTIC REGRESSION - ROCK #####################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
data_rock = data.select(rock_column)
train, validate, test = data_rock.randomSplit([0.7, 0.2, 0.1], seed=42)
train1 = train.filter(train['rock'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['rock'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Training
lr = LogisticRegression(featuresCol='features', labelCol='rock')
model = lr.fit(train)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "rock")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='rock')
print("LOGISTIC REGRESSION - ROCK Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### LOGISTIC REGRESSION - SOUL #####################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(soul_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
lr = LogisticRegression(featuresCol='features', labelCol='soul')
model = lr.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "soul")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='soul')
print("LOGISTIC REGRESSION - SOUL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))







############### LINEAR SVC - ALTERNATIVE #####################
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(alternative_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
svm = LinearSVC(maxIter=5, featuresCol='features', labelCol='alternative')
model = svm.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "alternative")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='alternative')
print("LINEAR SVC - ALTERNATIVE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

############### LINEAR SVC - DANCE #####################
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(dance_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
svm = LinearSVC(maxIter=5, featuresCol='features', labelCol='dance')
model = svm.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "dance")
evaluator = MulticlassClassificationEvaluator(metricName="weightedFalsePositiveRate",predictionCol='prediction', labelCol='dance')
print("LINEAR SVC - DANCE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### LINEAR SVC - ELECTRONIC #####################
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(electronic_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
svm = LinearSVC(maxIter=5, featuresCol='features', labelCol='electronic')
model = svm.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "electronic")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='electronic')
print("LINEAR SVC - ELECTRONIC Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### LINEAR SVC - INDIE #####################
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(indie_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
svm = LinearSVC(maxIter=5, featuresCol='features', labelCol='indie')
model = svm.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "indie")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='indie')
print("LINEAR SVC - INDIE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### LINEAR SVC - INSTRUMENTAL #####################
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(instrumental_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
svm = LinearSVC(maxIter=5, featuresCol='features', labelCol='instrumental')
model = svm.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "instrumental")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='instrumental')
print("LINEAR SVC - INSTRUMENTAL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

############### LINEAR SVC - JAZZ #####################
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(jazz_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
svm = LinearSVC(maxIter=5, featuresCol='features', labelCol='jazz')
model = svm.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "jazz")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='jazz')
print("LINEAR SVC - JAZZ Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

############### LINEAR SVC - METAL #####################
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(metal_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
svm = LinearSVC(maxIter=5, featuresCol='features', labelCol='metal')
model = svm.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "metal")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='metal')
print("LINEAR SVC - METAL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

############### LINEAR SVC - POP #####################
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(pop_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
svm = LinearSVC(maxIter=5, featuresCol='features', labelCol='pop')
model = svm.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "pop")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='pop')
print("LINEAR SVC - POP Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### LINEAR SVC - ROCK #####################
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
data_rock = data.select(rock_column)
train, validate, test = data_rock.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
svm = LinearSVC(maxIter=5, featuresCol='features', labelCol='rock')
model = svm.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "rock")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='rock')
print("LINEAR SVC - ROCK Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### LINEAR SVC - SOUL #####################
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(soul_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
svm = LinearSVC(maxIter=5, featuresCol='features', labelCol='soul')
model = svm.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "soul")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='soul')
print("LINEAR SVC - SOUL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))



# ############### NAIVE BAYES - ALTERNATIVE #####################
# from pyspark.ml.classification import NaiveBayes
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(alternative_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# nb = NaiveBayes(featuresCol='features', labelCol='alternative')
# model = nb.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "alternative")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='alternative')
# print("NAIVE BAYES - ALTERNATIVE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


# ############### NAIVE BAYES - DANCE #####################
# from pyspark.ml.classification import NaiveBayes
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(dance_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# nb = NaiveBayes(featuresCol='features', labelCol='dance')
# model = nb.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "dance")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='dance')
# print("NAIVE BAYES - DANCE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

# ############### NAIVE BAYES - ELECTRONIC #####################
# from pyspark.ml.classification import LinearSVC
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(electronic_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# nb = NaiveBayes(featuresCol='features', labelCol='electronic')
# model = nb.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "electronic")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='electronic')
# print("NAIVE BAYES - ELECTRONIC Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


# ############### NAIVE BAYES - INDIE #####################
# from pyspark.ml.classification import LinearSVC
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(indie_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# nb = NaiveBayes(featuresCol='features', labelCol='indie')
# model = nb.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "indie")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='indie')
# print("NAIVE BAYES - INDIE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


# ############### NAIVE BAYES - INSTRUMENTAL #####################
# from pyspark.ml.classification import LinearSVC
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(instrumental_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# nb = NaiveBayes(featuresCol='features', labelCol='instrumental')
# model = nb.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "instrumental")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='instrumental')
# print("NAIVE BAYES - INSTRUMENTAL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

# ############### NAIVE BAYES - JAZZ #####################
# from pyspark.ml.classification import LinearSVC
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(jazz_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# nb = NaiveBayes(featuresCol='features', labelCol='jazz')
# model = nb.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "jazz")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='jazz')
# print("NAIVE BAYES - JAZZ Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

# ############### NAIVE BAYES - METAL #####################
# from pyspark.ml.classification import LinearSVC
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(metal_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# nb = NaiveBayes(featuresCol='features', labelCol='metal')
# model = nb.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "metal")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='metal')
# print("NAIVE BAYES - METAL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

# ############### NAIVE BAYES - POP #####################
# from pyspark.ml.classification import LinearSVC
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(pop_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# nb = NaiveBayes(featuresCol='features', labelCol='pop')
# model = nb.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "pop")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='pop')
# print("NAIVE BAYES - POP Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


# ############### NAIVE BAYES - ROCK #####################
# from pyspark.ml.classification import LinearSVC
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# data_rock = data.select(rock_column)
# train, validate, test = data_rock.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# nb = NaiveBayes(featuresCol='features', labelCol='rock')
# model = nb.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "rock")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='rock')
# print("NAIVE BAYES - ROCK Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


# ############### NAIVE BAYES - SOUL #####################
# from pyspark.ml.classification import LinearSVC
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(soul_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# nb = NaiveBayes(featuresCol='features', labelCol='soul')
# model = nb.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "soul")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='soul')
# print("NAIVE BAYES - SOUL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))



# ############### DECISION TREE - ALTERNATIVE #####################
# from pyspark.ml.classification import DecisionTreeClassifier
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(alternative_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# dt = DecisionTreeClassifier(maxDepth=20, labelCol="alternative", featuresCol='features')
# model = dt.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "alternative")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='alternative')
# print("DECISION TREE - ALTERNATIVE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
# print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
# print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


# ############### DECISION TREE - DANCE #####################
# from pyspark.ml.classification import DecisionTreeClassifier
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(dance_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# dt = DecisionTreeClassifier(maxDepth=20, labelCol="dance", featuresCol='features')
# model = dt.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "dance")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='dance')
# print("DECISION TREE - DANCE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
# print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
# print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

# ############### DECISION TREE - ELECTRONIC #####################
# from pyspark.ml.classification import DecisionTreeClassifier
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(electronic_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# dt = DecisionTreeClassifier(maxDepth=20, labelCol="electronic", featuresCol='features')
# model = dt.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "electronic")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='electronic')
# print("DECISION TREE - ELECTRONIC Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
# print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
# print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


# ############### DECISION TREE - INDIE #####################
# from pyspark.ml.classification import DecisionTreeClassifier
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(indie_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# dt = DecisionTreeClassifier(maxDepth=20, labelCol="indie", featuresCol='features')
# model = dt.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "indie")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='indie')
# print("DECISION TREE - INDIE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
# print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
# print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


# ############### DECISION TREE - INSTRUMENTAL #####################
# from pyspark.ml.classification import DecisionTreeClassifier
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(instrumental_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# dt = DecisionTreeClassifier(maxDepth=20, labelCol="instrumental", featuresCol='features')
# model = dt.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "instrumental")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='instrumental')
# print("DECISION TREE - INSTRUMENTAL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
# print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
# print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

# ############### DECISION TREE - JAZZ #####################
# from pyspark.ml.classification import DecisionTreeClassifier
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(jazz_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# dt = DecisionTreeClassifier(maxDepth=20, labelCol="jazz", featuresCol='features')
# model = dt.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "jazz")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='jazz')
# print("DECISION TREE - JAZZ Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
# print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
# print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

# ############### DECISION TREE - METAL #####################
# from pyspark.ml.classification import DecisionTreeClassifier
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(metal_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# dt = DecisionTreeClassifier(maxDepth=20, labelCol="metal", featuresCol='features')
# model = dt.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "metal")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='metal')
# print("DECISION TREE - METAL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
# print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
# print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

# ############### DECISION TREE - POP #####################
# from pyspark.ml.classification import DecisionTreeClassifier
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(pop_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# dt = DecisionTreeClassifier(maxDepth=20, labelCol="pop", featuresCol='features')
# model = dt.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "pop")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='pop')
# print("DECISION TREE - POP Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
# print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
# print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


# ############### DECISION TREE - ROCK #####################
# from pyspark.ml.classification import DecisionTreeClassifier
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# data_rock = data.select(rock_column)
# train, validate, test = data_rock.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# dt = DecisionTreeClassifier(maxDepth=10, labelCol="rock", featuresCol='features')
# model = dt.fit(train)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "rock")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='rock')
# print("DECISION TREE - ROCK Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
# print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
# print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))




# ############### DECISION TREE - SOUL #####################
# from pyspark.ml.classification import DecisionTreeClassifier
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# ### Data splitting
# filter_data = data.select(soul_column)
# train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
# ### Training
# dt = DecisionTreeClassifier(maxDepth=20, labelCol="soul", featuresCol='features')
# model = dt.fit(train)
# train1 = train.filter(train['electronic'] == 1)
# num1 = float(train1.count())
# train0 = train.filter(train['electronic'] == 0)
# num0 = float(train0.count())
# train0 = train0.sample(fraction=num1/num0, seed=42)
# train = train1.union(train0)
# ### Testing
# result = model.transform(validate)
# predictionAndLabels = result.select("prediction", "soul")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='soul')
# print("DECISION TREE - SOUL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
# print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
# print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))



############### GBTC - ALTERNATIVE #####################
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(alternative_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
dt = GBTClassifier(maxDepth=10, labelCol="alternative", featuresCol='features')
model = dt.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "alternative")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='alternative')
print("GBTC - ALTERNATIVE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### GBTC - DANCE #####################
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(dance_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
dt = GBTClassifier(maxDepth=10, labelCol="dance", featuresCol='features')
model = dt.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "dance")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='dance')
print("GBTC - DANCE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

############### GBTC - ELECTRONIC #####################
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(electronic_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
dt = GBTClassifier(maxDepth=10, labelCol="electronic", featuresCol='features')
model = dt.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "electronic")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='electronic')
print("GBTC - ELECTRONIC Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### GBTC - INDIE #####################
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(indie_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
dt = GBTClassifier(maxDepth=10, labelCol="indie", featuresCol='features')
model = dt.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "indie")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='indie')
print("GBTC - INDIE Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### GBTC - INSTRUMENTAL #####################
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(instrumental_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
dt = GBTClassifier(maxDepth=10, labelCol="instrumental", featuresCol='features')
model = dt.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "instrumental")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='instrumental')
print("GBTC - INSTRUMENTAL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

############### GBTC - JAZZ #####################
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(jazz_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
dt = GBTClassifier(maxDepth=10, labelCol="jazz", featuresCol='features')
model = dt.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "jazz")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='jazz')
print("GBTC - JAZZ Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

############### GBTC - METAL #####################
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(metal_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
dt = GBTClassifier(maxDepth=10, labelCol="metal", featuresCol='features')
model = dt.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "metal")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='metal')
print("GBTC - METAL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

############### GBTC - POP #####################
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(pop_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
dt = GBTClassifier(maxDepth=10, labelCol="pop", featuresCol='features')
model = dt.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "pop")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='pop')
print("GBTC - POP Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


############### GBTC - ROCK #####################
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
data_rock = data.select(rock_column)
train, validate, test = data_rock.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
dt = GBTClassifier(maxDepth=10, labelCol="rock", featuresCol='features')
model = dt.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "rock")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='rock')
print("GBTC - ROCK Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))

############### GBTC - SOUL #####################
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### Data splitting
filter_data = data.select(soul_column)
train, validate, test = filter_data.randomSplit([0.7, 0.2, 0.1], seed=42)
### Training
dt = GBTClassifier(maxDepth=10, labelCol="soul", featuresCol='features')
model = dt.fit(train)
train1 = train.filter(train['electronic'] == 1)
num1 = float(train1.count())
train0 = train.filter(train['electronic'] == 0)
num0 = float(train0.count())
train0 = train0.sample(fraction=num1/num0, seed=42)
train = train1.union(train0)
### Testing
result = model.transform(validate)
predictionAndLabels = result.select("prediction", "soul")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",predictionCol='prediction', labelCol='soul')
print("GBTC - SOUL Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
print("Predicted 1s: " + str(result.filter(result["prediction"]==1).count()))
print("Predicted 0s: " + str(result.filter(result["prediction"]==0).count()))


