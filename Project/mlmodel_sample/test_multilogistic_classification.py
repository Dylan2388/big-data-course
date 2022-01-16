from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression

bdf = sc.parallelize([
    Row(label=1.0, weight=1.0, features=Vectors.dense(0.0, 5.0)),
    Row(label=0.0, weight=2.0, features=Vectors.dense(1.0, 2.0)),
    Row(label=1.0, weight=3.0, features=Vectors.dense(2.0, 1.0)),
    Row(label=0.0, weight=4.0, features=Vectors.dense(3.0, 3.0))]).toDF()
test = sc.parallelize([Row(features=Vectors.dense(-1.0, 1.0))]).toDF()

######## LOGISTIC REGRESSION ##############
blor = LogisticRegression(regParam=0.3, elasticNetParam=0.8)
# Fit the model
blorModel = blor.fit(bdf)
trainingSummary = blorModel.summary

# Test model
result = blorModel.transform(test0)
result.prediction
result.probability
result.rawPrediction
result.features


from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


df = spark.createDataFrame([
    (0.0, Vectors.dense([0.0, 0.0])),
    (1.0, Vectors.dense([0.0, 1.0])),
    (1.0, Vectors.dense([1.0, 0.0])),
    (0.0, Vectors.dense([1.0, 1.0]))], ["label", "features"])

testDF = spark.createDataFrame([
    (Vectors.dense([1.0, 0.0]),),
    (Vectors.dense([0.0, 0.0]),)], ["features"])

####### Multi-layer Perceptron ##########3
layers = [4, 5, 4, 3]
# create the trainer and set its parameters
mlp = MultilayerPerceptronClassifier(maxIter=100, layers=[2, 2, 2], blockSize=1, seed=123)
# train the model
model = mlp.fit(df)

# test model
model.transform(testDF).select("features", "prediction", "probability").show(truncate=False)

# compute accuracy on the test set
result = model.transform(testDF)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))