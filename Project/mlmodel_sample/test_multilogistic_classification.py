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
result = blorModel.transform(test)
result.prediction
result.probability
result.rawPrediction
result.features

# Export result

########### Multilayer Perceptron Classifier #################

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
mlp = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=1, seed=123)
# train the model
model = mlp.fit(df)

# test model
model.transform(testDF).select("features", "prediction", "probability").show(truncate=False)

# compute accuracy on the test set
result = model.transform(testDF)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


################ Linear SVC #################
from pyspark.ml.classification import LinearSVC

### Data
df = sc.parallelize([
    Row(label=1.0, features=Vectors.dense(1.0, 1.0, 1.0)),
    Row(label=0.0, features=Vectors.dense(1.0, 2.0, 3.0))]).toDF()
test = sc.parallelize([Row(features=Vectors.dense(-1.0, -1.0, -1.0))]).toDF()

### Train
svm = LinearSVC(maxIter=5, regParam=0.01)
model = svm.fit(df)

### Test
result = model.transform(test).show(truncate=False)


############ Naive Bayes #################
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

nb = NaiveBayes(smoothing=1.0)
model = nb.fit(df)

result = model.transform(test).show(truncate=False)



############# Decision Tree Classifier ###############
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer

df = spark.createDataFrame([
    (1.0, Vectors.dense(1.0)),
    (0.0, Vectors.sparse(1, [], []))], ["label", "features"])
test = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
### Train Model ###
stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
si_model = stringIndexer.fit(df)
td = si_model.transform(df)
dt = DecisionTreeClassifier(maxDepth=2, labelCol="indexed")
model = dt.fit(td)

### Test Model ###
result = model.transform(test).show(truncate=False)


########### Random Forrest Classifier ###########
rf = RandomForestClassifier(maxDepth=2, labelCol="indexed")
model = rf.fit(td)
### Test Model ###
result = model.transform(test).show(truncate=False)



########## Gradient-boosted tree classifier #########
gc = GBTClassifier(maxDepth=2, labelCol="indexed")
model = gc.fit(td)
### Test Model ###
result = model.transform(test).show(truncate=False)

