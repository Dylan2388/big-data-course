from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

bdf = sc.parallelize([
    Row(label=1.0, weight=1.0, features=Vectors.dense(0.0, 5.0)),
    Row(label=0.0, weight=2.0, features=Vectors.dense(1.0, 2.0)),
    Row(label=1.0, weight=3.0, features=Vectors.dense(2.0, 1.0)),
    Row(label=0.0, weight=4.0, features=Vectors.dense(3.0, 3.0))]).toDF()
blor = LogisticRegression(regParam=0.01, weightCol="weight")
blorModel = blor.fit(bdf)
blorModel.coefficients
blorModel.intercept
data_path = "mlmodel-test/sample_multiclass_classification_data.txt"
mdf = spark.read.format("libsvm").load(data_path)
mlor = LogisticRegression(regParam=0.1, elasticNetParam=1.0, family="multinomial")
mlorModel = mlor.fit(mdf)
mlorModel.coefficientMatrix

mlorModel.interceptVector
test0 = sc.parallelize([Row(features=Vectors.dense(-1.0, 1.0))]).toDF()
result = blorModel.transform(test0).head()
result.prediction
result.probability
result.rawPrediction
test1 = sc.parallelize([Row(features=Vectors.sparse(2, [0], [1.0]))]).toDF()
blorModel.transform(test1).head().prediction
blor.setParams("vector")
lr_path = temp_path + "/lr"
blor.save(lr_path)
lr2 = LogisticRegression.load(lr_path)
lr2.getRegParam()
model_path = temp_path + "/lr_model"
blorModel.save(model_path)
model2 = LogisticRegressionModel.load(model_path)
blorModel.coefficients[0] == model2.coefficients[0]
blorModel.intercept == model2.intercept
model2