"""
Group 17
Pham Nguyen Hoang Dung - s2845016
Silvi Fitria - s2800209
Run time: time spark-submit KNN-s2845016-s2800209-RSTSTR.py > logfile.txt 2>&1 /dev/null
real	0m22.837s
user	0m15.786s
sys  	0m1.425s
"""

# Import packages
from pyspark import SparkContext

sc = SparkContext(appName="KNN algorithm")
sc.setLogLevel("ERROR")

# Read the text file
rdd = sc.textFile("/data/doina/xyvalue.csv")
# Split the text files and convert to float data type
data = rdd.map(lambda i : i.split(",")).map(lambda df : [float(x) for x in df])
# Define point and k-values
point = (100,100)
k = 100
# Calculate the Euclidean distance between two vectors
data2 = data.map(lambda (i,j,k) : ((((i-point[0])**2) + ((j-point[1])**2))**0.5, k)).sortByKey()
# Make a prediction with neighbors
pred = data2.values().take(k)
predict = sum(pred)/k
print("prediction:", predict)
