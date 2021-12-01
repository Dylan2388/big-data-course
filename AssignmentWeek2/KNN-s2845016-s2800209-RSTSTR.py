"""
Group 17
Pham Nguyen Hoang Dung - s2845016
Silvi Fitria - s2800209

Run time: time spark-submit KNN-s2845016-s2800209-RSTSTR.py > logfile.txt 2>&1 /dev/null
real	0m1.956s
user	0m3.513s
sys	    0m0.399s

Notation:
n - number of data points
p - number of input points

Algorithm step:
Step 1: Get all training data from the cluster
Step 2: Compute the distance of input points with training data - O(n)
Step 3: Sort training data points base on distance - O(n * log n)
Step 4: Take the first k. Compute mean value

Overall, time complexity for the algorithm is O(n + n*logn) -> O(n * log n) for 1 input.
For each input, we have to recompute Step 2 and 3, which make the time complexity O(p * n * log n).

"""

# Import packages
from pyspark import SparkContext
from math import sqrt
import numpy as np

sc = SparkContext(appName="KNN algorithm")
sc.setLogLevel("ERROR")

# Read a single text file
rdd = sc.wholeTextFiles("/data/doina/xyvalue.csv")
data = [map(float, i.split(",")) for i in rdd.take(100)[0][1].split("\n")[:-1]]

# Calculate the Euclidean distance between two vectors
def euclidean_distance(x, y):
    distance = 0.0
    for i in range(2):
        distance += (x[i] - y[i])**2
    return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row[:2])
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Make a prediction with neighbors
def predict(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = np.mean(output_values)
    return prediction

prediction = predict(data, (100,100), 100)
print(prediction)
