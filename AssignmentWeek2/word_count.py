"""
This computes the most frequent words across the books in /data/doina/Gutenberg-EBooks.

To execute on a machine:
    time spark-submit word_count.py 2> /dev/null

The "time" prefix simply times the execution.

This code works on the default Python2 
   (Python3 doesn't support tuples in lambda expressions, like on line 21).

If you write code for Python3, add 
    --conf "spark.pyspark.python=/usr/bin/python3.6" --conf "spark.pyspark.driver.python=/usr/bin/python3.6"
to the spark-submit command.
"""

from pyspark import SparkContext

sc = SparkContext(appName="Word count")
sc.setLogLevel("ERROR")

rdd = sc.wholeTextFiles("/data/doina/Gutenberg-EBooks")

words = rdd \
    .flatMap(lambda (file, contents): contents.lower().split()) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a+b)

top_words = words.top(40, key=lambda t: t[1])

for (w, c) in top_words:
    print "Word:\t", w, "\t occurrences:\t", c
