"""
Pham Nguyen Hoang Dung - s2845016
Silvi Fitria - s2800209

Run time: time spark-submit IINDEX-s2845016-s2800209-RVSSTR.py > logfile.txt 2>&1 /dev/null
real	0m9.605s
user	0m14.651s
sys	    0m1.428s
"""

from pyspark import SparkContext
# please install regex package if needed: pip install regex
import regex as re

sc = SparkContext(appName="IINDEX")
sc.setLogLevel("ERROR")

def remove_punctuation(text):
    return re.sub(ur"\p{P}+", "", text)

def separate((path, contents)):
    return [(content, path) for content in contents]

def to_list(a):
    return [a]

def append(a, b):
    a.append(b)
    return a

def extend(a, b):
    a.extend(b)
    return a

# read the document
rdd = sc.wholeTextFiles("/data/doina/Gutenberg-EBooks")
# split the contents of file into words (after convert to lower case and remove punctuation)
rdd2 = rdd.map(lambda (path, contents): (path, remove_punctuation(contents).lower().split()))
# seperate. i.e: {(key, [value1, value2])} --> {(value1, key), (value2, key)}
rdd3 = rdd2.flatMap(separate)
# combine records with same key. i.e: {(key, value1), (key, value2)} --> {(key, [value1, value2])}
final_result = rdd3.combineByKey(to_list, append, extend).map(lambda (key, value): (key, list(set(value))))
# filter records with lengths of value at least 13
rdd5 = final_result.filter(lambda (key, value): len(value) >= 13)
# sort rdd by key, then return list of key
rdd6 = rdd5.sortByKey().keys()
# merge and print
' '.join(rdd6.take(1000))

