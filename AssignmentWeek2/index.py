from pyspark import SparkContext

sc = SparkContext(appName="IINDEX")
sc.setLogLevel("ERROR")


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

rdd = sc.wholeTextFiles("/data/doina/Gutenberg-EBooks")
rdd2 = rdd.map(lambda (path, contents): (path, contents.lower().split()))
rdd3 = rdd2.flatMap(separate)
rdd4 = rdd3.combineByKey(to_list, append, extend).map(lambda (key, value): (key, set(value)))
rdd5 = rdd4.filter(lambda (key, value): len(value) >= 13)
rdd6 = rdd5.sortByKey().keys()
rdd6.take(100)