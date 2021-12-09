"""
Group 17
Pham Nguyen Hoang Dung - s2845016
Silvi Fitria - s2800209

Runs with Python3:
pyspark --conf "spark.pyspark.python=/usr/bin/python3.6" --conf "spark.pyspark.driver.python=/usr/bin/python3.6"

Run time: time spark-submit --conf "spark.pyspark.python=/usr/bin/python3.6" --conf "spark.pyspark.driver.python=/usr/bin/python3.6" BENCH3-s2845016-s2800209-BENFSD.py > logfile.txt 2>&1 /dev/null

Variable of interest:
1. How much data you put in? (MB, or number of files)
2. How much compute you use? (number of executors) (2, 5, 10)
3. Runtime of the program? (Dash board: http://ctit048.ewi.utwente.nl:8088/cluster) 

Fix (1) or (2)
Vary the other
Measure (3)

Requirements:
1. At least 5 measurement times
2. Increase the data size or executors (big data size). What size match the number of executor?

Output: A graphical plot showing 2 variables of either
1. 


"""

