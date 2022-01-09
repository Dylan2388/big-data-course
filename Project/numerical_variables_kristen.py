# Run command:
# time spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=10 --executor-memory 2G numerical_variables_kristen.py > logfile_numerical_variables_kristen.txt 2>&1 /dev/null

# Import packages and initiate session
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType

spark = SparkSession.builder.getOrCreate()

# Feature: Duration
# Create schema
schema = StructType(\
    [StructField("track_id", StringType(), True),\
     StructField("duration", FloatType(), True)])
# Read the text file
df_duration = spark.read.csv("/user/s2733226/project/column_data/duration/part*.csv", header = 'False', schema = schema)
# Remove duplicates
# with duplicates df_duration.count() = 3.000.000
# without duplicates df_duration.count() = 1.000.000
df_duration = df_duration.distinct()
# Convert duration from float to integer
#df_duration = df_duration.withColumn("duration", df_duration["duration"].cast(IntegerType()))
# Filter out songs which have a duration of 0 second
df_duration = df_duration.filter(df_duration.duration != 0)
# Filter out songs with abnormal durations e.g. songs that last 5 seconds
# duration_rdd = df_duration.select(df_duration.duration).rdd
#ScaledData
#read data as float
#df_duration = spark.read.csv("/user/s2733226/project/column_data/duration/part*.csv", header = 'False', schema = schema)
#Convert to vector
assembler = VectorAssembler().setInputCols(['duration']).setOutputCol('features')
#Standard Scaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
#Scaled Model
scalerModel = scaler.fit(assembler.transform(df_duration))
scaledData = scalerModel.transform(assembler.transform(df_duration))
#Transform vector data tp float
from pyspark.sql import functions as f
from pyspark.sql.types import FloatType
firstelement=f.udf(lambda v:float(v[0]),FloatType())
transform = scaledData.withColumn("features", firstelement("features")).withColumn("scaledFeatures", firstelement("scaledFeatures"))
#Filter data
transform.filter((col('scaledFeatures') > -1.5)).show()


# 3. Energy: already normalized in original data
# Read the text file
df_energy = spark.read.csv("/user/s2733226/project/column_data/duration/part-0*.csv")
# Showing the data
df_energy.show()
# Rename the column names
df_energy = df_energy.selectExpr("_c0 as name", "_c1 as value")
# Convert the data types to integer
df_energy = df_energy.withColumn("value",df_energy["value"].cast(IntegerType()))
# Count the whole data
df_energy.count()
# 3000000
# Count the data that is zero
df_energy.select('name').where(df_energy.value==0).count()
# 255
# Count the data that is not zero
df_energy.select('name').where(df_energy.value!=0).count()
# 2999745

# 4. Loudness
# Read the text file
df_loudness = spark.read.csv("/user/s2733226/project/column_data/loudness/part-0*.csv")
# Rename the column names
df_loudness = df_loudness.selectExpr("_c0 as name", "_c1 as value")
# Showing the data
df_loudness.show()
# Convert the data types to integer
df_loudness = df_loudness.withColumn("value",df_loudness["value"].cast(IntegerType()))
# Count the whole data
df_loudness.count()
# Count the data that is zero
df_loudness.select('name').where(df_loudness.value==0).count()
# 1824
# Count the data that is not zero
df_loudness.select('name').where(df_loudness.value!=0).count()
# 2998176

# 5. Song Hotness
# Read the text file
df_hotness = spark.read.csv("/user/s2733226/project/column_data/song_hotttnesss/part-0*.csv")
# Showing the data
df_hotness.show()
# Rename the column names
df_hotness = df_hotness.selectExpr("_c0 as name", "_c1 as value")
# Convert the data types to integer
df_hotness = df_hotness.withColumn("value",df_hotness["value"].cast(IntegerType()))
# Count the whole data
df_hotness.count()
# Count the data that is zero
df_hotness.select('name').where(df_hotness.value==0).count()
# 1745379
# Count the data that is not zero
df_hotness.select('name').where(df_hotness.value!=0).count()
# 516
# 1255137
# there are nan and 0 data in hotness

# 6. Tempo
# Read the text file
df_tempo = spark.read.csv("/user/s2733226/project/column_data/tempo/part-0*.csv")
# Rename the column names
df_tempo = df_tempo.selectExpr("_c0 as name", "_c1 as value")
# Showing the data
df_tempo.show()
# Convert the data types to integer
df_tempo = df_tempo.withColumn("value",df_tempo["value"].cast(IntegerType()))
# Count the whole data
# 3000000
df_tempo.distinct().count()
# Count the data that is zero
df_tempo.select('name').where(df_tempo.value==0).distinct().count()
# 9690
# Count the data that is not zero
df_tempo.select('name').where(df_tempo.value!=0).count()
# 2990310

# Tempo =  beat/minutes
# There are triple duplicates but it should be 1000000 datasets
