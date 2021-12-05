import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
spark = SparkSession.builder.appName('GAMES').getOrCreate()

### Runs with Python3:
### pyspark --conf "spark.pyspark.python=/usr/bin/python3.6" --conf "spark.pyspark.driver.python=/usr/bin/python3.6"
    

df = spark.read.json("/data/doina/UCSD-Amazon-Data/meta_Video_Games.json.gz")
df2 = df.select(F.col('related')['also_bought'].alias('item')).where(F.col('item').isNotNull())
df3 = df2.select(F.explode(df2['item']).alias('item'))
df4 = df3.groupby('item').count()
df5 = df4.sort('count', ascending=False)
record_id = df5.take(1)[0]['item']
record = df.filter(df['asin'] == record_id)



## Step 1: Skim through all data
## Step 2: Count
## Step 3: Count number of product.asin
## Step 4: Find max record


