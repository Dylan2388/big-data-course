from pyspark import SparkContext
sc = SparkContext(appName="hashtags")
sc.setLogLevel("ERROR")

#Read
lastfm_a = spark.read.json("/user/s2733226/lastfm/lastfm_train/A/*/*/*.json")
lastfm_b = spark.read.json("/user/s2733226/lastfm/lastfm_train/B/*/*/*.json")
lastfm_c = spark.read.json("/user/s2733226/lastfm/lastfm_train/C/*/*/*.json")
lastfm_d = spark.read.json("/user/s2733226/lastfm/lastfm_train/D/*/*/*.json")
lastfm_e = spark.read.json("/user/s2733226/lastfm/lastfm_train/E/*/*/*.json")
lastfm_f = spark.read.json("/user/s2733226/lastfm/lastfm_train/F/*/*/*.json")
lastfm_g = spark.read.json("/user/s2733226/lastfm/lastfm_train/G/*/*/*.json")
lastfm_h = spark.read.json("/user/s2733226/lastfm/lastfm_train/H/*/*/*.json")
lastfm_i = spark.read.json("/user/s2733226/lastfm/lastfm_train/I/*/*/*.json")
lastfm_j = spark.read.json("/user/s2733226/lastfm/lastfm_train/J/*/*/*.json")
lastfm_k = spark.read.json("/user/s2733226/lastfm/lastfm_train/K/*/*/*.json")
lastfm_l = spark.read.json("/user/s2733226/lastfm/lastfm_train/L/*/*/*.json")
lastfm_m = spark.read.json("/user/s2733226/lastfm/lastfm_train/M/*/*/*.json")
lastfm_n = spark.read.json("/user/s2733226/lastfm/lastfm_train/N/*/*/*.json")
lastfm_o = spark.read.json("/user/s2733226/lastfm/lastfm_train/O/*/*/*.json")
lastfm_p = spark.read.json("/user/s2733226/lastfm/lastfm_train/P/*/*/*.json")
lastfm_q = spark.read.json("/user/s2733226/lastfm/lastfm_train/Q/*/*/*.json")
lastfm_r = spark.read.json("/user/s2733226/lastfm/lastfm_train/R/*/*/*.json")
lastfm_s = spark.read.json("/user/s2733226/lastfm/lastfm_train/S/*/*/*.json")
lastfm_t = spark.read.json("/user/s2733226/lastfm/lastfm_train/T/*/*/*.json")
lastfm_u = spark.read.json("/user/s2733226/lastfm/lastfm_train/U/*/*/*.json")
lastfm_v = spark.read.json("/user/s2733226/lastfm/lastfm_train/V/*/*/*.json")
lastfm_w = spark.read.json("/user/s2733226/lastfm/lastfm_train/W/*/*/*.json")
lastfm_x = spark.read.json("/user/s2733226/lastfm/lastfm_train/X/*/*/*.json")
lastfm_y = spark.read.json("/user/s2733226/lastfm/lastfm_train/Y/*/*/*.json")
lastfm_z = spark.read.json("/user/s2733226/lastfm/lastfm_train/Z/*/*/*.json")

#All DFs
lastfm = [lastfm_a, lastfm_b, lastfm_c, lastfm_d, lastfm_e, lastfm_f, lastfm_g, lastfm_h, lastfm_i, lastfm_j, lastfm_k, lastfm_l, lastfm_m, lastfm_n, lastfm_o, lastfm_p, lastfm_q, lastfm_r, lastfm_s, lastfm_t, lastfm_u, lastfm_v, lastfm_w, lastfm_x, lastfm_y, lastfm_z]

#Union
df = reduce(DataFrame.unionAll, lastfm)

#Write
df.write.format('json').save('/user/s2733226/project/lastfm_data.json')
