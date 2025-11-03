from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .appName('Prediction_Attrition_Client')
         .master('local[*]')
         .getOrCreate())

print(spark.version)
