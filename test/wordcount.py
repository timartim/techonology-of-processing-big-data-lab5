from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("WordCount")
    .master("local[*]")
    .getOrCreate()
)

sc = spark.sparkContext

counts = (
    sc.textFile("data/wordcount/input.txt")
      .flatMap(lambda line: line.split())
      .map(lambda word: (word, 1))
      .reduceByKey(lambda a, b: a + b)
      .sortBy(lambda x: x[0])
)

for word, count in counts.collect():
    print(f"{word}: {count}")

spark.stop()