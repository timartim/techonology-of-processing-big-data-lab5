from pyspark.sql import SparkSession
from app_config import SparkConfig


def build_spark(cfg: SparkConfig) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(cfg.app_name)
        .master(cfg.master)
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.sql.shuffle.partitions", cfg.shuffle_partitions)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(cfg.log_level)
    return spark