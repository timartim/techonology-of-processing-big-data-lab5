from pyspark.sql import SparkSession
from app_config import SparkConfig


class SparkManager:
    def __init__(self, cfg: SparkConfig):
        self.cfg = cfg

    def create_session(self) -> SparkSession:
        spark = (
            SparkSession.builder
            .appName(self.cfg.app_name)
            .master(self.cfg.master)
            .config("spark.driver.memory", self.cfg.driver_memory)
            .config("spark.sql.shuffle.partitions", self.cfg.shuffle_partitions)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel(self.cfg.log_level)
        return spark