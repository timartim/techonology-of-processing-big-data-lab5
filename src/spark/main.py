from pathlib import Path
import json
import shutil
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    DecimalType,
    ArrayType,
    StructType,
)
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


NUMERIC_TYPES = (DoubleType, FloatType, IntegerType, LongType, ShortType, DecimalType)


def load_config():
    config_path = Path(__file__).resolve().parent / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_spark(cfg):
    spark = (
        SparkSession.builder
        .appName(cfg["spark"]["app_name"])
        .master(cfg["spark"]["master"])
        .config("spark.driver.memory", cfg["spark"]["driver_memory"])
        .config("spark.sql.shuffle.partitions", cfg["spark"]["shuffle_partitions"])
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(cfg["spark"]["log_level"])
    return spark


def build_product_name_columns(df):
    if "product_name" not in df.columns:
        return [], []

    dtype = df.schema["product_name"].dataType

    if isinstance(dtype, ArrayType) and isinstance(dtype.elementType, StructType):
        element_fields = {field.name for field in dtype.elementType.fields}
        exprs = []
        names = []

        if "lang" in element_fields:
            exprs.append(
                F.concat_ws(" | ", F.expr("transform(product_name, x -> x.lang)")).alias("product_name_langs")
            )
            names.append("product_name_langs")

        if "text" in element_fields:
            exprs.append(
                F.concat_ws(" | ", F.expr("transform(product_name, x -> x.text)")).alias("product_name_texts")
            )
            names.append("product_name_texts")

        return exprs, names

    if isinstance(dtype, StructType):
        exprs = []
        names = []
        for field in dtype.fields:
            exprs.append(F.col(f"product_name.{field.name}").cast("string").alias(f"product_name_{field.name}"))
            names.append(f"product_name_{field.name}")
        return exprs, names

    return [F.col("product_name").cast("string").alias("product_name")], ["product_name"]


def write_single_csv(df, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_file.with_suffix("")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if output_file.exists():
        output_file.unlink()

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(tmp_dir))

    part_file = next(tmp_dir.glob("part-*.csv"))
    shutil.move(str(part_file), str(output_file))
    shutil.rmtree(tmp_dir)


def main():
    cfg = load_config()
    spark = build_spark(cfg)

    data_cfg = cfg["data"]
    prep_cfg = cfg["preprocessing"]
    train_cfg = cfg["training"]

    input_path = data_cfg["input_path"]
    clusters_csv_path = Path(data_cfg["clusters_csv_path"])
    profiles_csv_path = Path(data_cfg["profiles_csv_path"])
    centers_csv_path = Path(data_cfg["centers_csv_path"])
    metrics_json_path = Path(data_cfg["metrics_json_path"])

    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)

    model_root = Path(data_cfg["model_root"])
    model_root.mkdir(parents=True, exist_ok=True)

    kmeans_model_path = model_root / "kmeans_model"
    imputer_model_path = model_root / "imputer_model"
    scaler_model_path = model_root / "scaler_model"
    model_info_path = model_root / "model_info.json"

    raw = spark.read.parquet(input_path)

    total_rows = raw.count()
    if total_rows == 0:
        raise ValueError("Входной parquet пустой")

    numeric_cols = [
        field.name
        for field in raw.schema.fields
        if isinstance(field.dataType, NUMERIC_TYPES)
    ]

    if not numeric_cols:
        raise ValueError("Не найдено числовых top-level колонок")

    non_null_counts = raw.agg(
        *[F.count(F.col(c)).alias(c) for c in numeric_cols]
    ).first().asDict()

    min_non_null_ratio = prep_cfg["min_non_null_ratio"]
    feature_cols = [
        c for c in numeric_cols
        if non_null_counts[c] / total_rows >= min_non_null_ratio
    ]

    if not feature_cols:
        raise ValueError("Нет числовых колонок с достаточной долей заполненности")

    product_exprs, product_col_names = build_product_name_columns(raw)

    select_exprs = []
    select_exprs.extend(product_exprs)
    select_exprs.extend(F.col(c).cast("double").alias(c) for c in feature_cols)

    df = raw.select(*select_exprs)

    std_map = df.agg(
        *[F.stddev_samp(F.col(c)).alias(c) for c in feature_cols]
    ).first().asDict()

    feature_cols = [
        c for c in feature_cols
        if std_map[c] is not None and std_map[c] > 0
    ]

    if not feature_cols:
        raise ValueError("После удаления константных признаков не осталось колонок")

    keep_cols = product_col_names + feature_cols
    df = df.select(*keep_cols).dropDuplicates().cache()

    total_n = df.count()
    target_n = prep_cfg["target_n"]

    if total_n > target_n:
        fraction = target_n / total_n
        df = df.sample(withReplacement=False, fraction=fraction, seed=train_cfg["seed"]).limit(target_n)

    working_n = df.count()

    if working_n < 10:
        raise ValueError("Слишком мало строк после предобработки")

    imputed_cols = [f"{c}_imp" for c in feature_cols]

    imputer = Imputer(
        inputCols=feature_cols,
        outputCols=imputed_cols,
        strategy=prep_cfg["imputer_strategy"]
    )

    imputer_model = imputer.fit(df)
    df_imp = imputer_model.transform(df)

    assembler = VectorAssembler(
        inputCols=imputed_cols,
        outputCol="features_raw"
    )

    assembled = assembler.transform(df_imp)

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,
        withStd=True
    )

    scaler_model = scaler.fit(assembled)
    prepared = scaler_model.transform(assembled).cache()

    prepared_n = prepared.count()
    max_k = min(train_cfg["k_max"], prepared_n - 1)
    min_k = train_cfg["k_min"]

    if max_k < min_k:
        raise ValueError("Недостаточно строк для кластеризации")

    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="prediction",
        metricName=train_cfg["metric_name"],
        distanceMeasure=train_cfg["distance_measure"]
    )

    best_k = None
    best_score = -1.0
    best_model = None
    best_predictions = None

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(
            featuresCol="features",
            predictionCol="prediction",
            k=k,
            seed=train_cfg["seed"]
        )
        model = kmeans.fit(prepared)
        predictions = model.transform(prepared)
        score = evaluator.evaluate(predictions)
        print(f"k={k}, silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
            best_predictions = predictions

    cols_to_save = product_col_names + feature_cols + ["prediction"]
    clusters_df = best_predictions.select(*cols_to_save)

    agg_exprs = [F.count("*").alias("n")]
    agg_exprs.extend(F.round(F.avg(c), 4).alias(c) for c in feature_cols)

    profiles_df = best_predictions.groupBy("prediction").agg(
        *agg_exprs
    ).orderBy("prediction")

    means = np.array(scaler_model.mean.toArray())
    stds = np.array(scaler_model.std.toArray())

    centers_rows = []
    for i, center in enumerate(best_model.clusterCenters()):
        center_scaled = np.array(center)
        center_original = center_scaled * stds + means
        row = {"prediction": int(i)}
        for col_name, value in zip(feature_cols, center_original):
            row[col_name] = float(value)
        centers_rows.append(row)

    centers_df = spark.createDataFrame(centers_rows).orderBy("prediction")

    write_single_csv(clusters_df, clusters_csv_path)
    write_single_csv(profiles_df, profiles_csv_path)
    write_single_csv(centers_df, centers_csv_path)

    metrics = {
        "best_k": int(best_k),
        "best_silhouette": float(best_score),
        "rows_total": int(total_rows),
        "rows_working": int(working_n),
        "features_count": int(len(feature_cols)),
        "features": feature_cols,
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    best_model.write().overwrite().save(str(kmeans_model_path))
    imputer_model.write().overwrite().save(str(imputer_model_path))
    scaler_model.write().overwrite().save(str(scaler_model_path))

    model_info = {
        "model_type": "pyspark.ml.clustering.KMeansModel",
        "best_k": int(best_k),
        "best_silhouette": float(best_score),
        "feature_cols": feature_cols,
        "imputed_cols": imputed_cols,
        "product_cols": product_col_names,
        "input_path": input_path,
        "artifacts": {
            "kmeans_model": str(kmeans_model_path),
            "imputer_model": str(imputer_model_path),
            "scaler_model": str(scaler_model_path),
        },
    }

    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"Сохранен файл: {clusters_csv_path}")
    print(f"Сохранен файл: {profiles_csv_path}")
    print(f"Сохранен файл: {centers_csv_path}")
    print(f"Сохранен файл: {metrics_json_path}")
    print(f"Сохранена модель: {kmeans_model_path}")
    print(f"Сохранена модель imputera: {imputer_model_path}")
    print(f"Сохранена модель scaler: {scaler_model_path}")
    print(f"Сохранен файл: {model_info_path}")

    spark.stop()


if __name__ == "__main__":
    main()