from pathlib import Path
import json
import numpy as np
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import ImputerModel, StandardScalerModel
from pyspark.sql import functions as F

from app_config import AppConfig
from io_utils import write_single_csv, save_json
from preprocessing import FoodPreprocessor
from spark_session import build_spark


class FoodClusterService:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.preprocessor = FoodPreprocessor(
            min_non_null_ratio=cfg.preprocessing.min_non_null_ratio,
            imputer_strategy=cfg.preprocessing.imputer_strategy,
        )

    def _resolve(self, path_str: str) -> Path:
        return (self.cfg.base_dir / path_str).resolve()

    def _model_paths(self, model_root_path: str | None = None):
        root_path_str = model_root_path or self.cfg.data.model_root
        model_root = self._resolve(root_path_str)
        model_root.mkdir(parents=True, exist_ok=True)

        return {
            "root": model_root,
            "kmeans_model": model_root / "kmeans_model",
            "imputer_model": model_root / "imputer_model",
            "scaler_model": model_root / "scaler_model",
            "model_info": model_root / "model_info.json",
        }

    def _select_best_model(self, prepared):
        prepared_n = prepared.count()
        min_k = self.cfg.training.k_min
        max_k = min(self.cfg.training.k_max, prepared_n - 1)

        if max_k < min_k:
            raise ValueError("Недостаточно строк для кластеризации")

        evaluator = ClusteringEvaluator(
            featuresCol="features",
            predictionCol="prediction",
            metricName=self.cfg.training.metric_name,
            distanceMeasure=self.cfg.training.distance_measure,
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
                seed=self.cfg.training.seed,
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

        return best_k, best_score, best_model, best_predictions

    def _build_profiles_df(self, best_predictions, feature_cols):
        agg_exprs = [F.count("*").alias("n")]
        agg_exprs.extend(F.round(F.avg(c), 4).alias(c) for c in feature_cols)

        return best_predictions.groupBy("prediction").agg(
            *agg_exprs
        ).orderBy("prediction")

    def _build_centers_df(self, spark, best_model, scaler_model, feature_cols):
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

        return spark.createDataFrame(centers_rows).orderBy("prediction")

    def train(self):
        spark = build_spark(self.cfg.spark)
        try:
            input_path = self._resolve(self.cfg.data.input_path)
            clusters_csv_path = self._resolve(self.cfg.data.clusters_csv_path)
            profiles_csv_path = self._resolve(self.cfg.data.profiles_csv_path)
            centers_csv_path = self._resolve(self.cfg.data.centers_csv_path)
            metrics_json_path = self._resolve(self.cfg.data.metrics_json_path)

            raw = spark.read.parquet(str(input_path))

            df, feature_cols, product_col_names, total_rows = self.preprocessor.prepare_training_frame(raw)
            df, working_n = self.preprocessor.sample_frame(
                df,
                target_n=self.cfg.preprocessing.target_n,
                seed=self.cfg.training.seed,
            )

            prepared, imputer_model, scaler_model, imputed_cols = self.preprocessor.fit_transform(df, feature_cols)

            best_k, best_score, best_model, best_predictions = self._select_best_model(prepared)

            cols_to_save = product_col_names + feature_cols + ["prediction"]
            clusters_df = best_predictions.select(*cols_to_save)
            profiles_df = self._build_profiles_df(best_predictions, feature_cols)
            centers_df = self._build_centers_df(spark, best_model, scaler_model, feature_cols)

            write_single_csv(clusters_df, clusters_csv_path)
            write_single_csv(profiles_df, profiles_csv_path)
            write_single_csv(centers_df, centers_csv_path)

            metrics = {
                "best_k": int(best_k),
                "best_score": float(best_score),
                "metric_name": self.cfg.training.metric_name,
                "rows_total": int(total_rows),
                "rows_working": int(working_n),
                "features_count": int(len(feature_cols)),
                "features": feature_cols,
            }
            save_json(metrics, metrics_json_path)

            paths = self._model_paths()
            best_model.write().overwrite().save(str(paths["kmeans_model"]))
            imputer_model.write().overwrite().save(str(paths["imputer_model"]))
            scaler_model.write().overwrite().save(str(paths["scaler_model"]))

            model_info = {
                "model_type": "pyspark.ml.clustering.KMeansModel",
                "best_k": int(best_k),
                "best_score": float(best_score),
                "metric_name": self.cfg.training.metric_name,
                "feature_cols": feature_cols,
                "imputed_cols": imputed_cols,
                "product_cols": product_col_names,
                "input_path": str(input_path),
                "artifacts": {
                    "kmeans_model": str(paths["kmeans_model"]),
                    "imputer_model": str(paths["imputer_model"]),
                    "scaler_model": str(paths["scaler_model"]),
                },
            }
            save_json(model_info, paths["model_info"])

            print(f"Сохранен файл: {clusters_csv_path}")
            print(f"Сохранен файл: {profiles_csv_path}")
            print(f"Сохранен файл: {centers_csv_path}")
            print(f"Сохранен файл: {metrics_json_path}")
            print(f"Сохранена модель: {paths['kmeans_model']}")
            print(f"Сохранена модель imputera: {paths['imputer_model']}")
            print(f"Сохранена модель scaler: {paths['scaler_model']}")
            print(f"Сохранен файл: {paths['model_info']}")
        finally:
            spark.stop()

    def predict(self, model_path: str, input_path: str | None = None, output_path: str | None = None):
        spark = build_spark(self.cfg.spark)
        try:
            paths = self._model_paths(model_path)

            kmeans_model = KMeansModel.load(str(paths["kmeans_model"]))
            imputer_model = ImputerModel.load(str(paths["imputer_model"]))
            scaler_model = StandardScalerModel.load(str(paths["scaler_model"]))

            with open(paths["model_info"], "r", encoding="utf-8") as f:
                model_info = json.load(f)

            feature_cols = model_info["feature_cols"]
            product_cols = model_info["product_cols"]

            resolved_input = self._resolve(input_path or self.cfg.data.input_path)
            resolved_output = self._resolve(output_path or self.cfg.data.predictions_csv_path)

            raw = spark.read.parquet(str(resolved_input))

            df = self.preprocessor.prepare_inference_frame(
                raw=raw,
                feature_cols=feature_cols,
                expected_product_cols=product_cols,
            )

            prepared = self.preprocessor.transform_with_models(
                df=df,
                feature_cols=feature_cols,
                imputer_model=imputer_model,
                scaler_model=scaler_model,
            )

            predictions = kmeans_model.transform(prepared)

            cols_to_save = product_cols + feature_cols + ["prediction"]
            predictions_df = predictions.select(*cols_to_save)

            write_single_csv(predictions_df, resolved_output)

            print(f"Сохранен файл с предсказаниями: {resolved_output}")
        finally:
            spark.stop()