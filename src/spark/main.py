import argparse

from app_config import load_config
from service import FoodClusterService


def build_parser():
    parser = argparse.ArgumentParser(description="CLI для кластеризации продуктов на PySpark")
    parser.add_argument("--config", default=None, help="Путь до config.json")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Обучить модель и сохранить артефакты")

    predict_parser = subparsers.add_parser("predict", help="Загрузить модель и сделать предсказание")
    predict_parser.add_argument("--model-path", required=True, help="Путь до папки сохраненной модели")
    predict_parser.add_argument("--input", default=None, help="Путь до parquet для предсказания")
    predict_parser.add_argument("--output", default=None, help="Куда сохранить CSV с предсказаниями")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    service = FoodClusterService(cfg)

    if args.command == "train":
        service.train()
    elif args.command == "predict":
        service.predict(
            model_path=args.model_path,
            input_path=args.input,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()