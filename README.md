# OpenFoodFacts KMeans Clustering on PySpark

Проект для кластеризации продуктов из OpenFoodFacts с помощью `PySpark` и алгоритма `KMeans`.

## Что делает проект

Проект:

- читает данные из `.parquet`
- автоматически выбирает числовые признаки с достаточной заполненностью
- выполняет предобработку данных
- масштабирует признаки
- обучает несколько моделей `KMeans` для разных `k`
- выбирает лучшую модель по метрике `silhouette`
- сохраняет:
  - CSV с кластерами
  - CSV с профилями кластеров
  - CSV с центрами кластеров
  - JSON с метриками
  - артефакты модели в папку `models`

## Основные команды
#### установка зависимостей
```bash
pip install -r requirements.txt
```
#### Установка данных
В папку src/data требуется положить файл с именем food_small.parquet, модель будет обучаться на них. 
Другой путь можно описать в файле src/spark/config.json в поле input_path. Путь до данных должен быть относительным. 

#### Обучение модели
```bash
python src/spark/main.py train
```
После обучения будут сохранены:

- src/artifacts/food_clusters.csv
- src/artifacts/food_cluster_profiles.csv
- src/artifacts/food_cluster_centers.csv
- src/artifacts/food_metrics.json
- src/models/openfoodfacts_kmeans/

#### Предсказание новых данных

```bash
python src/spark/main.py predict \
  --model-path ../models/openfoodfacts_kmeans \
  --input ../data/food_small.parquet \
  --output ../artifacts/predictions.csv
```

Результирующий файл записывается в --output.

## Настройки

Настройки spark и гиперпараметры алгоритма находятся в по пути
```bash
src/spark/config.json 
```

## Требования

- Python 3.10+
- Java 17 или 21
- установленный и работающий PySpark
