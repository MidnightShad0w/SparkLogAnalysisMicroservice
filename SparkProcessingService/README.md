# Log Anomaly Detection Project

## Description

Проект для обнаружения аномалий в логах с использованием BERT + AutoEncoder (unsupervised). 
- **train_script.py**: обучает автоэнкодер на логах, сохраняет `autoencoder.pt` + `threshold.txt`.
- **spark_model_service.py**: при запуске, загружает модель из MinIO и обрабатывает новый CSV, выдавая аномалии.
- **app.py**: Flask-приложение, экспортирует REST endpoint `/spark/api/process` для вызова process_file.

## Usage

...

## Project Structure

- **src/ml_model.py**: классы (BertEncoder, AutoEncoder) и функции (train_autoencoder, compute_reconstruction_errors).
- **src/train_script.py**: скрипт обучения (Spark чтение CSV, подготовка данных, обучение AE).
- **src/spark_model_service.py**: класс SparkModelService, загружает модель из MinIO, обрабатывает новые логи.
- **src/app.py**: Flask-приложение (REST).
- **model**: куда сохраняются `autoencoder.pt` и `threshold.txt`.
- **output**: где можно сохранять аномальные логи (или использовать S3/minio).

