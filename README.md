# **<p align="center">Log anomaly detection</p>**

## **About the project**

Microservice application for log aggregation and anomaly detection.

The application consists of several microservices: **UploadService**, **KafkaConsumerService**, **SparkProcessingService**.

**UploadService**, **KafkaConsumerService** based on `Java` with `Spring` framework

**SparkProcessingService** based on `Python` with `Flask` framework

You can also retrain the model on your own dataset using a [script](https://github.com/MidnightShad0w/SparkLogAnalysisMicroservice/blob/master/SparkProcessingService/src/train_script.py).

## **Project Stack**

![Java](https://img.shields.io/badge/java-%23ED8B00.svg?style=for-the-badge&logo=java&logoColor=white)
![Python](https://img.shields.io/badge/python-%23dbde4f.svg?style=for-the-badge&logo=python&logoColor=%234491c1)

![Spring](https://img.shields.io/badge/spring-%236DB33F.svg?style=for-the-badge&logo=spring&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23d9d9d9.svg?style=for-the-badge&logo=flask&logoColor=%234491c1)
![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-white.svg?style=for-the-badge&logo=apacheairflow&logoColor=%233365ff)
![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-000?style=for-the-badge&logo=apachekafka)
![PyTorch](https://img.shields.io/badge/pytorch-%23000.svg?style=for-the-badge&logo=pytorch&logoColor=%23ff8614)
![Spark](https://img.shields.io/badge/apache%20spark-%234491c1.svg?style=for-the-badge&logo=apachespark&logoColor=%23ff8614)

![Clickhouse](https://img.shields.io/badge/clickhouse-%23000.svg?style=for-the-badge&logo=clickhouse&logoColor=%23fff)
![Minio](https://img.shields.io/badge/minio-%23000.svg?style=for-the-badge&logo=minio&logoColor=%23c62425)

![Grafana](https://img.shields.io/badge/grafana-%23fff.svg?style=for-the-badge&logo=grafana&logoColor=%23ff8614)

----------

## **<p align="center">Links, commands, etc.</p>**

### **Launching API**

1. Launch [docker-compose](https://github.com/MidnightShad0w/SparkLogAnalysisMicroservice/blob/master/docker-compose.yml)
``` bash
docker-compose up
```
2. Go to the [SparkProcessingService](https://github.com/MidnightShad0w/SparkLogAnalysisMicroservice/tree/master/SparkProcessingService) folder and open the console in it. Execute command:
``` bash
docker-compose up -d
```
3. Launch Airflow UI:
``` bash
docker-compose run --rm airflow-webserver airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email <your_email>
```

Now you have Airflow UI at [http://localhost:8080](http://localhost:8080), Minio storage at [http://localhost:9000](http://localhost:9000) (with credentials: minioadmin, minioadmin), Grafana UI at [http://localhost:3000](http://localhost:3000).
