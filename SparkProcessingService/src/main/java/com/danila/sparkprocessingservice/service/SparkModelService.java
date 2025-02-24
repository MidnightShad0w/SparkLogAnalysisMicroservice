package com.danila.sparkprocessingservice.service;

import org.apache.spark.ml.PipelineModel;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class SparkModelService {

    private final SparkSession spark;

//    @Value("${minio.endpoint}")
//    private String minioEndpoint;
//
//    @Value("${minio.accessKey}")
//    private String minioAccessKey;
//
//    @Value("${minio.secretKey}")
//    private String minioSecretKey;

    @Autowired
    public SparkModelService() {
        this.spark = SparkSession
                .builder()
                .appName("AnomalyDetection")
                .master("local[*]")
                .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
                .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
                .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .getOrCreate();
    }

    public void processFile(String csvPath, String modelPath, String resultPath) {
        // 1) Читаем CSV
        Dataset<Row> inputDf = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(csvPath);

        System.out.println("=== Схема входного CSV ===");
        inputDf.printSchema();

        // 2) Загружаем модель, если это действительно Spark ML PipelineModel
        PipelineModel model = PipelineModel.load(modelPath);

        // 3) Применяем модель
        Dataset<Row> predictions = model.transform(inputDf);

        System.out.println("=== Схема DataFrame после применения модели ===");
        predictions.printSchema();

        // 4) Сохраняем результат в parquet (или csv)
        predictions
                .write()
                .mode(SaveMode.Overwrite)
                .parquet(resultPath);

        System.out.println("Результаты сохранены в: " + resultPath);
    }
}

