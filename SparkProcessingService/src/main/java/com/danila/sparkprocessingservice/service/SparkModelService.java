package com.danila.sparkprocessingservice.service;

import org.apache.spark.ml.PipelineModel;

import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

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
        System.out.println("Путь к файлу внутри processFile: " + csvPath);
        // 1) Читаем CSV
        Dataset<Row> inputDf = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(csvPath);

        System.out.println("=== Схема входного CSV ===");
        inputDf.printSchema();

        PipelineModel model = PipelineModel.load(modelPath);

        Dataset<Row> predictions = model.transform(inputDf);

        System.out.println("=== Схема DataFrame после применения модели ===");
        predictions.printSchema();

        KMeansModel kmeansModel = (KMeansModel) model.stages()[3];
        Vector[] centers = kmeansModel.clusterCenters();

        spark.udf().register("computeDistance", (Vector features, Integer prediction) -> {
            Vector center = centers[prediction];
            double sqDist = Vectors.sqdist(features, center);
            return Math.sqrt(sqDist);
        }, DataTypes.DoubleType);

        Dataset<Row> predictionsWithDistance = predictions.withColumn("distance",
                callUDF("computeDistance", col("features"), col("prediction")));

        double[] quantiles = predictionsWithDistance.stat()
                .approxQuantile("distance", new double[]{0.90}, 0.0);
        double threshold = quantiles[0];
        System.out.println("Anomaly threshold (90th percentile): " + threshold);

        Dataset<Row> anomalies = predictionsWithDistance.filter(col("distance").gt(threshold));
        anomalies.show(20, false);

        anomalies
                .write()
                .mode(SaveMode.Overwrite) //todo: тут надо не перезаписывать
                .parquet(resultPath);

        System.out.println("Результаты сохранены в: " + resultPath);
    }
}

