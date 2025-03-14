package com.danila.sparkprocessingservice.service;

import com.amazonaws.ClientConfiguration;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.client.builder.AwsClientBuilder;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.amazonaws.services.s3.model.ListObjectsV2Request;
import com.amazonaws.services.s3.model.ListObjectsV2Result;
import com.amazonaws.services.s3.model.ObjectMetadata;
import com.amazonaws.services.s3.model.S3ObjectSummary;
import org.apache.spark.ml.PipelineModel;

import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;

import java.io.ByteArrayInputStream;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

@Service
public class SparkModelService {

    private final SparkSession spark;

    @Value("${minio.bucket}")
    private String bucketName;

    @Value("${minio.endpoint}")
    private String minioEndpoint;

    @Value("${minio.accessKey}")
    private String minioAccessKey;

    @Value("${minio.secretKey}")
    private String minioSecretKey;

    private AmazonS3 s3Client;

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

    @PostConstruct
    public void initS3Client() {
        BasicAWSCredentials credentials = new BasicAWSCredentials(minioAccessKey, minioSecretKey);
        ClientConfiguration clientConfig = new ClientConfiguration();
        clientConfig.setSignerOverride("AWSS3V4SignerType");

        s3Client = AmazonS3ClientBuilder.standard()
                .withEndpointConfiguration(new AwsClientBuilder.EndpointConfiguration(minioEndpoint, "us-east-1"))
                .withPathStyleAccessEnabled(true)
                .withClientConfiguration(clientConfig)
                .withCredentials(new AWSStaticCredentialsProvider(credentials))
                .build();
    }

    public void processFile(String csvPath, String modelPath, String resultPath) {
        System.out.println("Путь к файлу внутри processFile: " + csvPath);
        Dataset<Row> inputDf = null;
        try {
            inputDf = spark.read()
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .csv(csvPath);
        } catch (Exception e) {
            if (e.getMessage().contains("Path does not exist")) {
                System.out.println("Входной путь " + csvPath + " не существует. Возможно, все файлы уже обработаны.");
                return;
            } else {
                throw e;
            }
        }

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
        ensureFolderExists("result-anomaly-logs/");
        anomalies
                .write()
                .mode(SaveMode.Append)
                .parquet(resultPath);

        System.out.println("Результаты сохранены в: " + resultPath);

        deleteProcessedFiles();
    }

    public void deleteProcessedFiles() {
        try {
            ListObjectsV2Request req = new ListObjectsV2Request()
                    .withBucketName(bucketName)
                    .withPrefix("uploads/");
            ListObjectsV2Result result;
            do {
                result = s3Client.listObjectsV2(req);
                for (S3ObjectSummary objectSummary : result.getObjectSummaries()) {
                    s3Client.deleteObject(bucketName, objectSummary.getKey());
                    System.out.println("Deleted: " + objectSummary.getKey());
                }
                req.setContinuationToken(result.getNextContinuationToken());
            } while (result.isTruncated());
        } catch (Exception e) {
            System.err.println("Ошибка при удалении файлов из папки uploads: " + e.getMessage());
        }
    }

    public void ensureFolderExists(String folderKey) {
        if (!s3Client.doesObjectExist(bucketName, folderKey)) {
            ObjectMetadata metadata = new ObjectMetadata();
            metadata.setContentLength(0);
            s3Client.putObject(bucketName, folderKey, new ByteArrayInputStream(new byte[0]), metadata);
            System.out.println("Создана виртуальная папка: " + folderKey);
        }
    }
}

