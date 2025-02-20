package org.danila;

import static org.apache.spark.sql.functions.*;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

public class LogAnomalyDetectionJob {

    public static void main(String[] args) {

//        if (args.length < 2) {
//            System.err.println("Usage: LogAnomalyDetectionJob <input_csv_path> <model_save_s3a_path>");
//            System.err.println("Example: LogAnomalyDetectionJob s3a://bucket-spark/fulldata/logdata.csv s3a://bucket-spark/model");
//            System.exit(1);
//        }
        String logDataPath = "s3a://bucket-spark/fulldata/logdata.csv";      // Например: "s3a://bucket-spark/fulldata/logdata.csv"
        String modelSavePath = "s3a://bucket-spark/model";    // Например: "s3a://bucket-spark/model"

        SparkSession spark = SparkSession.builder()
                .appName("Log Anomaly Detection Training")
                .master("local[*]")
                .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
                .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
                .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .getOrCreate();

        // Читаем CSV с логами
        Dataset<Row> logs = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(logDataPath);

        // Преобразуем TimeTaken: удаляем "ms" и приводим к double
        Dataset<Row> logsWithTime = logs.withColumn(
                "TimeTakenNumeric",
                regexp_replace(col("TimeTaken"), "ms", "").cast("double")
        );

        // Преобразуем LogLevel в числовой индекс
        StringIndexer logLevelIndexer = new StringIndexer()
                .setInputCol("LogLevel")
                .setOutputCol("LogLevelIndex");
        Dataset<Row> indexedLogs = logLevelIndexer.fit(logsWithTime).transform(logsWithTime);

        // Собираем признаки
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"TimeTakenNumeric", "LogLevelIndex"})
                .setOutputCol("features");

        Dataset<Row> featureData = assembler.transform(indexedLogs)
                .select("Timestamp", "LogLevel", "Service", "Message", "RequestID", "User", "ClientIP", "TimeTaken", "features");

        // Разбиваем данные на обучающую, тестовую и валидационную выборки
        Dataset<Row>[] splits = featureData.randomSplit(new double[]{0.7, 0.15, 0.15}, 12345L);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];
        Dataset<Row> validationData = splits[2];

        // Обучаем модель KMeans
        int k = 3;
        KMeans kmeans = new KMeans()
                .setK(k)
                .setSeed(1L)
                .setFeaturesCol("features")
                .setPredictionCol("prediction");
        KMeansModel model = kmeans.fit(trainingData);

        // Оценка модели
        Dataset<Row> trainPredictions = model.transform(trainingData);
        ClusteringEvaluator evaluator = new ClusteringEvaluator();
        double silhouette = evaluator.evaluate(trainPredictions);
        System.out.println("Silhouette score (training): " + silhouette);

        // Сохраняем модель в S3a‑хранилище
        try {
            model.save(modelSavePath);
            System.out.println("Model saved to: " + modelSavePath);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Применяем модель к тестовому набору
        Dataset<Row> testPredictions = model.transform(testData);
        final Vector[] centers = model.clusterCenters();

        spark.udf().register("computeDistance", (Vector features, Integer prediction) -> {
            Vector center = centers[prediction];
            double sqDist = Vectors.sqdist(features, center);
            return Math.sqrt(sqDist);
        }, DataTypes.DoubleType);

        Dataset<Row> testWithDistance = testPredictions.withColumn("distance",
                callUDF("computeDistance", col("features"), col("prediction")));

        Dataset<Row> trainWithDistance = trainPredictions.withColumn("distance",
                callUDF("computeDistance", col("features"), col("prediction")));
        double[] quantiles = trainWithDistance.stat().approxQuantile("distance", new double[]{0.90}, 0.0);
        double threshold = quantiles[0];
        System.out.println("Anomaly threshold (90th percentile): " + threshold);

        // Фильтруем аномалии
        Dataset<Row> anomalies = testWithDistance.filter(col("distance").gt(threshold));
        System.out.println("Anomalous log rows from test data:");
        anomalies.show(20, false);

        Dataset<Row> anomaliesOutput = anomalies.withColumn("features", col("features").cast("string"));
        // Сохраняем результаты в S3a, например:
        anomaliesOutput.coalesce(1)
                .write()
                .option("header", "true")
                .csv("s3a://bucket-spark/output/anomaly-logdata");

        System.out.println("Spark UI доступна (если локально) по адресу http://localhost:4040");
        System.out.println("Нажмите ENTER для завершения программы.");
        new java.util.Scanner(System.in).nextLine();
        spark.stop();
    }
}
