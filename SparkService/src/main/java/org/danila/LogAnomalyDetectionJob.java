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

        if (args.length < 2) {
            System.err.println("Usage: LogAnomalyDetectionJob C:\\Users\\admin\\Desktop\\Диплом\\LogAnalysisMicroservice\\SparkService\\data\\logdata.csv C:\\Users\\admin\\Desktop\\Диплом\\LogAnalysisMicroservice\\SparkService\\model");
            System.exit(1);
        }
        String logDataPath = args[0];
        String modelSavePath = args[1];

        SparkSession spark = SparkSession.builder()
                .appName("Log Anomaly Detection Training")
                .master("local[*]")
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

        // Сохраним исходные данные вместе с признаками
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"TimeTakenNumeric", "LogLevelIndex"})
                .setOutputCol("features");

        // Выбираем исходные колонки для дальнейшего анализа
        Dataset<Row> featureData = assembler.transform(indexedLogs)
                .select("Timestamp", "LogLevel", "Service", "Message", "RequestID", "User", "ClientIP", "TimeTaken", "features");

        // Разбиваем данные на обучающую, тестовую и валидационную
        Dataset<Row>[] splits = featureData.randomSplit(new double[]{0.7, 0.15, 0.15}, 12345L);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];
        Dataset<Row> validationData = splits[2];

        // Обучаем модель KMeans на обучающих данных
        int k = 3;
        KMeans kmeans = new KMeans()
                .setK(k)
                .setSeed(1L)
                .setFeaturesCol("features")
                .setPredictionCol("prediction");
        KMeansModel model = kmeans.fit(trainingData);

        // Оценка на обучающих данных
        Dataset<Row> trainPredictions = model.transform(trainingData);
        ClusteringEvaluator evaluator = new ClusteringEvaluator();
        double silhouette = evaluator.evaluate(trainPredictions);
        System.out.println("Silhouette score (training): " + silhouette);

        // Сохраняем модель
        try {
            model.save(modelSavePath);
            System.out.println("Model saved to: " + modelSavePath);
        } catch (Exception e) {
            e.printStackTrace();
        }


        // Применяем модель к тестовому набору
        Dataset<Row> testPredictions = model.transform(testData);

        // Получаем центры кластеров
        final Vector[] centers = model.clusterCenters();

        // Регистрируем UDF для вычисления евклидова расстояния до центра кластера
        spark.udf().register("computeDistance", (Vector features, Integer prediction) -> {
            Vector center = centers[prediction];
            double sqDist = Vectors.sqdist(features, center);
            return Math.sqrt(sqDist);
        }, DataTypes.DoubleType);

        // Добавляем колонку с расстоянием
        Dataset<Row> testWithDistance = testPredictions.withColumn("distance",
                callUDF("computeDistance", col("features"), col("prediction")));

        // Вычисляем порог аномалии на основе 90-го процентиля расстояний в обучающих данных
        Dataset<Row> trainWithDistance = trainPredictions.withColumn("distance",
                callUDF("computeDistance", col("features"), col("prediction")));
        double[] quantiles = trainWithDistance.stat().approxQuantile("distance", new double[]{0.90}, 0.0);
        double threshold = quantiles[0];
        System.out.println("Anomaly threshold (90th percentile): " + threshold);

        // Фильтруем аномальные записи (где distance > threshold)
        Dataset<Row> anomalies = testWithDistance.filter(col("distance").gt(threshold));
        System.out.println("Anomalous log rows from test data:");
        anomalies.show(20, false);

        Dataset<Row> anomaliesOutput = anomalies.withColumn("features", col("features").cast("string"));
        // Запись в файл аномалий
        anomaliesOutput.write()
                .option("header", "true")
                .csv("C:\\Users\\admin\\Desktop\\Диплом\\LogAnalysisMicroservice\\SparkService\\result\\anomaly-logdata");

        System.out.println("Spark UI доступна по адресу http://localhost:4040");
        System.out.println("Нажмите ENTER для завершения программы.");
        new java.util.Scanner(System.in).nextLine();
        spark.stop();
    }
}
