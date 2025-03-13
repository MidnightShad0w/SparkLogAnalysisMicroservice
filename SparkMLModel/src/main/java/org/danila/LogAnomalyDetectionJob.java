package org.danila;

import static org.apache.spark.sql.functions.*;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

public class LogAnomalyDetectionJob {

    public static void main(String[] args) {

        // Входные и выходные пути
        String logDataPath = "C:\\Users\\admin\\Desktop\\Diplom\\LogAnalysisMicroservice\\SparkMLModel\\data\\logdata.csv";
        String modelSavePath = "C:\\Users\\admin\\Desktop\\Diplom\\LogAnalysisMicroservice\\SparkMLModel\\model"; // "s3a://bucket-spark/model"
        String resultSavePath = "C:\\Users\\admin\\Desktop\\Diplom\\LogAnalysisMicroservice\\SparkMLModel\\output\\anomaly-logdata";

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

        // Собираем Pipeline, включающий:
        // 1. TimeTakenConverterForBert – создаёт столбец "TimeTakenNumeric" из "TimeTaken"
        // 2. StringIndexer для "LogLevel" => "LogLevelIndex"
        // 3. VectorAssembler, который создаёт "features" из "TimeTakenNumeric" и "LogLevelIndex"
        // 4. KMeans для кластеризации
        TimeTakenConverter converter = new TimeTakenConverter();

        StringIndexer logLevelIndexer = new StringIndexer()
                .setInputCol("LogLevel")
                .setOutputCol("LogLevelIndex");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"TimeTakenNumeric", "LogLevelIndex"})
                .setOutputCol("features");

        KMeans kmeans = new KMeans()
                .setK(3)
                .setSeed(1L)
                .setFeaturesCol("features")
                .setPredictionCol("prediction");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{converter, logLevelIndexer, assembler, kmeans});

        // Разбиваем данные на обучающую и тестовую выборки
        Dataset<Row>[] splits = logs.randomSplit(new double[]{0.7, 0.3}, 12345L);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Обучаем Pipeline => PipelineModel
        PipelineModel pipelineModel = pipeline.fit(trainingData);

        // Оцениваем модель
        Dataset<Row> trainPredictions = pipelineModel.transform(trainingData);
        ClusteringEvaluator evaluator = new ClusteringEvaluator();
        double silhouette = evaluator.evaluate(trainPredictions);
        System.out.println("Silhouette score (training): " + silhouette);

        // Сохраняем обученную PipelineModel
        try {
            pipelineModel.write().overwrite().save(modelSavePath);
            System.out.println("PipelineModel saved to: " + modelSavePath);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Применяем модель к тестовым данным
        Dataset<Row> testPredictions = pipelineModel.transform(testData);

        // Для вычисления дистанций получаем KMeansModel из последнего этапа PipelineModel
        KMeansModel kmeansModel = (KMeansModel) pipelineModel.stages()[3];
        Vector[] centers = kmeansModel.clusterCenters();

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

        Dataset<Row> anomalies = testWithDistance.filter(col("distance").gt(threshold));
        anomalies.show(20, false);

        Dataset<Row> anomaliesOutput = anomalies.withColumn("features", col("features").cast("string"));
        anomaliesOutput.coalesce(1)
                .write()
                .mode(SaveMode.Overwrite)
                .option("header", "true")
                .csv(resultSavePath);

        System.out.println("Spark UI доступна (если локально) по адресу http://localhost:4040");
        System.out.println("Нажмите ENTER для завершения программы.");
        new java.util.Scanner(System.in).nextLine();
        spark.stop();
    }
}
