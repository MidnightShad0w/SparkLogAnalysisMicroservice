package org.danila;

import com.johnsnowlabs.nlp.DocumentAssembler;
import com.johnsnowlabs.nlp.annotators.Tokenizer;

import com.johnsnowlabs.nlp.embeddings.BertEmbeddings;
import org.apache.spark.ml.*;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.*;

public class LogAnomalyDetectionWithBertAutoencoderJob {

    public static void main(String[] args) {

        // Пути к данным и сохранению
        String logDataPath = "C:\\Users\\admin\\Desktop\\logs\\logdata.csv";  // CSV с полями Timestamp,LogLevel,Service,Message,RequestID,User,ClientIP,TimeTaken
        String modelSavePath = "C:\\Users\\admin\\Desktop\\model\\bert_ae_model";
        String resultSavePath = "C:\\Users\\admin\\Desktop\\output\\bert_ae_anomalies";

        // Создаём SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("Log Anomaly Detection with BERT and AutoEncoder")
                .master("local[*]")
                .getOrCreate();

        // 1. Читаем CSV
        Dataset<Row> logs = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(logDataPath);

        // 2. Разбиваем на train/test
        Dataset<Row>[] splits = logs.randomSplit(new double[]{0.7, 0.3}, 12345L);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // 3. Собираем Pipeline:

        // 3.1. Преобразуем TimeTaken (например, "28ms" -> 28.0)
        TimeTakenConverter timeTakenConverter = new TimeTakenConverter()
                .setInputCol("TimeTaken")
                .setOutputCol("TimeTakenNumeric");

        // 3.2. NLP: Message -> Document -> Token -> BertEmbeddings -> SentenceEmbeddings
        DocumentAssembler documentAssembler = (DocumentAssembler) new DocumentAssembler()
                .setInputCol("Message")
                .setOutputCol("document");

        Tokenizer tokenizer = (Tokenizer)((Tokenizer) new Tokenizer()
                .setInputCols(new String[] {"document"}))
                .setOutputCol("token");


        // Подгружаем предобученную BERT-модель "small_bert_L4_128" (можно заменить на другую)
        BertEmbeddings bertEmbeddings = (BertEmbeddings) ((BertEmbeddings)BertEmbeddings.pretrained("small_bert_L4_128", "en")
                .setInputCols(new String[]{"document", "token"}))
                .setOutputCol("embeddings");

        SentenceEmbeddingsTransformer sentenceEmbeddings = new SentenceEmbeddingsTransformer()
                .setInputCol("embeddings")
                .setOutputCol("sentenceEmbeddings");

        // 3.3. VectorAssembler: объединяем эмбеддинги и TimeTakenNumeric в столбец "features"
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"sentenceEmbeddings", "TimeTakenNumeric"})
                .setOutputCol("features");

        // 3.4. AutoEncoder для обнаружения аномалий
        AutoEncoder autoEncoder = new AutoEncoder()
                .setInputCol("features")
                .setOutputCol("reconstructionError");

        // Собираем Pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{
                        timeTakenConverter,
                        documentAssembler,
                        tokenizer,
                        bertEmbeddings,
                        sentenceEmbeddings,
                        vectorAssembler,
                        autoEncoder
                });

        // 4. Тренируем модель на trainingData
        PipelineModel pipelineModel = pipeline.fit(trainingData);

        // 5. Сохраняем обученную PipelineModel
        try {
            pipelineModel.write().overwrite().save(modelSavePath);
            System.out.println("PipelineModel saved to: " + modelSavePath);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // 6. Применяем модель к тренировочным данным, чтобы вычислить распределение ошибок
        Dataset<Row> trainPredictions = pipelineModel.transform(trainingData);

        // Вычислим 90-процентиль ошибки для определения порога аномалий
        double[] quantiles = trainPredictions.stat().approxQuantile("reconstructionError", new double[]{0.90}, 0.0);
        double threshold = quantiles[0];
        System.out.println("Anomaly threshold (90th percentile): " + threshold);

        // 7. Применяем модель к тестовым данным
        Dataset<Row> testPredictions = pipelineModel.transform(testData);

        // 8. Фильтруем аномалии
        Dataset<Row> anomalies = testPredictions.filter(col("reconstructionError").gt(threshold));
        anomalies.show(20, false);

        // 9. Сохраняем аномалии в CSV
        anomalies.coalesce(1)
                .write()
                .mode(SaveMode.Overwrite)
                .option("header", "true")
                .csv(resultSavePath);

        System.out.println("Spark UI доступна по адресу http://localhost:4040 (если локально).");
        System.out.println("Нажмите ENTER для завершения...");
        new java.util.Scanner(System.in).nextLine();
        spark.stop();
    }
}