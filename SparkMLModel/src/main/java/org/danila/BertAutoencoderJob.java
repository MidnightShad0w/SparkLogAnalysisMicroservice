package org.danila;

import com.johnsnowlabs.nlp.DocumentAssembler;
import com.johnsnowlabs.nlp.annotators.Tokenizer;
import com.johnsnowlabs.nlp.embeddings.BertEmbeddings;
import org.apache.spark.ml.*;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.*;

import static org.apache.spark.sql.functions.*;

public class BertAutoencoderJob {

    public static void main(String[] args) {

        // Пути к данным и сохранению
        // Обратите внимание, используем прямые слэши и префикс "file:///".
        String logDataPath = "file:///Z:/Diplom/SparkLogAnalysisMicroservice/SparkMLModel/data/logdata.csv";
        String modelSavePath = "file:///Z:/Diplom/SparkLogAnalysisMicroservice/SparkMLModel/model-ae-bert";
        String resultSavePath = "file:///Z:/Diplom/SparkLogAnalysisMicroservice/SparkMLModel/output/anomaly-logdata-ae-bert";

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

        // 3.1. TimeTaken ( "28ms" -> 28.0 )
        TimeTakenConverterForBert timeTakenConverterForBert = new TimeTakenConverterForBert()
                .setInputCol("TimeTaken")
                .setOutputCol("TimeTakenNumeric");

        // 3.2. NLP Stages: DocumentAssembler -> Tokenizer -> BertEmbeddings -> SentenceEmbeddings
        DocumentAssembler documentAssembler = (DocumentAssembler) new DocumentAssembler()
                .setInputCol("Message")
                .setOutputCol("document");

        Tokenizer tokenizer = (Tokenizer)((Tokenizer) new Tokenizer()
                .setInputCols(new String[] {"document"}))
                .setOutputCol("token");

        BertEmbeddings bertEmbeddings = (BertEmbeddings) ((BertEmbeddings)BertEmbeddings.pretrained("small_bert_L4_128", "en")
                .setInputCols(new String[]{"document", "token"}))
                .setOutputCol("embeddings");

        SentenceEmbeddingsTransformer sentenceEmbeddings = new SentenceEmbeddingsTransformer()
                .setInputCol("embeddings")
                .setOutputCol("sentenceEmbeddings");

        // 3.3. VectorAssembler
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"sentenceEmbeddings", "TimeTakenNumeric"})
                .setOutputCol("features");

        // 3.4. AutoEncoder для аномалий
        AutoEncoder autoEncoder = new AutoEncoder()
                .setInputCol("features")
                .setOutputCol("reconstructionError");

        // Собираем Pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{
                        timeTakenConverterForBert,
                        documentAssembler,
                        tokenizer,
                        bertEmbeddings,
                        sentenceEmbeddings,
                        vectorAssembler,
                        autoEncoder
                });

        // 4. fit
        PipelineModel pipelineModel = pipeline.fit(trainingData);

        // 5a. Применяем модель (вызов transform) до сохранения
        //     (Spark уже знает net, т.к. тренировка прошла в этой же JVM)
        Dataset<Row> trainPredictions = pipelineModel.transform(trainingData);
        double[] trainQuantiles = trainPredictions.stat()
                .approxQuantile("reconstructionError", new double[]{0.90}, 0.0);
        double threshold = trainQuantiles[0];
        System.out.println("Train 90th quantile = " + threshold);

        Dataset<Row> testPredictions = pipelineModel.transform(testData);
        Dataset<Row> anomalies = testPredictions.filter(col("reconstructionError").gt(threshold));
        anomalies.show(false);

        // 5b. Сохраняем модель (теперь AutoEncoderModelWriter сохранит network.zip)
        try {
            pipelineModel.write().overwrite().save(modelSavePath);
            System.out.println("PipelineModel saved to: " + modelSavePath);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // 6. (Опционально) Вы можете загрузить модель заново (имитируя "другой процесс"):
        // PipelineModel loadedModel = PipelineModel.load(modelSavePath);
        // Dataset<Row> newPredictions = loadedModel.transform(testData);

        anomalies.coalesce(1)
                .write()
                .mode(SaveMode.Overwrite)
                .option("header", "true")
                .csv(resultSavePath);

        spark.stop();
    }
}
