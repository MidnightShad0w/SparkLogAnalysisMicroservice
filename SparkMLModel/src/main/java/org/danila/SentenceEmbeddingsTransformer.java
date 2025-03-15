package org.danila;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.api.java.UDF1;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import static org.apache.spark.sql.functions.*;

/**
 * Трансформер, который усредняет эмбеддинги (array<Row>), превращая их в Vector.
 * Реализует MLWritable для возможности сохранения в PipelineModel.
 */
public class SentenceEmbeddingsTransformer extends Transformer
        implements MLWritable {

    private String uid;
    private String inputCol;
    private String outputCol;

    public SentenceEmbeddingsTransformer() {
        this.uid = Identifiable.randomUID("org.danila.SentenceEmbeddingsTransformer");
    }

    public SentenceEmbeddingsTransformer setInputCol(String inputCol) {
        this.inputCol = inputCol;
        return this;
    }

    public SentenceEmbeddingsTransformer setOutputCol(String outputCol) {
        this.outputCol = outputCol;
        return this;
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        SparkSession spark = dataset.sparkSession();

        // Регистрируем UDF по имени (например "avgEmbeddingsUDF_xxx")
        String functionName = uid + "_avgEmbeddingsUDF";
        UDF1<Object, Vector> avgEmbeddingsFn = (Object input) -> {
            if (input == null) {
                return Vectors.dense(new double[768]);
            }
            @SuppressWarnings("unchecked")
            Seq<org.apache.spark.sql.Row> rowSeq = (Seq<org.apache.spark.sql.Row>) input;
            java.util.List<org.apache.spark.sql.Row> annotations = JavaConverters.seqAsJavaList(rowSeq);
            if (annotations.isEmpty()) {
                return Vectors.dense(new double[768]);
            }
            // Предполагаем, что embeddings:Array<float> => List<...>
            java.util.List<?> firstEmb = annotations.get(0).getList(annotations.get(0).fieldIndex("embeddings"));
            int dim = firstEmb.size();
            double[] sum = new double[dim];
            int count = 0;
            for (org.apache.spark.sql.Row ann : annotations) {
                java.util.List<?> embList = ann.getList(ann.fieldIndex("embeddings"));
                for (int i = 0; i < dim; i++) {
                    sum[i] += Double.parseDouble(embList.get(i).toString());
                }
                count++;
            }
            for (int i = 0; i < dim; i++) {
                sum[i] /= count;
            }
            return Vectors.dense(sum);
        };

        spark.udf().register(functionName, avgEmbeddingsFn, new VectorUDT());

        return dataset.withColumn(outputCol, callUDF(functionName, col(inputCol)));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema.add(outputCol, new VectorUDT(), false);
    }

    @Override
    public SentenceEmbeddingsTransformer copy(ParamMap extra) {
        SentenceEmbeddingsTransformer t = new SentenceEmbeddingsTransformer();
        t.setInputCol(this.inputCol);
        t.setOutputCol(this.outputCol);
        return t;
    }

    @Override
    public String uid() {
        return uid;
    }

    // ====== Реализация MLWritable ======
    @Override
    public MLWriter write() {
        return new SentenceEmbeddingsTransformerWriter(this);
    }

    private static class SentenceEmbeddingsTransformerWriter extends MLWriter {
        private final SentenceEmbeddingsTransformer instance;

        SentenceEmbeddingsTransformerWriter(SentenceEmbeddingsTransformer instance) {
            this.instance = instance;
        }

        @Override
        public void saveImpl(String path) {
            // 1) Сохраняем метаданные
            try {
                DefaultParamsWriter.saveMetadata(instance, path, sc(), scala.Option.empty(), scala.Option.empty());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            // 2) Сохраняем поля inputCol, outputCol
            String extraJson = "{ \"inputCol\":\"" + instance.inputCol + "\", " +
                    "\"outputCol\":\"" + instance.outputCol + "\" }";
            String extraPath = path + "/sentenceEmbParams";
            TimeTakenConverterForBert.saveJsonToFile(extraJson, extraPath); // используем метод, написанный в другом классе
        }
    }

    // ====== MLReadable ======
    public static MLReader<SentenceEmbeddingsTransformer> read() {
        return new SentenceEmbeddingsTransformerReader();
    }

    public static SentenceEmbeddingsTransformer load(String path) {
        return (SentenceEmbeddingsTransformer) read().load(path);
    }

    private static class SentenceEmbeddingsTransformerReader extends MLReader<SentenceEmbeddingsTransformer> {
        @Override
        public SentenceEmbeddingsTransformer load(String path) {
            try {
                DefaultParamsReader.Metadata metadata = DefaultParamsReader.loadMetadata(path, sc(), "org.danila.SentenceEmbeddingsTransformer");
                SentenceEmbeddingsTransformer instance = new SentenceEmbeddingsTransformer();
                instance.uid = metadata.uid();
                String extraPath = path + "/sentenceEmbParams";
                String extraJson = TimeTakenConverterForBert.readJsonFile(extraPath);

                String inputColValue = TimeTakenConverterForBert.parseJsonField(extraJson, "inputCol");
                String outputColValue = TimeTakenConverterForBert.parseJsonField(extraJson, "outputCol");
                instance.setInputCol(inputColValue);
                instance.setOutputCol(outputColValue);
                return instance;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
