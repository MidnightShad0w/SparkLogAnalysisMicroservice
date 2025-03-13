package org.danila;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.Identifiable;
import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.api.java.UDF1;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import java.util.List;

import static org.apache.spark.sql.functions.*;

public class SentenceEmbeddingsTransformer extends Transformer {

    private final String uid;
    private String inputCol;
    private String outputCol;

    public SentenceEmbeddingsTransformer() {
        this.uid = Identifiable.randomUID("sentenceEmbeddingsTransformer");
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
        // Получаем SparkSession из Dataset
        SparkSession spark = dataset.sparkSession();

        // Создаём логику UDF с помощью интерфейса UDF1
        UDF1<Object, Vector> avgEmbeddings = new UDF1<Object, Vector>() {
            @Override
            public Vector call(Object input) throws Exception {
                if (input == null) {
                    // Возвращаем нулевой вектор (пример)
                    return Vectors.dense(new double[768]);
                }
                // Приводим к scala.collection.Seq<Row>
                @SuppressWarnings("unchecked")
                Seq<org.apache.spark.sql.Row> rowSeq = (Seq<Row>) input;

                // Конвертируем в Java-список
                java.util.List<org.apache.spark.sql.Row> annotations =
                        JavaConverters.seqAsJavaList(rowSeq);

                if (annotations.isEmpty()) {
                    return Vectors.dense(new double[768]); // например, размерность 768
                }
                List<?> firstEmb = annotations.get(0).getList(annotations.get(0).fieldIndex("embeddings"));
                int dim = firstEmb.size();
                double[] sum = new double[dim];
                int count = 0;
                for (Row ann : annotations) {
                    List<?> embList = ann.getList(ann.fieldIndex("embeddings"));
                    for (int i = 0; i < dim; i++) {
                        sum[i] += Double.parseDouble(embList.get(i).toString());
                    }
                    count++;
                }
                for (int i = 0; i < dim; i++) {
                    sum[i] /= count;
                }
                return Vectors.dense(sum);
            }
        };

        UserDefinedFunction avgEmbeddingsUDF = udf(avgEmbeddings, new VectorUDT());

        return dataset.withColumn(outputCol, avgEmbeddingsUDF.apply(col(inputCol)));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema.add(outputCol, new org.apache.spark.ml.linalg.VectorUDT(), false);
    }

    @Override
    public SentenceEmbeddingsTransformer copy(ParamMap extra) {
        return defaultCopy(extra);
    }

    @Override
    public String uid() {
        return uid;
    }
}
