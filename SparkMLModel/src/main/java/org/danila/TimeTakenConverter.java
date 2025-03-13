package org.danila;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.DefaultParamsWritable;
import org.apache.spark.ml.util.Identifiable;
import org.apache.spark.ml.util.MLReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;

import java.io.Serializable;
import java.util.Arrays;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.regexp_replace;

/**
 * Кастомный Transformer для преобразования TimeTaken => TimeTakenNumeric.
 * Реализует DefaultParamsWritable, чтобы сохраняться в Pipeline,
 * и даёт статические методы read()/load() для загрузки.
 */
public class TimeTakenConverter extends Transformer
        implements DefaultParamsWritable, Serializable {

    private final String uid;

    public TimeTakenConverter() {
        this.uid = Identifiable.randomUID("TimeTakenConverter");
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        return dataset.withColumn(
                "TimeTakenNumeric",
                regexp_replace(col("TimeTaken"), "ms", "").cast("double")
        );
    }

    @Override
    public StructType transformSchema(StructType schema) {
        if (!Arrays.asList(schema.fieldNames()).contains("TimeTakenNumeric")) {
            return schema.add("TimeTakenNumeric", "double", true);
        }
        return schema;
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return defaultCopy(extra);
    }

    @Override
    public String uid() {
        return uid;
    }

    // ======= Часть, необходимая Spark для чтения (де)сериализации =======

    /**
     * Статический метод, который Spark вызовет рефлексией при загрузке PipelineModel.
     */
    public static MLReader<TimeTakenConverter> read() {
        return new TimeTakenConverterReader();
    }

    /**
     * Аналогично (по примеру Spark ML) — метод load(...).
     */
    public static TimeTakenConverter load(String path) {
        return read().load(path);
    }

    /**
     * Класс-ридер, наследующий MLReader<TimeTakenConverter>.
     */
    private static class TimeTakenConverterReader extends MLReader<TimeTakenConverter> {
        @Override
        public TimeTakenConverter load(String path) {
            // Можно прочитать metadata JSON, если в Transformer
            // есть какие-то Params.
            // Здесь достаточно вернуть новый объект.
            return new TimeTakenConverter();
        }
    }
}
