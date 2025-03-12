package org.danila;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.Identifiable;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.regexp_replace;

public class TimeTakenConverter extends Transformer {

    private final String uid;
    private String inputCol;
    private String outputCol;

    public TimeTakenConverter() {
        this.uid = Identifiable.randomUID("timeTakenConverter");
    }

    public TimeTakenConverter setInputCol(String inputCol) {
        this.inputCol = inputCol;
        return this;
    }

    public TimeTakenConverter setOutputCol(String outputCol) {
        this.outputCol = outputCol;
        return this;
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        // Удаляем "ms" и приводим к Double
        return dataset.withColumn(outputCol,
                regexp_replace(col(inputCol), "ms", "").cast(DataTypes.DoubleType));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema.add(outputCol, DataTypes.DoubleType, false);
    }

    @Override
    public TimeTakenConverter copy(ParamMap extra) {
        return defaultCopy(extra);
    }

    @Override
    public String uid() {
        return uid;
    }
}
