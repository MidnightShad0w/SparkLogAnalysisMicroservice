package org.danila;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.Identifiable;
import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.api.java.UDF1;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.api.ndarray.INDArray;

public class AutoEncoderModel extends Model<AutoEncoderModel> {

    private final String uid;
    private final String inputCol;
    private final String outputCol;
    private final MultiLayerNetwork model;

    public AutoEncoderModel(MultiLayerNetwork model, String inputCol, String outputCol) {
        this.uid = Identifiable.randomUID("autoEncoderModel");
        this.model = model;
        this.inputCol = inputCol;
        this.outputCol = outputCol;
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        SparkSession spark = dataset.sparkSession();

        UDF1<Vector, Double> recErrorFn = new UDF1<Vector, Double>() {
            @Override
            public Double call(Vector vector) throws Exception {
                INDArray inputArr = Nd4j.create(vector.toArray());
                INDArray outputArr = model.output(inputArr, false);
                return Transforms.euclideanDistance(inputArr, outputArr);
            }
        };

        spark.udf().register("recErrorUDF", recErrorFn, DataTypes.DoubleType);

        return dataset.withColumn(outputCol, callUDF("recErrorUDF", col(inputCol)));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        // Добавляем столбец outputCol с типом Double
        return schema.add(outputCol, DataTypes.DoubleType, false);
    }

    @Override
    public AutoEncoderModel copy(ParamMap extra) {
        return new AutoEncoderModel(model, inputCol, outputCol);
    }

    @Override
    public String uid() {
        return uid;
    }
}
