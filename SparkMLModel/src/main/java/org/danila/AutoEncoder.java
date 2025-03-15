package org.danila;

import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import org.apache.spark.sql.Encoders;

import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;

import static org.apache.spark.sql.functions.col;

/**
 * AutoEncoder - Estimator, который обучает DL4J‑модель (автоэнкодер) на features
 * и возвращает AutoEncoderModel.
 *
 * Реализует MLWritable, чтобы Spark мог сохранить PipelineModel,
 * но фактическую DL4J‑модель сериализуем в AutoEncoderModel.
 */
public class AutoEncoder extends Estimator<AutoEncoderModel>
        implements MLWritable {

    private String uid;
    private String inputCol;
    private String outputCol;

    public AutoEncoder() {
        this.uid = Identifiable.randomUID("org.danila.AutoEncoder");
    }

    public AutoEncoder setInputCol(String inputCol) {
        this.inputCol = inputCol;
        return this;
    }

    public AutoEncoder setOutputCol(String outputCol) {
        this.outputCol = outputCol;
        return this;
    }

    @Override
    public AutoEncoderModel fit(Dataset<?> dataset) {
        // 1) Считываем features => List<Vector>
        // (Сериализовать VectorUDT через collectAsList)
        List<Row> rows = dataset.select(col(inputCol)).collectAsList();
        List<org.apache.spark.ml.linalg.Vector> vectors = new ArrayList<>(rows.size());
        for (Row r : rows) {
            // Предполагаем, что столбец уже имеет тип Vector (VectorUDT)
            org.apache.spark.ml.linalg.Vector v = r.getAs(0);
            vectors.add(v);
        }

        if (vectors.isEmpty()) {
            throw new RuntimeException("Нет данных для обучения автоэнкодера");
        }

        int inputDim = vectors.get(0).size();
        int hiddenDim = Math.max(1, inputDim / 2);
        int n = vectors.size();

        double[][] data = new double[n][inputDim];
        for (int i = 0; i < n; i++) {
            data[i] = vectors.get(i).toArray();
        }
        org.nd4j.linalg.api.ndarray.INDArray inputData = Nd4j.create(data);

        // 2) Создаём DL4J‑конфигурацию (пример)
        // OutputLayer с MSE
        org.deeplearning4j.nn.conf.MultiLayerConfiguration conf =
                new org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder()
                        .seed(123)
                        .list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                                .nIn(inputDim)
                                .nOut(hiddenDim)
                                .activation(org.nd4j.linalg.activations.Activation.RELU)
                                .build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.MSE)
                                .nIn(hiddenDim)
                                .nOut(inputDim)
                                .activation(org.nd4j.linalg.activations.Activation.IDENTITY)
                                .build())
                        .build();

        org.deeplearning4j.nn.multilayer.MultiLayerNetwork net =
                new org.deeplearning4j.nn.multilayer.MultiLayerNetwork(conf);
        net.init();

        // 3) Обучаем
        int nEpochs = 5;
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            net.fit(inputData, inputData);
            System.out.println("Epoch " + (epoch+1) + " done");
        }

        // 4) Создаём AutoEncoderModel, передаём DL4J‑модель
        AutoEncoderModel model = new AutoEncoderModel(uid, net, inputCol, outputCol, "");
        return model;
    }

    @Override
    public StructType transformSchema(StructType schema) {
        // Добавляем столбец outputCol (Double), в котором хранится ошибка
        return schema.add(outputCol, DataTypes.DoubleType, false);
    }

    @Override
    public AutoEncoder copy(ParamMap extra) {
        AutoEncoder copy = new AutoEncoder();
        copy.setInputCol(this.inputCol);
        copy.setOutputCol(this.outputCol);
        copy.uid = this.uid;
        return copy;
    }

    @Override
    public String uid() {
        return uid;
    }

    // ===== MLWritable =====
    @Override
    public MLWriter write() {
        return new AutoEncoderWriter(this);
    }

    private static class AutoEncoderWriter extends MLWriter {
        private final AutoEncoder instance;
        AutoEncoderWriter(AutoEncoder instance) {
            this.instance = instance;
        }

        @Override
        public void saveImpl(String path) {
            try {
                DefaultParamsWriter.saveMetadata(instance, path, sc(), scala.Option.empty(), scala.Option.empty());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            String extraJson = "{ \"inputCol\":\"" + instance.inputCol + "\", " +
                    "\"outputCol\":\"" + instance.outputCol + "\" }";
            String extraPath = path + "/autoEncoderParams";
            TimeTakenConverterForBert.saveJsonToFile(extraJson, extraPath);
        }
    }

    // ====== MLReadable ======
    public static MLReader<AutoEncoder> read() {
        return new AutoEncoderReader();
    }

    public static AutoEncoder load(String path) {
        return (AutoEncoder) read().load(path);
    }

    private static class AutoEncoderReader extends MLReader<AutoEncoder> {
        @Override
        public AutoEncoder load(String path) {
            try {
                DefaultParamsReader.Metadata metadata = DefaultParamsReader.loadMetadata(path, sc(), "org.danila.AutoEncoder");
                String extraPath = path + "/autoEncoderParams";
                String extraJson = TimeTakenConverterForBert.readJsonFile(extraPath);

                AutoEncoder instance = new AutoEncoder();
                instance.uid = metadata.uid();
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
