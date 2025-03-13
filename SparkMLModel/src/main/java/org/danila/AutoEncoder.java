package org.danila;

import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.Identifiable;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.apache.spark.sql.Encoders;

import java.util.ArrayList;
import java.util.List;

public class AutoEncoder extends Estimator<AutoEncoderModel> {

    private final String uid;
    private String inputCol;
    private String outputCol;

    public AutoEncoder() {
        this.uid = Identifiable.randomUID("autoEncoder");
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

        // Возьмём только нужный столбец (features)
        Dataset<Row> onlyFeatures = dataset.select(inputCol);

        // Собираем все строки
        List<Row> rows = onlyFeatures.collectAsList();

        // Преобразуем к List<Vector>
        List<Vector> vectors = new ArrayList<>(rows.size());
        for (Row r : rows) {
            // getAs(0) или getAs("features")
            Vector v = r.getAs(0);
            vectors.add(v);
        }
        if (vectors.isEmpty()) {
            throw new RuntimeException("Нет данных для обучения автоэнкодера");
        }

        int inputDim = vectors.get(0).size();
        int hiddenDim = Math.max(1, inputDim / 2); // Простейший вариант - половина входной размерности
        int n = vectors.size();

        double[][] data = new double[n][inputDim];
        for (int i = 0; i < n; i++) {
            data[i] = vectors.get(i).toArray();
        }
        INDArray inputData = Nd4j.create(data);

        // Простейшая архитектура автоэнкодера: вход -> (Dense, ReLU) -> (Dense, Identity) -> выход
        org.deeplearning4j.nn.conf.MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(inputDim)
                        .nOut(hiddenDim)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(hiddenDim)
                        .nOut(inputDim)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Обучаем модель, например, 10 эпох
        int nEpochs = 10;
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            model.fit(inputData, inputData);
            System.out.println("Epoch " + (epoch + 1) + " completed.");
        }

        return new AutoEncoderModel(model, inputCol, outputCol);
    }

    @Override
    public StructType transformSchema(StructType schema) {
        // Добавляем новый столбец outputCol (Double), в котором будет храниться ошибка реконструкции
        return schema.add(outputCol, DataTypes.DoubleType, false);
    }

    @Override
    public AutoEncoder copy(ParamMap extra) {
        return defaultCopy(extra);
    }

    @Override
    public String uid() {
        return uid;
    }
}
