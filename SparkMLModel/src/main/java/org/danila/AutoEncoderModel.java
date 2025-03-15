package org.danila;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.expressions.UserDefinedFunction;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;

/**
 * AutoEncoderModel с DL4J MultiLayerNetwork + MLWritable.
 */
public class AutoEncoderModel extends Model<AutoEncoderModel>
        implements MLWritable {

    private String uid;
    private String inputCol;
    private String outputCol;

    // DL4J-модель
    private transient MultiLayerNetwork net;
    private String savedNetPath; // путь к network.zip

    public AutoEncoderModel(String uid, MultiLayerNetwork net,
                            String inputCol, String outputCol, String savedNetPath) {
        this.uid = uid;
        this.net = net;
        this.inputCol = inputCol;
        this.outputCol = outputCol;
        this.savedNetPath = savedNetPath;
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        // Если net == null, загрузить из savedNetPath
        if (net == null && savedNetPath != null && !savedNetPath.isEmpty()) {
            try {
                File netFile = new File(savedNetPath);
                if (!netFile.exists()) {
                    throw new RuntimeException("DL4J model file not found: " + savedNetPath);
                }
                net = org.deeplearning4j.util.ModelSerializer.restoreMultiLayerNetwork(netFile);
            } catch (Exception e) {
                throw new RuntimeException("Failed to load DL4J model from " + savedNetPath, e);
            }
        }

        String fnName = uid + "_reconstructionUDF";
        UserDefinedFunction recErrorUDF = udf((Vector features) -> {
            if (features == null) return Double.NaN;
            org.nd4j.linalg.api.ndarray.INDArray inputArr = Nd4j.create(features.toArray());
            org.nd4j.linalg.api.ndarray.INDArray outputArr = net.output(inputArr, false);
            return Transforms.euclideanDistance(inputArr, outputArr);
        }, DataTypes.DoubleType);

        SparkSession spark = dataset.sparkSession();
        spark.udf().register(fnName, recErrorUDF);
        return dataset.withColumn(outputCol, callUDF(fnName, col(inputCol)));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema.add(outputCol, DataTypes.DoubleType, false);
    }

    @Override
    public AutoEncoderModel copy(ParamMap extra) {
        return new AutoEncoderModel(uid, net, inputCol, outputCol, savedNetPath);
    }

    @Override
    public String uid() {
        return uid;
    }

    // ==== MLWritable =====
    @Override
    public MLWriter write() {
        return new AutoEncoderModelWriter(this);
    }

    private static class AutoEncoderModelWriter extends MLWriter {
        private final AutoEncoderModel instance;

        AutoEncoderModelWriter(AutoEncoderModel instance) {
            this.instance = instance;
        }

        @Override
        public void saveImpl(String path) {
            try {
                // 1) Сохраняем метаданные
                // Spark передаст path = "file:///Z:/.../stages/6_autoEncoder_xxxx"
                DefaultParamsWriter.saveMetadata(instance, path, sc(), scala.Option.empty(), scala.Option.empty());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            // 2) Преобразуем Spark URI "file:///Z:/..." -> локальный путь "Z:/..."
            String localPath = stripFileURI(path); // метод, который убирает "file:///"
            // Создаём подпапку dl4jNetwork
            File netDir = new File(localPath, "dl4jNetwork");
            if (!netDir.exists()) {
                netDir.mkdirs();
            }
            File outFile = new File(netDir, "network.zip");
            try {
                org.deeplearning4j.util.ModelSerializer.writeModel(instance.net, outFile, true);
                instance.savedNetPath = outFile.getAbsolutePath();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            // 3) Сохраняем extra JSON
            String extraJson = "{ \"inputCol\":\"" + instance.inputCol + "\", " +
                    "\"outputCol\":\"" + instance.outputCol + "\", " +
                    "\"savedNetPath\":\"" + instance.savedNetPath + "\" }";
            String extraPath = path + "/autoEncoderModelParams";
            TimeTakenConverterForBert.saveJsonToFile(extraJson, extraPath);
        }

        // Убираем "file:///" из начала, и заменяем backslashes, если нужно
        private String stripFileURI(String sparkPath) {
            // sparkPath может быть "file:///Z:/..."
            // убираем префикс "file:///" -> "Z:/..."
            String res = sparkPath.replaceFirst("^file:/*", "");
            // Теперь res ~ "Z:/Diplom/..."
            // Можно заменить обратные слэши, если sparkPath почему-то содержит их
            res = res.replace("\\", "/");
            return res;
        }
    }

    // ==== MLReadable =====
    public static MLReader<AutoEncoderModel> read() {
        return new AutoEncoderModelReader();
    }

    public static AutoEncoderModel load(String path) {
        return (AutoEncoderModel) read().load(path);
    }

    private static class AutoEncoderModelReader extends MLReader<AutoEncoderModel> {
        @Override
        public AutoEncoderModel load(String path) {
            try {
                DefaultParamsReader.Metadata metadata =
                        DefaultParamsReader.loadMetadata(path, sc(), "org.danila.AutoEncoderModel");
                String extraPath = path + "/autoEncoderModelParams";
                String extraJson = TimeTakenConverterForBert.readJsonFile(extraPath);

                String inputColValue = TimeTakenConverterForBert.parseJsonField(extraJson, "inputCol");
                String outputColValue = TimeTakenConverterForBert.parseJsonField(extraJson, "outputCol");
                String netPath = TimeTakenConverterForBert.parseJsonField(extraJson, "savedNetPath");

                // Загружаем DL4J-модель, если файл существует
                MultiLayerNetwork net = null;
                if (netPath != null && !netPath.isEmpty()) {
                    File netFile = new File(netPath);
                    if (netFile.exists()) {
                        net = org.deeplearning4j.util.ModelSerializer.restoreMultiLayerNetwork(netFile);
                    } else {
                        // Можно бросить исключение, или net останется null
                        System.err.println("Warning: no DL4J model found at " + netPath);
                    }
                }
                return new AutoEncoderModel(metadata.uid(), net, inputColValue, outputColValue, netPath);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
