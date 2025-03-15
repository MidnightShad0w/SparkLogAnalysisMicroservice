package org.danila;

import scala.Option;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.*;

/**
 * Простейший кастомный Transformer, который удаляет "ms" из столбца TimeTaken и приводит к double.
 * Реализует MLWritable/MLReadable, чтобы Spark мог сохранить/загрузить Pipeline, содержащий его.
 */
public class TimeTakenConverterForBert extends Transformer
        implements MLWritable {

    private String uid;
    private String inputCol;
    private String outputCol;

    public TimeTakenConverterForBert() {
        this.uid = Identifiable.randomUID("org.danila.TimeTakenConverterForBert");
    }

    public TimeTakenConverterForBert setInputCol(String inputCol) {
        this.inputCol = inputCol;
        return this;
    }

    public TimeTakenConverterForBert setOutputCol(String outputCol) {
        this.outputCol = outputCol;
        return this;
    }

    // ====== Реализация Transformer ======
    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        return dataset.withColumn(outputCol,
                regexp_replace(col(inputCol), "ms", "").cast(DataTypes.DoubleType));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        // Добавляем столбец outputCol с типом double
        return schema.add(outputCol, DataTypes.DoubleType, false);
    }

    @Override
    public TimeTakenConverterForBert copy(ParamMap extra) {
        TimeTakenConverterForBert copy = new TimeTakenConverterForBert();
        copy.setInputCol(this.inputCol);
        copy.setOutputCol(this.outputCol);
        return copy;
    }

    @Override
    public String uid() {
        return uid;
    }

    // ====== Реализация MLWritable ======
    @Override
    public MLWriter write() {
        return new TimeTakenConverterForBertWriter(this);
    }

    private static class TimeTakenConverterForBertWriter extends MLWriter {
        private final TimeTakenConverterForBert instance;

        private TimeTakenConverterForBertWriter(TimeTakenConverterForBert instance) {
            this.instance = instance;
        }

        @Override
        public void saveImpl(String path) {
            // 1) Сохраняем метаданные (uid, inputCol, outputCol)
            // Используем Helper-класс DefaultParamsWriter
            try {
                DefaultParamsWriter.saveMetadata(instance, path, sc(), scala.Option.empty(), scala.Option.empty());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            // 2) Сохраняем сами поля (inputCol, outputCol) в JSON вручную
            String extraJson = "{ \"inputCol\":\"" + instance.inputCol + "\", " +
                    "\"outputCol\":\"" + instance.outputCol + "\" }";

            String extraPath = path + "/timeTakenConverterParams";
            sparkSession().sparkContext().hadoopConfiguration();
            // Записываем
            saveJsonToFile(extraJson, extraPath);
        }
    }

    // ====== MLReadable ======
    // Чтобы Spark мог .read() -> TimeTakenConverterForBert
    public static MLReader<TimeTakenConverterForBert> read() {
        return new TimeTakenConverterForBertReader();
    }

    public static TimeTakenConverterForBert load(String path) {
        return (TimeTakenConverterForBert) read().load(path);
    }

    private static class TimeTakenConverterForBertReader extends MLReader<TimeTakenConverterForBert> {
        @Override
        public TimeTakenConverterForBert load(String path) {
            // 1) Считываем метаданные
            try {
                DefaultParamsReader.Metadata metadata = DefaultParamsReader.loadMetadata(path, sc(), "org.danila.TimeTakenConverterForBert");
                // 2) Создаём новый экземпляр
                TimeTakenConverterForBert instance = new TimeTakenConverterForBert();
                // 3) Читаем JSON с параметрами
                String extraPath = path + "/timeTakenConverterParams";
                String extraJson = readJsonFile(extraPath);
                // Разбираем простым способом:
                //  {"inputCol":"TimeTaken","outputCol":"TimeTakenNumeric"}
                String inputColValue = parseJsonField(extraJson, "inputCol");
                String outputColValue = parseJsonField(extraJson, "outputCol");

                instance.setInputCol(inputColValue);
                instance.setOutputCol(outputColValue);

                // uid восстанавливается из metadata
                instance.uid = metadata.uid();
                return instance;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    // ===== вспомогательные методы (для примера) =====

    public static void saveJsonToFile(String json, String path) {
        // Упрощённо: Записываем json как "part-00000" обычным способом
        // В реальном проекте лучше использовать sparkSession().createDataset(...)
        try {
            org.apache.hadoop.conf.Configuration hadoopConf = SparkSession.active().sparkContext().hadoopConfiguration();
            org.apache.hadoop.fs.Path p = new org.apache.hadoop.fs.Path(path + "/params.json");
            org.apache.hadoop.fs.FileSystem fs = p.getFileSystem(hadoopConf);
            try (org.apache.hadoop.fs.FSDataOutputStream out = fs.create(p, true)) {
                out.writeUTF(json);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static String readJsonFile(String path) {
        try {
            org.apache.hadoop.conf.Configuration hadoopConf = SparkSession.active().sparkContext().hadoopConfiguration();
            org.apache.hadoop.fs.Path p = new org.apache.hadoop.fs.Path(path + "/params.json");
            org.apache.hadoop.fs.FileSystem fs = p.getFileSystem(hadoopConf);
            try (org.apache.hadoop.fs.FSDataInputStream in = fs.open(p)) {
                return in.readUTF();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static String parseJsonField(String json, String field) {
        // Минимальный парс: ищем "field":"value"
        // В боевом коде используйте Jackson/Gson.
        String needle = "\"" + field + "\":\"";
        int idx = json.indexOf(needle);
        if (idx < 0) return null;
        int start = idx + needle.length();
        int end = json.indexOf("\"", start);
        return json.substring(start, end);
    }
}
