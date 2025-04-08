import os
import re
import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, udf
from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType, FloatType

from ml_model import (
    BertEncoder,
    AutoEncoder,
    train_autoencoder,
    compute_reconstruction_errors
)


def normalize_message(msg):
    """
    - Заменяем IP-адреса -> [IP]
    - Заменяем User\d+ -> [USER]
    """
    msg = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]', str(msg))
    msg = re.sub(r'\bUser\d+\b', '[USER]', msg)
    return msg


def parse_time_taken(s):
    """
    '28ms' -> 28.0
    """
    if isinstance(s, str) and s.endswith("ms"):
        s = s[:-2]
    try:
        return float(s)
    except:
        return 0.0


def combine_fields_for_bert(loglevel, service, message, timestr=None):
    """
    Объединяем LogLevel, Service и сам текст лог-сообщения.
    """
    if timestr is not None:
        return f"LOGLEVEL={loglevel} SERVICE={service} TIME={timestr} TEXT={message}"
    else:
        return f"LOGLEVEL={loglevel} SERVICE={service} TEXT={message}"


def main():
    logDataPath = r"Z:\Diplom\SparkLogAnalysisMicroservice\SparkProcessingService\data\logdata.csv"
    modelSaveDir = r"Z:\Diplom\SparkLogAnalysisMicroservice\SparkProcessingService\model"
    anomaliesSavePath = r"Z:\Diplom\SparkLogAnalysisMicroservice\SparkProcessingService\anomaly-logdata"

    spark = SparkSession.builder \
        .appName("LogAnomalyDetectionBERT") \
        .master("local[*]") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.network.timeout", "36000s") \
        .config("spark.executor.heartbeatInterval", "3600s") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.python.worker.reuse", "false") \
        .config("spark.local.ip", "127.0.0.1") \
        .config("spark.driver.extraJavaOptions",
                "--add-opens=java.base/java.nio=ALL-UNNAMED " +
                "--add-exports=java.base/java.nio=ALL-UNNAMED " +
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED " +
                "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED") \
        .config("spark.executor.extraJavaOptions",
                "--add-opens=java.base/java.nio=ALL-UNNAMED " +
                "--add-exports=java.base/java.nio=ALL-UNNAMED " +
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED " +
                "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.local.dir", "Z:\\Diplom\\SparkLogAnalysisMicroservice\\SparkProcessingService\\temp\\spark") \
        .getOrCreate()

    print(spark.version)

    logs = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(logDataPath)

    if "_c0" in logs.columns:
        logs = logs.drop("_c0")

    normalize_message_udf = udf(normalize_message, StringType())
    parse_time_udf = udf(parse_time_taken, FloatType())

    logs = logs.withColumn("Message", normalize_message_udf(col("Message")))
    logs = logs.withColumn("TimeVal", parse_time_udf(col("TimeTaken")))

    logs.printSchema()

    train_df, test_df = logs.randomSplit([0.7, 0.3], seed=42)

    train_df = train_df.withColumn("row_id", monotonically_increasing_id())
    train_rows = train_df.select("row_id", "LogLevel", "Service", "Message", "TimeVal").na.drop().collect()
    train_ids = [r["row_id"] for r in train_rows]

    train_combined_texts = [
        combine_fields_for_bert(
            r["LogLevel"],
            r["Service"],
            r["Message"],
            timestr=str(r["TimeVal"])
        )
        for r in train_rows
    ]

    device = "cuda"
    bert_encoder = BertEncoder(device=device)
    autoencoder = AutoEncoder().to(device)

    print("=== Training AutoEncoder ===")
    train_autoencoder(
        texts=train_combined_texts,
        bert_encoder=bert_encoder,
        autoencoder=autoencoder,
        num_epochs=3,
        batch_size=32,
        device=device
    )

    print("=== Compute errors on train ===")
    train_errors = compute_reconstruction_errors(
        texts=train_combined_texts,
        bert_encoder=bert_encoder,
        autoencoder=autoencoder,
        device=device,
        batch_size=32
    )
    threshold = float(np.percentile(train_errors, 90))
    print("Anomaly threshold (90th percentile):", threshold)

    os.makedirs(modelSaveDir, exist_ok=True)
    torch.save(autoencoder.state_dict(), os.path.join(modelSaveDir, "autoencoder.pt"))
    with open(os.path.join(modelSaveDir, "threshold.txt"), "w") as f:
        f.write(str(threshold))

    print("=== Compute errors on test ===")
    test_df = test_df.withColumn("row_id", monotonically_increasing_id())
    test_rows = test_df.select("row_id", "LogLevel", "Service", "Message", "TimeVal").na.drop().collect()

    test_ids = [int(r["row_id"]) for r in test_rows]
    test_combined_texts = [
        combine_fields_for_bert(
            r["LogLevel"],
            r["Service"],
            r["Message"],
            timestr=str(r["TimeVal"])
        )
        for r in test_rows
    ]

    test_errors = compute_reconstruction_errors(
        texts=test_combined_texts,
        bert_encoder=bert_encoder,
        autoencoder=autoencoder,
        device=device,
        batch_size=32
    )

    test_errors = [float(e) for e in test_errors]

    records = list(zip(test_ids, test_errors))

    schema = StructType([
        StructField("row_id", LongType(), True),
        StructField("recon_error", DoubleType(), True)
    ])

    test_result_spark = spark.createDataFrame(records, schema=schema)

    joined_test_df = test_df.join(test_result_spark, on="row_id", how="inner")
    anomalies_df = joined_test_df.filter(col("recon_error") > threshold)

    count_anom = anomalies_df.count()
    print(f"Found {count_anom} anomalies in test set.")
    anomalies_df.show(10, False)

    anomalies_df.drop("row_id").coalesce(1).write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(anomaliesSavePath)

    print(f"Anomalies CSV saved to: {anomaliesSavePath}")
    print("Spark UI (if local) at http://localhost:4040")
    print("Нажмите ENTER для завершения.")
    input()
    spark.stop()


if __name__ == "__main__":
    main()
