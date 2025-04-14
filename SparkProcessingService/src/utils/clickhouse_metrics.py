from clickhouse_driver import Client
import time


def send_metric(run_id, stage, metric_name, metric_value):
    """
    Отправляет одну метрику в таблицу logs.metrics (ClickHouse).
    Параметры:
      run_id (str)      -- уникальный ID / имя запуска
      stage (str)       -- "AirflowDAG", "SparkProcess", ...
      metric_name (str) -- "total_logs", "anomalies_count", ...
      metric_value (float)
    """
    client = Client(
        host="localhost",
        port=9003,
        database="logs",
        user="default",
        password=""
    )
    rows_to_insert = [(run_id, stage, metric_name, metric_value)]
    client.execute(
        """
        INSERT INTO logs.metrics (run_id, stage, metric_name, metric_value)
        VALUES
        """,
        rows_to_insert
    )
