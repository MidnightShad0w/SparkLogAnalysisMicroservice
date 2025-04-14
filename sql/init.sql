-- docker-compose exec clickhouse clickhouse-client -q "CREATE DATABASE IF NOT EXISTS logs"
-- docker-compose exec clickhouse clickhouse-client -q "CREATE TABLE IF NOT EXISTS logs.metrics ( event_time DateTime DEFAULT now(), run_id String, stage String, metric_name String, metric_value Float64 ) ENGINE=MergeTree() ORDER BY (event_time, run_id)"
-- docker-compose exec clickhouse clickhouse-client -q "SHOW TABLES FROM logs"
-- очистка
-- docker-compose exec clickhouse clickhouse-client -q "TRUNCATE TABLE logs.metrics;"

CREATE DATABASE IF NOT EXISTS logs;
CREATE TABLE IF NOT EXISTS logs.metrics (
    event_time  DateTime DEFAULT now(),
    run_id      String,
    stage       String,
    metric_name String,
    metric_value Float64
)
ENGINE = MergeTree()
ORDER BY (event_time, run_id);
