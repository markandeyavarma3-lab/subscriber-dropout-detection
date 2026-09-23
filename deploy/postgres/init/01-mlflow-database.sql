-- MLflow gets its own database. It used to share `warehouse`, which put ~23
-- MLflow tables beside the subscriber tables in any database browser - and
-- made "what is in the warehouse" a harder question to answer than it is.
--
-- Postgres runs this only when the data volume is first created. For a volume
-- that already exists, `make mlflow-db` creates the same database idempotently.
CREATE DATABASE mlflow OWNER subscriber;
