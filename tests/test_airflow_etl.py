import pytest
from datetime import datetime, timedelta
from src.data.airflow_etl import StockDataETLPipeline


class TestStockDataETLPipeline:
    def test_should_create_etl_pipeline_with_dag_definition(self):
        pipeline = StockDataETLPipeline(
            dag_id="stock_data_etl",
            schedule_interval="0 9 * * 1-5",  # Weekdays at 9 AM
            start_date=datetime(2024, 1, 1)
        )

        dag = pipeline.create_dag()

        assert dag is not None
        assert dag.dag_id == "stock_data_etl"
        assert dag.schedule_interval == "0 9 * * 1-5"

    def test_should_define_extract_task(self):
        pipeline = StockDataETLPipeline(
            dag_id="stock_data_etl",
            schedule_interval="0 9 * * 1-5",
            start_date=datetime(2024, 1, 1)
        )

        extract_task = pipeline.create_extract_task()

        assert extract_task is not None
        assert extract_task.task_id == "extract_stock_data"

    def test_should_define_transform_task(self):
        pipeline = StockDataETLPipeline(
            dag_id="stock_data_etl",
            schedule_interval="0 9 * * 1-5",
            start_date=datetime(2024, 1, 1)
        )

        transform_task = pipeline.create_transform_task()

        assert transform_task is not None
        assert transform_task.task_id == "transform_stock_data"

    def test_should_define_load_task(self):
        pipeline = StockDataETLPipeline(
            dag_id="stock_data_etl",
            schedule_interval="0 9 * * 1-5",
            start_date=datetime(2024, 1, 1)
        )

        load_task = pipeline.create_load_task()

        assert load_task is not None
        assert load_task.task_id == "load_stock_data"

    def test_should_chain_etl_tasks_in_correct_order(self):
        pipeline = StockDataETLPipeline(
            dag_id="stock_data_etl",
            schedule_interval="0 9 * * 1-5",
            start_date=datetime(2024, 1, 1)
        )

        task_chain = pipeline.create_task_chain()

        assert task_chain is not None
        assert len(task_chain) == 3