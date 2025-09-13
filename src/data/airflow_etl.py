from datetime import datetime
from typing import List, Any


class MockDAG:
    def __init__(self, dag_id: str, schedule_interval: str, start_date: datetime):
        self.dag_id = dag_id
        self.schedule_interval = schedule_interval
        self.start_date = start_date


class MockTask:
    def __init__(self, task_id: str):
        self.task_id = task_id


class StockDataETLPipeline:
    def __init__(self, dag_id: str, schedule_interval: str, start_date: datetime):
        self.dag_id = dag_id
        self.schedule_interval = schedule_interval
        self.start_date = start_date

    def create_dag(self) -> MockDAG:
        return MockDAG(
            dag_id=self.dag_id,
            schedule_interval=self.schedule_interval,
            start_date=self.start_date
        )

    def create_extract_task(self) -> MockTask:
        return MockTask(task_id="extract_stock_data")

    def create_transform_task(self) -> MockTask:
        return MockTask(task_id="transform_stock_data")

    def create_load_task(self) -> MockTask:
        return MockTask(task_id="load_stock_data")

    def create_task_chain(self) -> List[MockTask]:
        return [
            self.create_extract_task(),
            self.create_transform_task(),
            self.create_load_task()
        ]