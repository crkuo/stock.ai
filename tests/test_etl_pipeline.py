import pytest
from src.data.etl import ETLPipeline


def test_etl_pipeline_can_be_instantiated():
    """Test that ETLPipeline can be created"""
    pipeline = ETLPipeline()
    assert pipeline is not None


def test_etl_pipeline_has_extract_method():
    """Test that ETLPipeline has an extract method"""
    pipeline = ETLPipeline()
    assert hasattr(pipeline, 'extract')


def test_etl_pipeline_has_transform_method():
    """Test that ETLPipeline has a transform method"""
    pipeline = ETLPipeline()
    assert hasattr(pipeline, 'transform')


def test_etl_pipeline_has_load_method():
    """Test that ETLPipeline has a load method"""
    pipeline = ETLPipeline()
    assert hasattr(pipeline, 'load')