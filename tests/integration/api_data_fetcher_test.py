import os

import pytest
from unittest.mock import patch
from pyspark.sql import SparkSession
import pandas as pd
import pyspark.sql as ps

# Import the functions to be tested
from library.fetch_data_from_api import get_class_data_from_api, get_score_data_from_api
from library.class_business_logic import inner_join_dataframes

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
@pytest.fixture
def spark():
    """Fixture for creating a Spark session for testing."""
    spark = SparkSession.builder.getOrCreate()
    return spark

@pytest.fixture
def mock_successful_class_data(spark) -> ps.sql.DataFrame:
    """Fixture to provide mock class data from the class API"""
    # create the data from sample_data/Class_Dataset.csv
    df_class = spark.read.csv("file:/Workspace/sample_data/Class_Dataset.csv", header=True, inferSchema=True)
    return df_class

@pytest.fixture
def mock_successful_score_data(spark) -> ps.sql.DataFrame:
    """Fixture to provide mock score data from the score API"""
    # create the data from sample_data/Score_Dataset.csv
    df_score = spark.read.csv("file:/Workspace/sample_data/Score_Dataset.csv", header=True, inferSchema=True)
    return df_score

@pytest.fixture
def mock_successful_joined_data(spark) -> ps.sql.DataFrame:
    """Fixture to provide mock joined data from the class and score APIs"""
    df_joined = spark.read.csv("file:/Workspace/sample_data/Joined_Dataset.csv", header=True, inferSchema=True)
    return df_joined

def test_inner_join_dataframes(spark, mock_successful_joined_data, mock_successful_class_data, mock_successful_score_data):
    """Test the inner_join_dataframes function with mocked data."""
    
    # Act
    df_joined = inner_join_dataframes(mock_successful_joined_data, mock_successful_class_data, ['class_id'])
    # compare df_joined with mock_successful_joined_data
    assert df_joined.count() == mock_successful_joined_data.count()
    assert df_joined.columns == mock_successful_joined_data.columns

