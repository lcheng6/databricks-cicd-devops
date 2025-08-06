import os

import pytest
from unittest.mock import patch
from pyspark.sql import SparkSession
import pyspark as ps
import pandas as pd
import pyspark as ps
import os
import sys

from pyspark.testing import assertDataFrameEqual

# Import the functions to be tested
# from library.fetch_data_from_api import get_class_data_from_api, get_score_data_from_api
sys.path.append(os.path.join(os.path.dirname(__file__), '../../notebooks'))
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
    class_data_path = os.path.join(CURRENT_DIR, "../sample_data/Class_Dataset.csv")
    class_data_path = "file:/Workspace" + class_data_path
    df_class = spark.read.csv(class_data_path, header=True, inferSchema=True)
    return df_class

@pytest.fixture
def mock_successful_score_data(spark) -> ps.sql.DataFrame:
    """Fixture to provide mock score data from the score API"""
    # create the data from sample_data/Score_Dataset.csv
    score_data_path = os.path.join(CURRENT_DIR, "../sample_data/Score_Dataset.csv")
    score_data_path = "file:/Workspace" + score_data_path
    df_score = spark.read.csv(score_data_path, header=True, inferSchema=True)
    return df_score

@pytest.fixture
def mock_successful_joined_data(spark) -> ps.sql.DataFrame:
    """Fixture to provide mock joined data from the class and score APIs"""
    df_joined = spark.read.csv("file:/Workspace/sample_data/Joined_Dataset.csv", header=True, inferSchema=True)
    return df_joined

def test_inner_join_dataframes(spark, mock_successful_class_data, mock_successful_score_data, mock_successful_joined_data):
    """Test the inner_join_dataframes function with mocked data."""
    
    # Act
    df_joined = inner_join_dataframes(mock_successful_class_data, mock_successful_score_data, ['class_id'])
    assert(True)

    assertDataFrameEqual(df_joined, mock_successful_joined_data)

