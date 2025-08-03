import pytest
from unittest.mock import patch
from pyspark.sql import SparkSession
import pandas as pd

# Import the functions to be tested
from library.fetch_data_from_api import get_class_data_from_api, get_score_data_from_api
from library.class_business_logic import inner_join_dataframes


@pytest.fixture(scope="module")
def spark_session():
    """Fixture for creating a Spark session for testing."""
    spark = SparkSession.builder \
        .appName("APIDataFetcherTests") \
        .master("local[*]") \
        .getOrCreate()
    yield spark
    spark.stop()


def test_get_class_data_from_api(spark_session):
    """Test the get_class_data_from_api function with mocked data."""
    # Arrange
    mock_data = pd.DataFrame({
        'class_id': ['C001', 'C002'],
        'class_name': ['Math', 'Science']
    })
    
    with patch('pyspark.sql.DataFrameReader.csv') as mock_read_csv:
        mock_read_csv.return_value = spark_session.createDataFrame(mock_data)
        
        # Act
        df_class = get_class_data_from_api(spark_session)
        result = df_class.collect()
        
        # Assert
        assert len(result) == 2
        assert result[0]['class_id'] == 'C001'
        assert result[0]['class_name'] == 'Math'
        assert result[1]['class_id'] == 'C002'
        assert result[1]['class_name'] == 'Science'


def test_get_score_data_from_api(spark_session):
    """Test the get_score_data_from_api function with mocked data."""
    # Arrange
    mock_data = pd.DataFrame({
        'student_id': ['S001', 'S002'],
        'score': [85, 90]
    })
    
    with patch('pyspark.sql.DataFrameReader.csv') as mock_read_csv:
        mock_read_csv.return_value = spark_session.createDataFrame(mock_data)
        
        # Act
        df_score = get_score_data_from_api(spark_session)
        result = df_score.collect()
        
        # Assert
        assert len(result) == 2
        assert result[0]['student_id'] == 'S001'
        assert result[0]['score'] == 85
        assert result[1]['student_id'] == 'S002'
        assert result[1]['score'] == 90


def test_inner_join_dataframes(spark_session):
    """Test the inner_join_dataframes function with mocked data."""
    # Arrange
    class_data = pd.DataFrame({
        'class_id': ['C001', 'C002'],
        'class_name': ['Math', 'Science']
    })
    score_data = pd.DataFrame({
        'class_id': ['C001', 'C002'],
        'student_id': ['S001', 'S002'],
        'score': [85, 90]
    })
    
    df_class = spark_session.createDataFrame(class_data)
    df_score = spark_session.createDataFrame(score_data)
    
    # Act
    df_joined = inner_join_dataframes(df_class, df_score, ['class_id'])
    result = df_joined.collect()
    
    # Assert
    assert len(result) == 2
    assert result[0]['class_id'] == 'C001'
    assert result[0]['class_name'] == 'Math'
    assert result[0]['student_id'] == 'S001'
    assert result[0]['score'] == 85
    assert result[1]['class_id'] == 'C002'
    assert result[1]['class_name'] == 'Science'
    assert result[1]['student_id'] == 'S002'
    assert result[1]['score'] == 90
