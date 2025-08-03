import pytest
import pandas as pd
import numpy as np
from datetime import date
import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

# Add the library path to sys.path to import the business logic
sys.path.append(os.path.join(os.path.dirname(__file__), '../../notebooks'))
from library.class_business_logic import calculate_statistics_with_sql


spark = SparkSession.builder.appName("TestCalculateStatisticsWithSQL").getOrCreate()

class TestCalculateStatisticsWithSQL:
    """Test suite for the calculate_statistics_with_sql function."""
    
    # @classmethod
    # def setup_class(cls):
    #     """Set up Spark session for testing."""
    #     cls.spark = SparkSession.builder.appName("TestCalculateStatisticsWithSQL2").getOrCreate()
    
    # @classmethod
    # def teardown_class(cls):
    #     """Clean up Spark session after testing."""
    #     cls.spark.stop()
    
    def test_calculate_statistics_with_sql_basic_case(self):
        """Test basic functionality with normal data."""
        # Arrange
        schema = StructType([
            StructField("class_id", StringType(), True),
            StructField("class_name", StringType(), True),
            StructField("score", DoubleType(), True)
        ])
        
        data = [
            ("C001", "Math", 85.0),
            ("C001", "Math", 90.0),
            ("C001", "Math", 78.0),
            ("C001", "Math", 92.0),
            ("C002", "Science", 88.0),
            ("C002", "Science", 95.0),
            ("C002", "Science", 82.0)
        ]
        
        df = spark.createDataFrame(data, schema)
        columns = ['score']
        
        # Act
        result = calculate_statistics_with_sql(df, columns, spark)
        result_collected = result.collect()
        
        # Assert
        assert len(result_collected) == 2  # Two classes
        
        # Find results for each class
        math_result = next(row for row in result_collected if row['class_id'] == 'C001')
        science_result = next(row for row in result_collected if row['class_id'] == 'C002')
        
        # Check Math class statistics
        assert math_result['class_name'] == 'Math'
        assert math_result['score_average'] == pytest.approx(86.25, rel=1e-5)  # (85+90+78+92)/4
        assert math_result['score_min'] == 78.0
        assert math_result['score_max'] == 92.0
        
        # Check Science class statistics
        assert science_result['class_name'] == 'Science'
        assert science_result['score_average'] == pytest.approx(88.333333, rel=1e-5)  # (88+95+82)/3
        assert science_result['score_min'] == 82.0
        assert science_result['score_max'] == 95.0
    
    def test_calculate_statistics_with_sql_multiple_columns(self):
        """Test functionality with multiple columns."""
        # Arrange
        schema = StructType([
            StructField("class_id", StringType(), True),
            StructField("class_name", StringType(), True),
            StructField("score", DoubleType(), True),
            StructField("attendance", DoubleType(), True)
        ])
        
        data = [
            ("C001", "Math", 85.0, 95.0),
            ("C001", "Math", 90.0, 90.0),
            ("C001", "Math", 78.0, 85.0)
        ]
        
        df = spark.createDataFrame(data, schema)
        columns = ['score', 'attendance']
        
        # Act
        result = calculate_statistics_with_sql(df, columns, spark)
        result_row = result.collect()[0]
        
        # Assert
        assert result_row['class_id'] == 'C001'
        assert result_row['class_name'] == 'Math'
        
        # Check score statistics
        assert result_row['score_average'] == pytest.approx(84.333333, rel=1e-5)
        assert result_row['score_min'] == 78.0
        assert result_row['score_max'] == 90.0
        
        # Check attendance statistics
        assert result_row['attendance_average'] == pytest.approx(90.0, rel=1e-5)
        assert result_row['attendance_min'] == 85.0
        assert result_row['attendance_max'] == 95.0
    
    def test_calculate_statistics_with_sql_single_row_per_class(self):
        """Test functionality with single row per class."""
        # Arrange
        schema = StructType([
            StructField("class_id", StringType(), True),
            StructField("class_name", StringType(), True),
            StructField("score", DoubleType(), True)
        ])
        
        data = [
            ("C001", "Math", 85.0),
            ("C002", "Science", 92.0)
        ]
        
        df = spark.createDataFrame(data, schema)
        columns = ['score']
        
        # Act
        result = calculate_statistics_with_sql(df, columns, spark)
        result_collected = result.collect()
        
        # Assert
        assert len(result_collected) == 2
        
        for row in result_collected:
            # When there's only one value, min, max, and average should be the same
            assert row['score_average'] == row['score_min']
            assert row['score_average'] == row['score_max']
    
    def test_calculate_statistics_with_sql_null_values(self):
        """Test functionality with null values in the data."""
        # Arrange
        schema = StructType([
            StructField("class_id", StringType(), True),
            StructField("class_name", StringType(), True),
            StructField("score", DoubleType(), True)
        ])
        
        data = [
            ("C001", "Math", 85.0),
            ("C001", "Math", None),  # NULL value
            ("C001", "Math", 90.0),
            ("C001", "Math", None)   # Another NULL value
        ]
        
        df = spark.createDataFrame(data, schema)
        columns = ['score']
        
        # Act
        result = calculate_statistics_with_sql(df, columns, spark)
        result_row = result.collect()[0]
        
        # Assert
        # SQL should automatically handle NULLs - they should be ignored in calculations
        assert result_row['score_average'] == pytest.approx(87.5, rel=1e-5)  # (85+90)/2
        assert result_row['score_min'] == 85.0
        assert result_row['score_max'] == 90.0
    
    def test_calculate_statistics_with_sql_empty_columns_list(self):
        """Test functionality with empty columns list."""
        # Arrange
        schema = StructType([
            StructField("class_id", StringType(), True),
            StructField("class_name", StringType(), True),
            StructField("score", DoubleType(), True)
        ])
        
        data = [("C001", "Math", 85.0)]
        df = spark.createDataFrame(data, schema)
        columns = []
        
        # Act
        result = calculate_statistics_with_sql(df, columns, spark)
        result_row = result.collect()[0]
        
        # Assert
        # Should only have class_id and class_name columns
        assert result_row['class_id'] == 'C001'
        assert result_row['class_name'] == 'Math'
        # No score statistics should be present
        assert 'score_average' not in result_row.asDict()
    
    def test_calculate_statistics_with_sql_identical_values(self):
        """Test functionality when all values in a column are identical."""
        # Arrange
        schema = StructType([
            StructField("class_id", StringType(), True),
            StructField("class_name", StringType(), True),
            StructField("score", DoubleType(), True)
        ])
        
        data = [
            ("C001", "Math", 85.0),
            ("C001", "Math", 85.0),
            ("C001", "Math", 85.0)
        ]
        
        df = spark.createDataFrame(data, schema)
        columns = ['score']
        
        # Act
        result = calculate_statistics_with_sql(df, columns, spark)
        result_row = result.collect()[0]
        
        # Assert
        assert result_row['score_average'] == 85.0
        assert result_row['score_min'] == 85.0
        assert result_row['score_max'] == 85.0
    
    
    def test_calculate_statistics_with_sql_decimal_precision(self):
        """Test functionality with decimal numbers requiring precision."""
        # Arrange
        schema = StructType([
            StructField("class_id", StringType(), True),
            StructField("class_name", StringType(), True),
            StructField("score", DoubleType(), True)
        ])
        
        data = [
            ("C001", "Math", 85.123),
            ("C001", "Math", 90.456),
            ("C001", "Math", 78.789)
        ]
        
        df = spark.createDataFrame(data, schema)
        columns = ['score']
        
        # Act
        result = calculate_statistics_with_sql(df, columns, spark)
        result_row = result.collect()[0]
        
        # Assert
        expected_avg = (85.123 + 90.456 + 78.789) / 3
        assert result_row['score_average'] == pytest.approx(expected_avg, rel=1e-5)
        assert result_row['score_min'] == 78.789
        assert result_row['score_max'] == 90.456
    
    def test_calculate_statistics_with_sql_column_naming(self):
        """Test that column names are generated correctly."""
        # Arrange
        schema = StructType([
            StructField("class_id", StringType(), True),
            StructField("class_name", StringType(), True),
            StructField("test_score", DoubleType(), True),
            StructField("homework_score", DoubleType(), True)
        ])
        
        data = [("C001", "Math", 85.0, 90.0)]
        df = spark.createDataFrame(data, schema)
        columns = ['test_score', 'homework_score']
        
        # Act
        result = calculate_statistics_with_sql(df, columns, spark)
        result_row = result.collect()[0]
        column_names = result_row.asDict().keys()
        
        # Assert
        assert 'test_score_average' in column_names
        assert 'test_score_min' in column_names
        assert 'test_score_max' in column_names
        assert 'homework_score_average' in column_names
        assert 'homework_score_min' in column_names
        assert 'homework_score_max' in column_names


