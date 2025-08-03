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
from library.class_business_logic import calculate_statistics_with_pandas

class TestCalculateStatisticsWithPandas:
    """Test suite for the calculate_statistics_with_pandas function."""
    
    def test_calculate_statistics_with_pandas_basic_case(self):
        """Test basic functionality with normal data."""
        # Arrange
        data = {
            'class_id': ['C001', 'C001', 'C001', 'C001'],
            'class_name': ['Math', 'Math', 'Math', 'Math'],
            'score': [85.0, 90.0, 78.0, 92.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['score']
        
        # Act
        result = calculate_statistics_with_pandas(pdf, columns)
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        
        # Check score stats
        score_stats = result['score_stats'].iloc[0]
        assert score_stats['score_average'] == 86.25  # (85+90+78+92)/4
        assert score_stats['score_min'] == 78.0
        assert score_stats['score_max'] == 92.0
    
    def test_calculate_statistics_with_pandas_nan_values(self):
        """Test functionality when some values are NaN."""
        # Arrange
        data = {
            'class_id': ['C001', 'C001', 'C001', 'C001'],
            'class_name': ['Math', 'Math', 'Math', 'Math'],
            'score': [85.0, np.nan, 78.0, 92.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['score']
        
        # Act
        result = calculate_statistics_with_pandas(pdf, columns)
        
        # Assert
        score_stats = result['score_stats'].iloc[0]
        assert score_stats['score_average'] == pytest.approx(85.0, rel=1e-5)  # (85+78+92)/3
        assert score_stats['score_min'] == 78.0
        assert score_stats['score_max'] == 92.0
    
    def test_calculate_statistics_with_pandas_all_nan_column(self):
        """Test functionality when an entire column is NaN."""
        # Arrange
        data = {
            'class_id': ['C001', 'C001', 'C001'],
            'class_name': ['Math', 'Math', 'Math'],
            'score': [np.nan, np.nan, np.nan]
        }
        pdf = pd.DataFrame(data)
        columns = ['score']
        
        # Act
        result = calculate_statistics_with_pandas(pdf, columns)
        
        # Assert
        score_stats = result['score_stats'].iloc[0]
        assert score_stats['score_average'] is None
        assert score_stats['score_min'] is None
        assert score_stats['score_max'] is None
    
    def test_calculate_statistics_with_pandas_single_row(self):
        """Test functionality with a single row."""
        # Arrange
        data = {
            'class_id': ['C001'],
            'class_name': ['Math'],
            'score': [85.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['score']
        
        # Act
        result = calculate_statistics_with_pandas(pdf, columns)
        
        # Assert
        score_stats = result['score_stats'].iloc[0]
        assert score_stats['score_average'] == 85.0
        assert score_stats['score_min'] == 85.0
        assert score_stats['score_max'] == 85.0
    
    def test_calculate_statistics_with_pandas_multiple_columns(self):
        """Test functionality with multiple columns."""
        # Arrange
        data = {
            'class_id': ['C001', 'C001', 'C001'],
            'class_name': ['Math', 'Math', 'Math'],
            'score': [85.0, 90.0, 78.0],
            'attendance': [95.0, 90.0, 85.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['score', 'attendance']
        
        # Act
        result = calculate_statistics_with_pandas(pdf, columns)
        
        # Assert
        score_stats = result['score_stats'].iloc[0]
        assert score_stats['score_average'] == pytest.approx(84.333333, rel=1e-5)
        assert score_stats['score_min'] == 78.0
        assert score_stats['score_max'] == 90.0
        
        attendance_stats = result['attendance_stats'].iloc[0]
        assert attendance_stats['attendance_average'] == pytest.approx(90.0, rel=1e-5)
        assert attendance_stats['attendance_min'] == 85.0
        assert attendance_stats['attendance_max'] == 95.0
    
    def test_calculate_statistics_with_pandas_empty_columns_list(self):
        """Test functionality with empty columns list."""
        # Arrange
        data = {
            'class_id': ['C001', 'C001', 'C001'],
            'class_name': ['Math', 'Math', 'Math'],
            'score': [85.0, 90.0, 78.0]
        }
        pdf = pd.DataFrame(data)
        columns = []
        
        # Act
        result = calculate_statistics_with_pandas(pdf, columns)
        
        # Assert
        assert len(result.columns) == 2  # Only class_id and class_name
        assert 'class_id' in result.columns
        assert 'class_name' in result.columns
    
    def test_calculate_statistics_with_pandas_identical_values(self):
        """Test functionality when all values in a column are identical."""
        # Arrange
        data = {
            'class_id': ['C001', 'C001', 'C001'],
            'class_name': ['Math', 'Math', 'Math'],
            'score': [85.0, 85.0, 85.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['score']
        
        # Act
        result = calculate_statistics_with_pandas(pdf, columns)
        
        # Assert
        score_stats = result['score_stats'].iloc[0]
        assert score_stats['score_average'] == 85.0
        assert score_stats['score_min'] == 85.0
        assert score_stats['score_max'] == 85.0
    
    def test_calculate_statistics_with_pandas_large_numbers(self):
        """Test functionality with large numbers."""
        # Arrange
        data = {
            'class_id': ['C001', 'C001', 'C001'],
            'class_name': ['Math', 'Math', 'Math'],
            'score': [1000000.0, 2000000.0, 3000000.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['score']
        
        # Act
        result = calculate_statistics_with_pandas(pdf, columns)
        
        # Assert
        score_stats = result['score_stats'].iloc[0]
        assert score_stats['score_average'] == pytest.approx(2000000.0, rel=1e-5)
        assert score_stats['score_min'] == 1000000.0
        assert score_stats['score_max'] == 3000000.0
    
    def test_calculate_statistics_with_pandas_decimal_precision(self):
        """Test functionality with decimal numbers requiring precision."""
        # Arrange
        data = {
            'class_id': ['C001', 'C001', 'C001'],
            'class_name': ['Math', 'Math', 'Math'],
            'score': [85.123, 90.456, 78.789]
        }
        pdf = pd.DataFrame(data)
        columns = ['score']
        
        # Act
        result = calculate_statistics_with_pandas(pdf, columns)
        
        # Assert
        score_stats = result['score_stats'].iloc[0]
        expected_avg = (85.123 + 90.456 + 78.789) / 3
        assert score_stats['score_average'] == pytest.approx(expected_avg, rel=1e-5)
        assert score_stats['score_min'] == 78.789
        assert score_stats['score_max'] == 90.456
    
    def test_calculate_statistics_with_pandas_column_naming(self):
        """Test that column names are generated correctly."""
        # Arrange
        data = {
            'class_id': ['C001'],
            'class_name': ['Math'],
            'test_score': [85.0],
            'homework_score': [90.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['test_score', 'homework_score']
        
        # Act
        result = calculate_statistics_with_pandas(pdf, columns)
        column_names = result.columns
        
        # Assert
        assert 'test_score_stats' in column_names
        assert 'homework_score_stats' in column_names