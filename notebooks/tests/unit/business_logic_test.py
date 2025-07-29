import pytest
import pandas as pd
import numpy as np
from datetime import date
import sys
import os

# Add the library path to sys.path to import the business logic
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from library.business_logic import calculate_statistics


class TestCalculateStatistics:
    """Test suite for the calculate_statistics function."""
    
    def test_calculate_statistics_basic_case(self):
        """Test basic functionality with normal data."""
        # Arrange
        data = {
            'pickup_date': [date(2023, 1, 1)] * 5,
            'trip_distance': [1.0, 2.0, 3.0, 4.0, 5.0],
            'fare_amount': [10.0, 15.0, 20.0, 25.0, 30.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['trip_distance', 'fare_amount']
        
        # Act
        result = calculate_statistics(pdf, columns)
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result['pickup_date'].iloc[0] == date(2023, 1, 1)
        
        # Check trip_distance stats
        trip_stats = result['trip_distance_stats'].iloc[0]
        assert trip_stats['mean'] == 3.0
        assert trip_stats['median'] == 3.0
        assert trip_stats['variance'] == 2.0
        
        # Check fare_amount stats
        fare_stats = result['fare_amount_stats'].iloc[0]
        assert fare_stats['mean'] == 20.0
        assert fare_stats['median'] == 20.0
        assert fare_stats['variance'] == 50.0
    
    def test_calculate_statistics_with_nan_values(self):
        """Test functionality when some values are NaN."""
        # Arrange
        data = {
            'pickup_date': [date(2023, 1, 1)] * 5,
            'trip_distance': [1.0, 2.0, np.nan, 4.0, 5.0],
            'fare_amount': [10.0, np.nan, 20.0, np.nan, 30.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['trip_distance', 'fare_amount']
        
        # Act
        result = calculate_statistics(pdf, columns)
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        
        # Check trip_distance stats (should exclude NaN)
        trip_stats = result['trip_distance_stats'].iloc[0]
        assert trip_stats['mean'] == 3.0  # (1+2+4+5)/4
        assert trip_stats['median'] == 3.0  # median of [1,2,4,5]
        assert trip_stats['variance'] == 2.5  # variance of [1,2,4,5] with ddof=0
        
        # Check fare_amount stats (should exclude NaN)
        fare_stats = result['fare_amount_stats'].iloc[0]
        assert fare_stats['mean'] == 20.0  # (10+20+30)/3
        assert fare_stats['median'] == 20.0  # median of [10,20,30]
        assert fare_stats['variance'] == pytest.approx(66.666667, rel=1e-5)  # variance of [10,20,30]
    
    def test_calculate_statistics_all_nan_column(self):
        """Test functionality when an entire column is NaN."""
        # Arrange
        data = {
            'pickup_date': [date(2023, 1, 1)] * 3,
            'trip_distance': [np.nan, np.nan, np.nan],
            'fare_amount': [10.0, 15.0, 20.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['trip_distance', 'fare_amount']
        
        # Act
        result = calculate_statistics(pdf, columns)
        
        # Assert
        trip_stats = result['trip_distance_stats'].iloc[0]
        assert trip_stats['mean'] is None
        assert trip_stats['median'] is None
        assert trip_stats['variance'] is None
        
        fare_stats = result['fare_amount_stats'].iloc[0]
        assert fare_stats['mean'] == 15.0
        assert fare_stats['median'] == 15.0
        assert fare_stats['variance'] == pytest.approx(16.666667, rel=1e-5)
    
    def test_calculate_statistics_single_row(self):
        """Test functionality with a single row."""
        # Arrange
        data = {
            'pickup_date': [date(2023, 1, 1)],
            'trip_distance': [5.0],
            'fare_amount': [25.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['trip_distance', 'fare_amount']
        
        # Act
        result = calculate_statistics(pdf, columns)
        
        # Assert
        trip_stats = result['trip_distance_stats'].iloc[0]
        assert trip_stats['mean'] == 5.0
        assert trip_stats['median'] == 5.0
        assert trip_stats['variance'] == 0.0
        
        fare_stats = result['fare_amount_stats'].iloc[0]
        assert fare_stats['mean'] == 25.0
        assert fare_stats['median'] == 25.0
        assert fare_stats['variance'] == 0.0
    
    def test_calculate_statistics_single_column(self):
        """Test functionality with a single column."""
        # Arrange
        data = {
            'pickup_date': [date(2023, 1, 1)] * 4,
            'trip_distance': [1.0, 2.0, 3.0, 4.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['trip_distance']
        
        # Act
        result = calculate_statistics(pdf, columns)
        
        # Assert
        assert 'trip_distance_stats' in result.columns
        assert 'fare_amount_stats' not in result.columns
        
        trip_stats = result['trip_distance_stats'].iloc[0]
        assert trip_stats['mean'] == 2.5
        assert trip_stats['median'] == 2.5
        assert trip_stats['variance'] == 1.25
    
    def test_calculate_statistics_empty_columns_list(self):
        """Test functionality with empty columns list."""
        # Arrange
        data = {
            'pickup_date': [date(2023, 1, 1)] * 3,
            'trip_distance': [1.0, 2.0, 3.0]
        }
        pdf = pd.DataFrame(data)
        columns = []
        
        # Act
        result = calculate_statistics(pdf, columns)
        
        # Assert
        assert len(result.columns) == 1  # Only pickup_date
        assert 'pickup_date' in result.columns
        assert result['pickup_date'].iloc[0] == date(2023, 1, 1)
    
    def test_calculate_statistics_preserve_pickup_date(self):
        """Test that pickup_date is correctly preserved from the first row."""
        # Arrange
        data = {
            'pickup_date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            'trip_distance': [1.0, 2.0, 3.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['trip_distance']
        
        # Act
        result = calculate_statistics(pdf, columns)
        
        # Assert
        # Should preserve the first pickup_date
        assert result['pickup_date'].iloc[0] == date(2023, 1, 1)
    
    def test_calculate_statistics_data_types(self):
        """Test that the function returns correct data types."""
        # Arrange
        data = {
            'pickup_date': [date(2023, 1, 1)] * 3,
            'trip_distance': [1.0, 2.0, 3.0],
            'fare_amount': [10.0, 15.0, 20.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['trip_distance', 'fare_amount']
        
        # Act
        result = calculate_statistics(pdf, columns)
        
        # Assert
        trip_stats = result['trip_distance_stats'].iloc[0]
        fare_stats = result['fare_amount_stats'].iloc[0]
        
        # Check that all numeric values are float (not numpy types)
        assert isinstance(trip_stats['mean'], float)
        assert isinstance(trip_stats['median'], float)
        assert isinstance(trip_stats['variance'], float)
        assert isinstance(fare_stats['mean'], float)
        assert isinstance(fare_stats['median'], float)
        assert isinstance(fare_stats['variance'], float)
    
    def test_calculate_statistics_with_duplicates(self):
        """Test functionality with duplicate values."""
        # Arrange
        data = {
            'pickup_date': [date(2023, 1, 1)] * 6,
            'trip_distance': [2.0, 2.0, 2.0, 3.0, 3.0, 4.0]
        }
        pdf = pd.DataFrame(data)
        columns = ['trip_distance']
        
        # Act
        result = calculate_statistics(pdf, columns)
        
        # Assert
        trip_stats = result['trip_distance_stats'].iloc[0]
        expected_mean = (2.0 + 2.0 + 2.0 + 3.0 + 3.0 + 4.0) / 6
        expected_median = 2.5  # median of [2,2,2,3,3,4]
        
        assert trip_stats['mean'] == pytest.approx(expected_mean, rel=1e-5)
        assert trip_stats['median'] == expected_median