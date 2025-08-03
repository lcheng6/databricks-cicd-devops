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