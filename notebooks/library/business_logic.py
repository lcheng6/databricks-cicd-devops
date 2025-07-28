import os
import sys
import numpy as np
import pandas as pd

def calculate_statistics(pdf: pd.DataFrame, columns: list):
  '''
  calculate_statistics function 
  input: 
    pdf: pandas dataframe
    columns: list of columns to calculate statistics for
  output: 
    pandas dataframe with statistics
  '''
  result = {"pickup_date": pdf["pickup_date"].iloc[0]}
  for col in columns:
      values = pdf[col].dropna()
      mean = values.mean()
      median = values.median()
      variance = values.var(ddof=0)
      result[f"{col}_stats"] = {
          "mean": float(mean) if not np.isnan(mean) else None,
          "median": float(median) if not np.isnan(median) else None,
          "variance": float(variance) if not np.isnan(variance) else None
      }
  return pd.DataFrame([result])