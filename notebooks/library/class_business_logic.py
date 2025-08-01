import os

# function that joins two pyspark datasets with inner join
def inner_join_dataframes(df1, df2, join_columns):
  return df1.join(df2, join_columns, 'inner')