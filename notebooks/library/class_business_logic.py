import os
import numpy as np
import pandas as pd
import pyspark as ps


# function that joins two pyspark datasets with inner join
def inner_join_dataframes(df1, df2, join_columns):
    return df1.join(df2, join_columns, "inner")


def calculate_statistics_with_sql(
    df: ps.sql.dataframe.DataFrame, columns: list, spark
) -> ps.sql.dataframe.DataFrame:
    """
    calculate_statistics_with_sql function
    input:
      df: pyspark dataframe
      columns: list of columns to calculate statistics for
    output:
      pyspark dataframe with statistics
    """
    result = {
        "class_id": df.first()["class_id"],
        "class_name": df.first()["class_name"],
    }
    columns_to_calculate = []
    for col in columns:
        columns_to_calculate.append(f"avg({col}) as {col}_average")
        columns_to_calculate.append(f"min({col}) as {col}_min")
        columns_to_calculate.append(f"max({col}) as {col}_max")

    df.createOrReplaceTempView("class_score_data")
    sql_command_fragments = [
        """
        select class_id
          , class_name
        """,
        "," if len(columns_to_calculate) > 0 else "",
        ", ".join(columns_to_calculate),
        """
        from class_score_data
        group by class_id, class_name
        """,
    ]
    sql_command = "".join(sql_command_fragments)
    df_result = spark.sql(sql_command)
    return df_result


def calculate_statistics_with_pandas(pdf: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    calculate_statistics_with_pandas function
    input:
      pdf: pandas dataframe
      columns: list of columns to calculate statistics for
    output:
      pandas dataframe with statistics
    """
    result = {
        "class_id": pdf["class_id"].iloc[0],
        "class_name": pdf["class_name"].iloc[0],
    }
    for col in columns:
        values = pdf[col].dropna()
        average = values.mean()  # mean is same as average
        min_val = values.min()
        max_val = values.max()
        variance = values.var(ddof=0)
        result[f"{col}_stats"] = {
            f"{col}_average": float(average) if not np.isnan(average) else None,
            f"{col}_min": float(min_val) if not np.isnan(min_val) else None,
            f"{col}_max": float(max_val) if not np.isnan(max_val) else None,
        }
    return pd.DataFrame([result])