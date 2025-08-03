import os

def get_class_data_from_api(spark):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    class_example_data_path = os.path.join(current_dir, "../data/Class_Dataset.csv")
    class_example_data_path = "file:/Workspace" + class_example_data_path
    print(class_example_data_path)
    df_class = spark.read.csv(class_example_data_path, header=True, inferSchema=True)
    return df_class

def get_score_data_from_api(spark):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    score_example_data_path = os.path.join(current_dir, "../data/Score_Dataset.csv")
    score_example_data_path = "file:/Workspace" + score_example_data_path
    print(score_example_data_path)

    df_score = spark.read.csv(score_example_data_path, header=True, inferSchema=True)
    return df_score