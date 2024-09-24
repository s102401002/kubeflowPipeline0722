from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import kfp
import kfp.components as comp
from kfp import dsl
from kfp.dsl import Output, Input, Artifact, Model

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pyspark==3.3.1']
)
def load_data(nas_mount_path: str, data_output: Output[Artifact]):
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder.appName("DiabetesPrediction").getOrCreate()
    
    # Read CSV files from the directory
    df = spark.read.csv(nas_mount_path + '/*.csv', header=True, inferSchema=True)
    
    # Define standard name mapping
    standard_name_mapping = {
        'gender': ['gender', 'gen', 'Gender', 'sex', 'Sex'],
        'age': ['age', 'Age', 'AGE'],
        'bmi': ['bmi', 'BMI', 'Bmi'],
        'HbA1c_level': ['HbA1c_level', 'HbA1c', 'hba1c'],
        'blood_glucose_level': ['blood_glucose_level', 'glucose', 'BloodGlucose'],
        'diabetes': ['diabetes', 'Diabetes']
    }

    # Rename columns based on the standard name mapping
    for standard_name, variants in standard_name_mapping.items():
        for variant in variants:
            if variant in df.columns:
                df = df.withColumnRenamed(variant, standard_name)
                break

    # Drop rows where 'diabetes' is 'No Info'
    df = df.filter(df['diabetes'] != 'No Info')

    # Drop rows with missing values
    df = df.dropna(thresh=4)

    # Map gender values to numerical
    df = df.withColumn('gender', when(col('gender') == 'Male', 0)
                                  .when(col('gender') == 'Female', 1)
                                  .otherwise(None))

    # Fill missing values
    df = df.na.fill({
        'age': df.agg(mean('age')).first()[0],
        'bmi': df.agg(mean('bmi')).first()[0],
        'HbA1c_level': df.agg(mean('HbA1c_level')).first()[0],
        'blood_glucose_level': df.agg(mean('blood_glucose_level')).first()[0]
    })

    # Save to CSV
    df.toPandas().to_csv(data_output.path, index=False)
@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pyspark==3.3.1']
)
def prepare_data(
    data_input: Input[Artifact], 
    x_train_output: Output[Artifact], x_test_output: Output[Artifact],
    y_train_output: Output[Artifact], y_test_output: Output[Artifact]
):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml import Pipeline
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
    spark = SparkSession.builder.appName("DiabetesPrediction").getOrCreate()

    # Load data
    df_data = spark.read.csv(data_input.path, header=True, inferSchema=True)

    # Prepare features and labels
    feature_columns = ['gender', 'age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    df_data = assembler.transform(df_data)

    # Split data
    train_data, test_data = df_data.randomSplit([0.8, 0.2], seed=42)

    # Separate features and labels
    train_data = train_data.select('features', 'diabetes')
    test_data = test_data.select('features', 'diabetes')

    # Save to CSV
    train_data.toPandas().to_csv(x_train_output.path, index=False)
    test_data.toPandas().to_csv(x_test_output.path, index=False)
    # Save labels separately
    train_data.select('diabetes').toPandas().to_csv(y_train_output.path, index=False)
    test_data.select('diabetes').toPandas().to_csv(y_test_output.path, index=False)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pyspark==3.3.1']
)
def train_model(x_train: Input[Artifact], y_train: Input[Artifact], train_model_output: Output[Model]):
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml import Pipeline
    import joblib
    
    spark = SparkSession.builder.appName("DiabetesPrediction").getOrCreate()

    # Load data
    x_train_df = spark.read.csv(x_train.path, header=True, inferSchema=True)
    y_train_df = spark.read.csv(y_train.path, header=True, inferSchema=True)

    # Combine features and labels
    train_data = x_train_df.withColumn('label', y_train_df['diabetes'])

    # Train the model
    rf = RandomForestClassifier(featuresCol='features', labelCol='label')
    model = rf.fit(train_data)

    # Save the model
    joblib.dump(model, train_model_output.path)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pyspark==3.3.1']
)
def evaluate_model(model_path: Input[Model], x_test: Input[Artifact], y_test: Input[Artifact], result_output: Output[Artifact]):
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import RandomForestClassificationModel
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    import joblib
    import pandas as pd

    spark = SparkSession.builder.appName("DiabetesPrediction").getOrCreate()

    # Load model
    model = joblib.load(filename=model_path.path)

    # Load test data
    x_test_df = spark.read.csv(x_test.path, header=True, inferSchema=True)
    y_test_df = spark.read.csv(y_test.path, header=True, inferSchema=True)

    # Combine features and labels
    test_data = x_test_df.withColumn('label', y_test_df['diabetes'])

    # Predict and evaluate
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)

    # Write the result to a file
    result_df = pd.DataFrame({'accuracy': [accuracy]})
    result_df.to_csv(result_output.path, index=False)

@dsl.pipeline(
    name='Diabetes Prediction Pipeline with PySpark',
    description='Using PySpark to train and evaluate a diabetes prediction model'
)
def diabetes_prediction_pipeline(nfs_mount_path: str = '/mnt/datasets') -> Output[Artifact]:
    load_data_task = load_data(nas_mount_path=nfs_mount_path)
    
    prepare_data_task = prepare_data(data_input=load_data_task.outputs['data_output'])
    
    train_model_task = train_model(
        x_train=prepare_data_task.outputs['x_train_output'], 
        y_train=prepare_data_task.outputs['y_train_output']
    )
    
    evaluate_task = evaluate_model(
        model_path=train_model_task.outputs['train_model_output'], 
        x_test=prepare_data_task.outputs['x_test_output'], 
        y_test=prepare_data_task.outputs['y_test_output']
    )

    return evaluate_task.outputs['result_output']

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(diabetes_prediction_pipeline, 'diabetes_prediction_pipeline_pyspark.yaml')
