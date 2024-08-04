# test

from typing import NewType

import kfp
import kfp.compiler

from kfp import dsl
from kfp.dsl import OutputPath, InputPath, Input, Output, Artifact

from pandas import DataFrame
# from kfp.components import func_to_container_op

DF = NewType('DF', DataFrame)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2']
)
def load_data(data_output: Output[Artifact], nfs_mount_path: str):
    import pandas as pd
    import os

    csv_files = [f for f in os.listdir(nfs_mount_path) if f.endswith('.csv')]
    
    datas = []
    for csv_file in csv_files:
        file_path = os.path.join(nfs_mount_path, csv_file)
        df = pd.read_csv(file_path)
        datas.append(df)

    if not datas:
        raise ValueError(f"No CSV files found in {nfs_mount_path}")

    standard_name_mapping = {
        'gender': ['gender', 'gen', 'Gender', 'sex', 'Sex'],
        'age': ['age', 'Age', 'AGE'],
        'bmi': ['bmi', 'BMI', 'Bmi'],
        'HbA1c_level': ['HbA1c_level', 'HbA1c', 'hba1c'],
        'blood_glucose_level': ['blood_glucose_level', 'glucose', 'BloodGlucose'],
        'diabetes': ['diabetes', 'Diabetes']
    }

    for df in datas:
        for standard_name, variants in standard_name_mapping.items():
            for variant in variants:
                if variant in df.columns:
                    df.rename(columns={variant: standard_name}, inplace=True)
                    break

    df_data = pd.concat(datas, ignore_index=True)
    
    df_data = df_data.drop(df_data[df_data['diabetes'] == 'No Info'].index)
    df_data = df_data[['gender','age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']]
    df_data = df_data.dropna(thresh=4)
    
    gender_map = {'Male': 0 , 'Female': 1  , 'Other': 2}
    df_data['gender'] = df_data['gender'].map(gender_map)
    df_data = df_data[df_data['gender'] != 2]
    df_data['age'] = df_data['age'].replace('No Info', df_data['age'].mean())
    df_data['bmi'] = df_data['bmi'].replace('No Info', df_data['bmi'].mean())
    df_data['HbA1c_level'] = df_data['HbA1c_level'].replace('No Info', df_data['HbA1c_level'].mean())
    df_data['blood_glucose_level'] = df_data['blood_glucose_level'].replace('No Info', df_data['blood_glucose_level'].mean())

    df_data.to_csv(data_output.path, index=False)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1']
)
def prepare_data(
    data_input: Input[Artifact], 
    x_train_output: Output[Artifact], x_test_output: Output[Artifact],
    y_train_output: Output[Artifact], y_test_output: Output[Artifact]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df_data = pd.read_csv(data_input.path)

    x = df_data.drop(labels=['diabetes'], axis=1)
    y = df_data[['diabetes']]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    x_train_df = pd.DataFrame(x_train)
    x_test_df = pd.DataFrame(x_test)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)

    x_train_df.to_csv(x_train_output.path, index=False)
    x_test_df.to_csv(x_test_output.path, index=False)
    y_train_df.to_csv(y_train_output.path, index=False)
    y_test_df.to_csv(y_test_output.path, index=False)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2', 'xgboost==2.0.3']
)
def train_model(x_train: Input[Artifact], y_train: Input[Artifact], train_model_output: Output[Artifact]):
    import pandas as pd
    from xgboost import XGBClassifier
    import joblib
    
    x_train = pd.read_csv(x_train.path)
    y_train = pd.read_csv(y_train.path)
    
    model = XGBClassifier(n_estimators=1000, learning_rate= 0.01)
    model.fit(x_train, y_train.values.ravel())
    
    joblib.dump(model, train_model_output.path)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2', 'xgboost==2.0.3']
)
def evaluate_model(model_path: Input[Artifact], x_test: Input[Artifact], y_test: Input[Artifact]) -> str:
    import pandas as pd
    from sklearn.metrics import accuracy_score
    import joblib

    model = joblib.load(filename=model_path.path)

    x_test_df = pd.read_csv(x_test.path)
    y_test_df = pd.read_csv(y_test.path)
    
    y_pred = model.predict(x_test_df)
    accuracy = accuracy_score(y_test_df, y_pred)
    
    return f'Test accuracy: {accuracy}'

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['joblib==1.4.2']
)
def save_model_to_nfs(model_path: Input[Artifact], nfs_mount_path: str):
    import joblib
    import os

    model = joblib.load(model_path.path)

    nfs_model_path = os.path.join(nfs_mount_path, 'model.joblib')

    joblib.dump(model, nfs_model_path)

    return f"Model saved to NFS at {nfs_model_path}"


@dsl.pipeline(
    name='Diabetes Prediction Pipeline',
    description='Using kubeflow pipeline to train and evaluate a diabetes prediction model'
)
def diabetes_prediction_pipeline(nfs_mount_path: str = '/mnt/nfs') -> str:
    load_data_task = load_data(nfs_mount_path=nfs_mount_path)

    prepare_data_task = prepare_data(data_input=load_data_task.outputs['data_output'])
    
    train_model_task = train_model(
        x_train = prepare_data_task.outputs['x_train_output'], 
        y_train = prepare_data_task.outputs['y_train_output']
    )
    
    evaluate_model_task = evaluate_model(
        model_path = train_model_task.outputs['train_model_output'], 
        x_test = prepare_data_task.outputs['x_test_output'], 
        y_test = prepare_data_task.outputs['y_test_output']
    )
    
    save_model_task = save_model_to_nfs(
        model_path=train_model_task.outputs['train_model_output'],
        nfs_mount_path=nfs_mount_path
    )
    
    return f"{evaluate_model_task.output}\n{save_model_task.output}"

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(diabetes_prediction_pipeline, 'diabetes_prediction_pipeline_xgboost.yaml')