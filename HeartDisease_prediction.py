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
def load_data(data_output: Output[Artifact]):
    import pandas as pd
    
    # 讀取與程式碼位於同一個資料夾中的 stroke.csv
    df_data = pd.read_csv('heart_2020_cleaned.csv')

    # 移除不需要的欄位
    df_data = df_data.drop(columns=['PhysicalHealth', 'MentalHealth', 'Race' , 'GenHealth'])


    # 定義映射
    HeartDisease_map = {'Yes': 1, 'No': 0}
    Smoking_map = {'Yes': 1, 'No': 0}
    AlcoholDrinking_map = {'Yes': 1, 'No': 0}
    Stroke_map = {'Yes': 1, 'No': 0}
    DiffWalking_map = {'Yes': 1, 'No': 0}
    Sex_map = {'Male': 0, 'Female': 1}
    AgeCategory_map = {
                        '0-4': 0,
                        '5-9': 1,
                        '10-14': 2,
                        '15-17': 3,
                        '18-24': 4,
                        '25-29': 5,
                        '30-34': 6,
                        '35-39': 7,
                        '40-44': 8,
                        '45-49': 9,
                        '50-54': 10,
                        '55-59': 11,
                        '60-64': 12,
                        '65-69': 13,
                        '70-74': 14,
                        '75-79': 15,
                        '80 or older': 16
                    }
    Diabetic_map = {'Yes (during pregnancy)':1 ,'Yes': 1, 'No': 0, 'No, borderline diabetes':0 }
    PhysicalActivity_map = {'Yes': 1, 'No': 0}
    Asthma_map = {'Yes': 1, 'No': 0}
    KidneyDisease_map = {'Yes': 1, 'No': 0}
    SkinCancer_map = {'Yes': 1, 'No': 0} 

    # 補齊資料
    df_data['HeartDisease'] = df_data['HeartDisease'].map(HeartDisease_map)
    df_data['Smoking'] = df_data['Smoking'].map(Smoking_map) 
    df_data['AlcoholDrinking'] = df_data['AlcoholDrinking'].map(AlcoholDrinking_map) 
    df_data['Stroke'] = df_data['Stroke'].map(Stroke_map) 
    df_data['DiffWalking'] = df_data['DiffWalking'].map(DiffWalking_map) 
    df_data['Sex'] = df_data['Sex'].map(Sex_map) 
    df_data['AgeCategory'] = df_data['AgeCategory'].map(AgeCategory_map) 
    df_data['Diabetic'] = df_data['Diabetic'].map(Diabetic_map) 
    df_data['PhysicalActivity'] = df_data['PhysicalActivity'].map(PhysicalActivity_map)
    df_data['Asthma'] = df_data['Asthma'].map(Asthma_map) 
    df_data['KidneyDisease'] = df_data['KidneyDisease'].map(KidneyDisease_map) 
    df_data['SkinCancer'] = df_data['SkinCancer'].map(SkinCancer_map) 

    # 將 'Sex' 和 'AgeCategory' 欄位分別移到 DataFrame 的第一和第二欄
    columns_order = ['Sex', 'AgeCategory'] + [col for col in df_data.columns if col not in ['Sex', 'AgeCategory']]
    df_data = df_data[columns_order]
    df_data.to_csv(data_output.path)

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

    X = df_data.drop(labels=['HeartDisease'], axis=1)
    Y = df_data[['HeartDisease']]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)
    print('X_Training data shape:' , X_train.shape)
    print('X_Validation data shape:' , X_val.shape)
    print('X_Test data shape:' , X_test.shape)
    print('Y_Training data shape:' , Y_train.shape)
    print('Y_Validation data shape:' , Y_val.shape)
    print('Y_Test data shape:' , Y_test.shape)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2']
)
def train_model(x_train: Input[Artifact], y_train: Input[Artifact], train_model_output: Output[Artifact]):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import joblib
    
    x_train = pd.read_csv(x_train.path)
    y_train = pd.read_csv(y_train.path)
    
    model = LogisticRegression(random_state=0, max_iter=10000) # 100 times for test p.s. it is 10000 times in beginning
    model.fit(x_train, y_train)
    
    #model_path = './diabete_prediction_model.pkl'
    joblib.dump(model, train_model_output.path)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2']
)
def evaluate_model(model_path: Input[Artifact], x_test: Input[Artifact], y_test: Input[Artifact]) -> str:
    import pandas as pd
    import sklearn
    import joblib

    model = joblib.load(filename=model_path.path)

    x_test_df = pd.read_csv(x_test.path)
    y_test_df = pd.read_csv(y_test.path)
    
    accuracy = model.score(x_test_df, y_test_df)
    
    return f'Test accuracy: {accuracy}'

@dsl.pipeline(
    name='Diabetes Prediction Pipeline',
    description='Using kubeflow pipeline to train and evaluate a diabetes prediction model'
)
def diabetes_prediction_pipeline() -> str:
    load_data_task = load_data()

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
    
    return evaluate_model_task.output

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(diabetes_prediction_pipeline, 'diabetes_prediction_pipeline.yaml')