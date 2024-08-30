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
    # urls = [
    #     "https://raw.githubusercontent.com/s102401002/kubeflowPipeline0722/main/heart_2020_cleaned.csv"
    # ]
    df_data = pd.read_csv("https://raw.githubusercontent.com/s102401002/kubeflowPipeline0722/main/heart_2020_cleaned.csv")
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
    X_train_output: Output[Artifact], X_test_output: Output[Artifact],
    Y_train_output: Output[Artifact], Y_test_output: Output[Artifact],
    X_val_output: Output[Artifact], Y_val_output: Output[Artifact]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df_data = pd.read_csv(data_input.path)

    X = df_data.drop(labels=['HeartDisease'], axis=1)
    Y = df_data[['HeartDisease']]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)
    X_train.to_csv(X_train_output.path, index=False)
    X_test.to_csv(X_test_output.path, index=False)
    Y_train.to_csv(Y_train_output.path, index=False)
    Y_test.to_csv(Y_test_output.path, index=False)
    X_val.to_csv(X_val_output.path, index=False)
    Y_val.to_csv(Y_val_output.path, index=False)


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2']
)
def train_model_LogisticRegression(
    X_train: Input[Artifact], 
    Y_train: Input[Artifact], 
    X_val: Input[Artifact], 
    Y_val: Input[Artifact],
    train_model_output: Output[Artifact]
):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import joblib
    
    X_train = pd.read_csv(X_train.path)
    Y_train = pd.read_csv(Y_train.path)
    X_val = pd.read_csv(X_val.path)
    Y_val = pd.read_csv(Y_val.path)
    
    # Logistic Regression
    print('Logistic Regression')
    model = LogisticRegression(random_state=0, max_iter=10000)
    model.fit(X_train, Y_train)
    print('Training accuracy:', model.score(X_train, Y_train))
    # lr_accuracy = model.score(X_test, Y_test)
    # print('Test accuracy:', lr_accuracy)
    
    # Save the model
    joblib.dump(model, train_model_output.path)
@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2', 'xgboost==2.0.3']
)
def train_model_xgboost(
    X_train: Input[Artifact], 
    Y_train: Input[Artifact],
    X_val: Input[Artifact], 
    Y_val: Input[Artifact],
    train_model_output: Output[Artifact]
):
    import pandas as pd
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import joblib
    
    X_train = pd.read_csv(X_train.path)
    Y_train = pd.read_csv(Y_train.path)
    X_val = pd.read_csv(X_val.path)
    Y_val = pd.read_csv(Y_val.path)
    
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dval = xgb.DMatrix(X_val, label=Y_val)
    # dtest = xgb.DMatrix(X_test, label=Y_test)

    scale_pos_weight = len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1])
    param = {
        'max_depth': 3,
        'eta': 0.3,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight
    }
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    num_round = 1000
    model = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
    # preds = model.predict(dtest)
    # predictions = [round(value) for value in preds]
    # global xgb_accuracy 
    # xgb_accuracy = accuracy_score(Y_test, predictions)
    # print('XGBoost Test accuracy:', xgb_accuracy)
    
    # Save the model
    joblib.dump(model, train_model_output.path)
    # return f'XGBoost accuracy: {xgb_accuracy}'

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2', 'xgboost==2.0.3']
)
def choose_model(
    X_test: Input[Artifact], 
    Y_test: Input[Artifact], 
    X_val: Input[Artifact], 
    Y_val: Input[Artifact],
    LogisticRegression_model: Input[Artifact],
    XGBoost_model: Input[Artifact]
    # ,final_model: Output[Artifact]
) -> str:
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import RandomForestClassificationModel
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    import joblib
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression

    X_test = pd.read_csv(X_test.path)
    Y_test = pd.read_csv(Y_test.path)

    # LogisticRegression_model accuracy
    lr_accuracy = LogisticRegression_model.score(X_test, Y_test)

    #XGB accuracy
    dtest = xgb.DMatrix(X_test, label=Y_test)
    preds = XGBoost_model.predict(dtest)
    predictions = [round(value) for value in preds]
    global xgb_accuracy 
    xgb_accuracy = accuracy_score(Y_test, predictions)
    # if xgb_accuracy > lr_accuracy:
    #     joblib.dump(XGBoost_model, final_model.path)
    # else:
    #     joblib.dump(LogisticRegression_model, final_model.path)
    print(f'Logistic Regression accuracy: {lr_accuracy}, XGBoost accuracy: {xgb_accuracy}')
    return f'Logistic Regression accuracy: {lr_accuracy}, XGBoost accuracy: {xgb_accuracy}'
    
@dsl.pipeline(
    name='HeartDisease Prediction Pipeline',
    description='Using Kubeflow pipeline to train and evaluate a HeartDisease prediction model'
)
def HeartDisease_prediction_pipeline() -> str:
    # 加載數據
    load_data_task = load_data()

    # 準備數據
    prepare_data_task = prepare_data(data_input=load_data_task.outputs['data_output'])
    
    # 訓練模型
    train_model_LogisticRegression_task = train_model_LogisticRegression(
        X_train=prepare_data_task.outputs['X_train_output'], 
        Y_train=prepare_data_task.outputs['Y_train_output'],
        X_val=prepare_data_task.outputs['X_val_output'],
        Y_val=prepare_data_task.outputs['Y_val_output']
    )

    train_model_xgboost_task = train_model_xgboost(
        X_train=prepare_data_task.outputs['X_train_output'], 
        Y_train=prepare_data_task.outputs['Y_train_output'],
        X_val=prepare_data_task.outputs['X_val_output'],
        Y_val=prepare_data_task.outputs['Y_val_output']
    )
    
    choose_model_task = choose_model(
        X_test=prepare_data_task.outputs['X_test_output'], 
        Y_test=prepare_data_task.outputs['Y_test_output'],
        X_val=prepare_data_task.outputs['X_val_output'],
        Y_val=prepare_data_task.outputs['Y_val_output'],
        LogisticRegression_model=train_model_LogisticRegression_task.outputs['train_model_output'],
        XGBoost_model=train_model_xgboost_task.outputs['train_model_output']
    )
    # 返回模型的準確度
    return choose_model_task.output

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(HeartDisease_prediction_pipeline, 'HeartDisease_prediction_pipeline.yaml')
