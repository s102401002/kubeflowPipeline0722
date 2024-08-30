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

    # 讀取與程式碼位於同一個資料夾中的 hypertension.csv
    df_data = pd.read_csv('https://raw.githubusercontent.com/s102401002/kubeflowPipeline0722/main/hypertension_data.csv')

    # 移除不需要的欄位
    df_data = df_data.drop(columns=['cp','thal'])


    # 補齊資料
    df_data = df_data[(df_data['age'] != 'N/A') & (~df_data['age'].isna())]
    df_data = df_data[(df_data['sex'] != 'N/A') & (~df_data['sex'].isna())]
    df_data = df_data[(df_data['trestbps'] != 'N/A') & (~df_data['trestbps'].isna())] #靜止血壓
    df_data = df_data[(df_data['chol'] != 'N/A') & (~df_data['chol'].isna())] #血清膽固醇
    df_data = df_data[(df_data['fbs'] != 'N/A') & (~df_data['fbs'].isna())] #是否患者的空腹血糖 > 120
    df_data = df_data[(df_data['restecg'] != 'N/A') & (~df_data['restecg'].isna())] #ST-T 波是否異常
    df_data = df_data[df_data['restecg'] != 2]
    df_data = df_data[(df_data['thalach'] != 'N/A') & (~df_data['thalach'].isna())] #最大心率
    df_data = df_data[(df_data['exang'] != 'N/A') & (~df_data['exang'].isna())] #是否運動誘發心絞痛
    df_data = df_data[(df_data['oldpeak'] != 'N/A') & (~df_data['oldpeak'].isna())] #運動相對於休息引起的 ST 。
    df_data = df_data[(df_data['slope'] != 'N/A') & (~df_data['slope'].isna())] #運動峰值 ST 段斜率：0：上坡 1：平坦 2：下坡
    df_data = df_data[(df_data['ca'] != 'N/A') & (~df_data['ca'].isna())]#螢光檢查著色的主要血管數量 (0–4)
    df_data = df_data[(df_data['target'] != 'N/A') & (~df_data['target'].isna())]
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

    X = df_data.drop(labels=['target'], axis=1)
    Y = df_data[['target']]
    
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
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2', 'xgboost==2.0.3']
)
def train_model(
    X_train: Input[Artifact], 
    Y_train: Input[Artifact], 
    X_test: Input[Artifact], 
    Y_test: Input[Artifact], 
    X_val: Input[Artifact], 
    Y_val: Input[Artifact]
) -> str:
    import pandas as pd
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import joblib
    
    X_train = pd.read_csv(X_train.path)
    Y_train = pd.read_csv(Y_train.path)
    X_test = pd.read_csv(X_test.path)
    Y_test = pd.read_csv(Y_test.path)
    X_val = pd.read_csv(X_val.path)
    Y_val = pd.read_csv(Y_val.path)
    
    #邏輯回歸
    print('Logistic Regression')
    model = LogisticRegression(random_state=0, max_iter=10000)
    model.fit(X_train, Y_train)
    print('Training accuracy:', model.score(X_train, Y_train))
    lr_accuracy = model.score(X_test, Y_test)
    print('Test accuracy:', lr_accuracy)

    #XGBoost
    def XGBoost_training():
        # 將資料轉換為 DMatrix 格式
        dtrain = xgb.DMatrix(X_train, label=Y_train)
        dval = xgb.DMatrix(X_val, label=Y_val)
        dtest = xgb.DMatrix(X_test, label=Y_test)

        # 計算正樣本和負樣本的比例
        scale_pos_weight = len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1])#負除以正
        # 設定參數
        param = {
            'max_depth': 3,  # 樹的最大深度
            'eta': 0.3,      # 學習率
            'objective': 'binary:logistic',  # 目標函數（二分類問題）
            'eval_metric': 'logloss',  # 評估指標
            'scale_pos_weight': scale_pos_weight  # 加權參數
        }
        # 訓練模型
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        num_round = 50 
        Model  = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)#10次不變即停止
        preds = Model .predict(dtest)
        predictions = [round(value) for value in preds]
        global xgb_accuracy 
        xgb_accuracy = accuracy_score(Y_test, predictions)
        print('XGBoost Test accuracy:', xgb_accuracy)

        return Model
    
    Model = XGBoost_training()#用XGboost進行訓練
    return f'Logistic Regression accuracy: {lr_accuracy}, XGBoost accuracy: {xgb_accuracy}'


@dsl.pipeline(
    name='Hypertension Prediction Pipeline',
    description='Using Kubeflow pipeline to train and evaluate a Hypertension prediction model'
)
def Hypertension_prediction_pipeline() -> str:
    # 加載數據
    load_data_task = load_data()

    # 準備數據
    prepare_data_task = prepare_data(data_input=load_data_task.outputs['data_output'])
    
    # 訓練模型
    train_model_task = train_model(
        X_train=prepare_data_task.outputs['X_train_output'], 
        Y_train=prepare_data_task.outputs['Y_train_output'],
        X_test=prepare_data_task.outputs['X_test_output'], 
        Y_test=prepare_data_task.outputs['Y_test_output'],
        X_val=prepare_data_task.outputs['X_val_output'],
        Y_val=prepare_data_task.outputs['Y_val_output'],
        # model_output=prepare_data_task.outputs['model_output']  # 確保從 prepare_data 中正確引用輸出
    )
    
    # 返回模型的準確度
    return train_model_task.output  # 返回 train_model 任務的輸出

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(Hypertension_prediction_pipeline, 'Hypertension_prediction_pipeline.yaml')
