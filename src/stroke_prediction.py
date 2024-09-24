from typing import NewType

import kfp
import kfp.compiler

from kfp import dsl
from kfp.dsl import OutputPath, InputPath, Input, Output, Artifact, Model, Dataset 

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
    df_data_1 = pd.read_csv('https://raw.githubusercontent.com/s102401002/kubeflowPipeline0722/main/stroke.csv')

    # 移除不需要的欄位
    df_data_1 = df_data_1.drop(columns=['id', 'ever_married', 'work_type'])

    # 定義映射
    gender_map = {'Male': 0, 'Female': 1}
    smoking_status_map = {'Unknown': 0, 'never smoked': 0, 'formerly smoked': 1, 'smokes': 1}
    Residence_type_map = {'Urban': 1, 'Rural': 0}

    # 補齊資料
    # gender
    df_data_1 = df_data_1[(df_data_1['gender'] != 'N/A') & (~df_data_1['gender'].isna())]
    df_data_1['gender'] = df_data_1['gender'].map(gender_map)  # map

    # age
    df_data_1 = df_data_1[(df_data_1['age'] != 'N/A') & (~df_data_1['age'].isna())]

    # hypertension
    df_data_1 = df_data_1[(df_data_1['hypertension'] != 'N/A') & (~df_data_1['hypertension'].isna())]

    # heart_disease
    df_data_1 = df_data_1[(df_data_1['heart_disease'] != 'N/A') & (~df_data_1['heart_disease'].isna())]

    # Residence_type
    df_data_1 = df_data_1[(df_data_1['Residence_type'] != 'N/A') & (~df_data_1['Residence_type'].isna())]
    df_data_1['Residence_type'] = df_data_1['Residence_type'].map(Residence_type_map)  # map

    # avg_glucose_level
    df_data_1 = df_data_1[(df_data_1['avg_glucose_level'] != 'N/A') & (~df_data_1['avg_glucose_level'].isna())]

    # bmi
    df_data_1 = df_data_1[(df_data_1['bmi'] != 'N/A') & (~df_data_1['bmi'].isna())]

    # smoking_status
    df_data_1 = df_data_1[(df_data_1['smoking_status'] != 'N/A') & (~df_data_1['smoking_status'].isna())]
    df_data_1['smoking_status'] = df_data_1['smoking_status'].map(smoking_status_map)  # map

    df_data_1 = df_data_1.drop(3116)#特殊處理
    df_data_1 = df_data_1.sample(frac=1).reset_index(drop=True)

    df_data_2 = pd.read_csv('https://raw.githubusercontent.com/s102401002/kubeflowPipeline0722/main/stroke_2.csv')
    df_data_2 = df_data_2.drop(columns=['ever_married', 'work_type'])
    df_data_2.rename(columns={'sex': 'gender'}, inplace=True)
    #合併
    df_data = pd.concat([df_data_1, df_data_2], ignore_index=True)

    # 删除指定的行
    rows_to_delete = [27386, 33816, 40092]
    df_data = df_data.drop(index=rows_to_delete)
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

    X = df_data.drop(labels=['stroke'], axis=1)
    Y = df_data[['stroke']]
    
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
    X_test: Input[Artifact], 
    Y_test: Input[Artifact], 
    model: Output[Artifact],
    file: Output[Artifact]
):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import joblib
    import json
    
    X_train = pd.read_csv(X_train.path)
    Y_train = pd.read_csv(Y_train.path)
    X_val = pd.read_csv(X_val.path)
    Y_val = pd.read_csv(Y_val.path)
    X_test = pd.read_csv(X_test.path)
    Y_test = pd.read_csv(Y_test.path)

    # Logistic Regression
    print('Logistic Regression')
    lr_model = LogisticRegression(random_state=0, max_iter=10000)
    lr_model.fit(X_train, Y_train.values.ravel())
    print('Training accuracy:', lr_model.score(X_train, Y_train))
    lr_accuracy = lr_model.score(X_test, Y_test)
    print('Test accuracy:', lr_accuracy)
    
    # Save the model
    joblib.dump(lr_model, model.path)
    # Save the accuracy
    jsonFile = open(file.path,'w')
    data = {}
    data['accuracy'] = lr_accuracy
    data['model_path'] = model.path
    json.dump(data, jsonFile, indent=2)
@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2', 'xgboost==2.0.3']
)
def train_model_xgboost(
    X_train: Input[Artifact], 
    Y_train: Input[Artifact],
    X_val: Input[Artifact], 
    Y_val: Input[Artifact],
    X_test: Input[Artifact], 
    Y_test: Input[Artifact], 
    model: Output[Artifact],
    file: Output[Artifact]
):
    import pandas as pd
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import joblib
    import json
    
    X_train = pd.read_csv(X_train.path)
    Y_train = pd.read_csv(Y_train.path)
    X_val = pd.read_csv(X_val.path)
    Y_val = pd.read_csv(Y_val.path)
    X_test = pd.read_csv(X_test.path)
    Y_test = pd.read_csv(Y_test.path)

    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dval = xgb.DMatrix(X_val, label=Y_val)
    dtest = xgb.DMatrix(X_test, label=Y_test)

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
    xgb_model = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
    preds = xgb_model.predict(dtest)
    predictions = [round(value) for value in preds]
    global xgb_accuracy 
    xgb_accuracy = accuracy_score(Y_test, predictions)
    print('XGBoost Test accuracy:', xgb_accuracy)
    
    # Save the model
    joblib.dump(xgb_model, model.path)

     # Save the accuracy
    jsonFile = open(file.path,'w')
    data = {}
    data['accuracy'] = xgb_accuracy
    data['model_path'] = model.path
    json.dump(data, jsonFile, indent=2)
    # with open(accuracy.path, 'w') as f:
    #     f.write(str(xgb_accuracy))

# @dsl.component(
#     base_image='python:3.9',
#     packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2']
# )
# def train_model_svm(
#     X_train: Input[Artifact], 
#     Y_train: Input[Artifact],
#     X_val: Input[Artifact], 
#     Y_val: Input[Artifact],
#     X_test: Input[Artifact], 
#     Y_test: Input[Artifact], 
#     model: Output[Artifact],
#     accuracy: Output[Artifact]
# ):
#     import pandas as pd
#     from sklearn.metrics import accuracy_score
#     from sklearn import svm
#     import joblib
    
#     X_train = pd.read_csv(X_train.path)
#     Y_train = pd.read_csv(Y_train.path)
#     X_val = pd.read_csv(X_val.path)
#     Y_val = pd.read_csv(Y_val.path)
#     X_test = pd.read_csv(X_test.path)
#     Y_test = pd.read_csv(Y_test.path)

#     clf=svm.SVC(kernel='poly',gamma='auto',C=100)
#     clf.fit(X_train,Y_train.values.ravel())
#     clf.predict(X_test)
#     svm_accuracy = clf.score(X_test, Y_test)
#     # Save the model
#     joblib.dump(model, clf.path)

#      # Save the accuracy
#     with open(accuracy.path, 'w') as f:
#         f.write(str(svm_accuracy))

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2']
)
def train_model_RandomForest(
    X_train: Input[Artifact], 
    Y_train: Input[Artifact],
    X_val: Input[Artifact], 
    Y_val: Input[Artifact],
    X_test: Input[Artifact], 
    Y_test: Input[Artifact], 
    model: Output[Artifact],
    file: Output[Artifact]
):
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import json
    
    X_train = pd.read_csv(X_train.path)
    Y_train = pd.read_csv(Y_train.path)
    X_val = pd.read_csv(X_val.path)
    Y_val = pd.read_csv(Y_val.path)
    X_test = pd.read_csv(X_test.path)
    Y_test = pd.read_csv(Y_test.path)

    rfc=RandomForestClassifier(n_estimators=5)
    rfc.fit(X_train,Y_train.values.ravel())    
    y_predict=rfc.predict(X_test)
    rfc.predict(X_test)
    rf_accuracy = rfc.score(X_test,Y_test)
    # Save the model
    joblib.dump(rfc, model.path)

     # Save the accuracy
    jsonFile = open(file.path,'w')
    data = {}
    data['accuracy'] = rf_accuracy
    data['model_path'] = model.path
    json.dump(data, jsonFile, indent=2)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2']
)
def train_model_KNN(
    X_train: Input[Artifact], 
    Y_train: Input[Artifact],
    X_val: Input[Artifact], 
    Y_val: Input[Artifact],
    X_test: Input[Artifact], 
    Y_test: Input[Artifact], 
    model: Output[Artifact],
    file: Output[Artifact]
):
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    import joblib
    import json

    X_train = pd.read_csv(X_train.path)
    Y_train = pd.read_csv(Y_train.path)
    X_val = pd.read_csv(X_val.path)
    Y_val = pd.read_csv(Y_val.path)
    X_test = pd.read_csv(X_test.path)
    Y_test = pd.read_csv(Y_test.path)

    knc = KNeighborsClassifier(n_neighbors=3)
    knc.fit(X_train,Y_train.values.ravel())
    knn_accuracy = knc.score(X_test,Y_test)
    # Save the model
    joblib.dump(knc, model.path)

     # Save the accuracy
    jsonFile = open(file.path,'w')
    data = {}
    data['accuracy'] = knn_accuracy
    data['model_path'] = model.path
    json.dump(data, jsonFile, indent=2)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['joblib==1.4.2', 'scikit-learn==1.5.1', 'xgboost==2.0.3']# 
)
def choose_model(
    LogisticRegression_model: Input[Artifact],
    XGBoost_model: Input[Artifact],
    # SVM_model: Input[Artifact],
    RandomForest_model: Input[Artifact],
    KNN_model: Input[Artifact],
    lr_file: Input[Artifact],
    xgb_file: Input[Artifact],
    # svm_accuracy: Input[Artifact],
    rf_file: Input[Artifact],
    knn_file: Input[Artifact],
    final_model: Output[Model],
    result: Output[Artifact]
) -> None:
    import joblib
    import json

    # Define a dictionary to store model artifacts and their corresponding JSON files
    models = {
        'LogisticRegression': lr_file,
        'XGBoost': xgb_file,
        'RandomForest': rf_file,
        'KNN': knn_file
    }

    accuracy = {}
    model_paths = {}

    # Read accuracies and model paths
    for model_name, json_file in models.items():
        with open(json_file.path, 'r') as f:
            data = json.load(f)
        accuracy[model_name] = data['accuracy']
        model_paths[model_name] = data['model_path']

    # Find the best model
    best_model_name = max(accuracy, key=accuracy.get)
    best_model = joblib.load(model_paths[best_model_name])
    
    # Save the best model
    joblib.dump(best_model, final_model.path)

    # Prepare result string
    result_string = f'Best Model is {best_model_name} : {accuracy[best_model_name]}'
    result_string += f'\nAccuracy:\n'
    for model_name, acc in accuracy.items():
        result_string += f'{model_name:17} : {acc}\n'
    print(result_string)


    # Write the result to a file
    with open(result.path, 'w') as f:
        f.write(result_string)
@dsl.pipeline(
    name='Stroke Prediction Pipeline',
    description='Using Kubeflow pipeline to train and evaluate a Stroke prediction model'
)
def Stroke_prediction_pipeline():
    # Load data
    load_data_task = load_data()

    # Prepare data
    prepare_data_task = prepare_data(data_input=load_data_task.outputs['data_output'])
    
    # Train models
    train_model_LogisticRegression_task = train_model_LogisticRegression(
        X_train=prepare_data_task.outputs['X_train_output'], 
        Y_train=prepare_data_task.outputs['Y_train_output'],
        X_val=prepare_data_task.outputs['X_val_output'],
        Y_val=prepare_data_task.outputs['Y_val_output'],
        X_test=prepare_data_task.outputs['X_test_output'], 
        Y_test=prepare_data_task.outputs['Y_test_output']
    )

    train_model_xgboost_task = train_model_xgboost(
        X_train=prepare_data_task.outputs['X_train_output'], 
        Y_train=prepare_data_task.outputs['Y_train_output'],
        X_val=prepare_data_task.outputs['X_val_output'],
        Y_val=prepare_data_task.outputs['Y_val_output'],
        X_test=prepare_data_task.outputs['X_test_output'], 
        Y_test=prepare_data_task.outputs['Y_test_output']
    )
    
    # train_model_svm_task = train_model_svm(
    #     X_train=prepare_data_task.outputs['X_train_output'], 
    #     Y_train=prepare_data_task.outputs['Y_train_output'],
    #     X_val=prepare_data_task.outputs['X_val_output'],
    #     Y_val=prepare_data_task.outputs['Y_val_output'],
    #     X_test=prepare_data_task.outputs['X_test_output'], 
    #     Y_test=prepare_data_task.outputs['Y_test_output']
    # )

    train_model_RandomForest_task = train_model_RandomForest(
        X_train=prepare_data_task.outputs['X_train_output'], 
        Y_train=prepare_data_task.outputs['Y_train_output'],
        X_val=prepare_data_task.outputs['X_val_output'],
        Y_val=prepare_data_task.outputs['Y_val_output'],
        X_test=prepare_data_task.outputs['X_test_output'], 
        Y_test=prepare_data_task.outputs['Y_test_output']
    )

    train_model_KNN_task = train_model_KNN(
        X_train=prepare_data_task.outputs['X_train_output'], 
        Y_train=prepare_data_task.outputs['Y_train_output'],
        X_val=prepare_data_task.outputs['X_val_output'],
        Y_val=prepare_data_task.outputs['Y_val_output'],
        X_test=prepare_data_task.outputs['X_test_output'], 
        Y_test=prepare_data_task.outputs['Y_test_output']
    )

    choose_model_task = choose_model(
        LogisticRegression_model=train_model_LogisticRegression_task.outputs['model'],
        XGBoost_model=train_model_xgboost_task.outputs['model'],
        # SVM_model=train_model_svm_task.outputs['model'],
        RandomForest_model=train_model_RandomForest_task.outputs['model'],
        KNN_model=train_model_KNN_task.outputs['model'],
        lr_file=train_model_LogisticRegression_task.outputs['file'],
        xgb_file=train_model_xgboost_task.outputs['file'],
        # svm_file=train_model_svm_task.outputs['file'],
        rf_file=train_model_RandomForest_task.outputs['file'],
        knn_file=train_model_KNN_task.outputs['file']
    )
    
    # The pipeline doesn't need to return anything explicitly now

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(Stroke_prediction_pipeline, '../Stroke_prediction_pipeline.yaml')