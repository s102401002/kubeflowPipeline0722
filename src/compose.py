from kfp import dsl, compiler, kubernetes, components
from kfp.dsl import Input, Output, Metrics, Dataset, Model, Artifact, component, ContainerSpec,  OutputPath, InputPath

import json


def get_spark_job_definition():
    import yaml
    import time
    # Read manifest file
    with open('spark-job-python-10kprocess.yaml', "r") as stream:
        spark_job_manifest = yaml.safe_load(stream)

    # Add epoch time in the job name
    epoch = int(time.time())
    spark_job_manifest["metadata"]["name"] = spark_job_manifest["metadata"]["name"].format(epoch=epoch)
    return spark_job_manifest


@component(
    base_image='python:3.10-slim', 
    packages_to_install=['pandas']
)
def load_file_from_nas_to_minio(
    x_train_input_path: str, 
    x_test_input_path: str, 
    y_train_input_path: str, 
    y_test_input_path: str, 
    x_train_output: Output[Dataset], 
    x_test_output: Output[Dataset], 
    y_train_output: Output[Dataset], 
    y_test_output: Output[Dataset]
):
    import pandas as pd

    df = pd.read_csv(x_train_input_path)
    df.to_csv(x_train_output.path, index=False)

    df = pd.read_csv(x_test_input_path)
    df.to_csv(x_test_output.path, index=False)

    df = pd.read_csv(y_train_input_path)
    df.to_csv(y_train_output.path, index=False)

    df = pd.read_csv(y_test_input_path)
    df.to_csv(y_test_output.path, index=False)

@component(base_image='python:3.10-slim')
def parse_input_json(
    json_file_path: str, 
    xgboost_input_metrics: Output[Metrics], 
    random_forest_input_metrics: Output[Metrics], 
    knn_input_metrics: Output[Metrics],
    lr_input_metrics: Output[Metrics]
):
    import json

    def log_metric(metrics: Metrics, input_dict: dict):
        for key in input_dict:
            if key == "method":
                continue
            else:
                metrics.log_metric(key, input_dict.get(key))

    with open(file=json_file_path, mode='r', encoding='utf8') as file:
        input_dict_arr: list[dict] = json.load(file)
    
    for input_dict in input_dict_arr:
        if input_dict["method"] == "xgboost":
            log_metric(xgboost_input_metrics, input_dict)
        elif input_dict["method"] == "random_forest":
            log_metric(random_forest_input_metrics, input_dict)
        elif input_dict["method"] == "knn":
            log_metric(knn_input_metrics, input_dict)
        elif input_dict["method"] == "lr":
            log_metric(lr_input_metrics, input_dict)
        else:
            continue

#xgboost_katib
@dsl.component(
    base_image='python:3.10-slim', 
    packages_to_install=[
        'kubeflow-katib==0.17.0'
    ]
)
def run_xgboost_katib_experiment(
    input_params_metrics: Input[Metrics], 
    best_params_metrics: Output[Metrics]
):
    from kubeflow.katib import KatibClient
    from kubernetes.client import V1ObjectMeta
    from kubeflow.katib import V1beta1Experiment
    from kubeflow.katib import V1beta1AlgorithmSpec
    from kubeflow.katib import V1beta1ObjectiveSpec
    from kubeflow.katib import V1beta1FeasibleSpace
    from kubeflow.katib import V1beta1ExperimentSpec
    from kubeflow.katib import V1beta1ObjectiveSpec
    from kubeflow.katib import V1beta1ParameterSpec
    from kubeflow.katib import V1beta1TrialTemplate
    from kubeflow.katib import V1beta1TrialParameterSpec

    from datetime import datetime, timezone, timedelta

    dt_str = datetime.now(timezone(timedelta(hours=8))).strftime("%-Y-%m-%d-%H-%M-%S")

    experiment_name = "xgboost-" + dt_str.replace("_", "-")
    experiment_namespace = input_params_metrics.metadata.get("experiment_namespace")

    if experiment_name is None or experiment_namespace is None:
        raise ValueError("Both experiment_name and experiment namespace needs to be a string!")

    metadata = V1ObjectMeta(
        name=experiment_name, 
        namespace=experiment_namespace
    )

    algorithm_spec = V1beta1AlgorithmSpec(
        algorithm_name="random"
    )

    objective_spec = V1beta1ObjectiveSpec(
        type="maximize",
        goal= 0.99,
        objective_metric_name="accuracy",
    )

    learning_rate_min = input_params_metrics.metadata.get("learning_rate_min")
    learning_rate_max = input_params_metrics.metadata.get("learning_rate_max")
    learning_rate_step = input_params_metrics.metadata.get("learning_rate_step")

    if learning_rate_min is None or learning_rate_max is None or learning_rate_step is None:
        raise ValueError("All learning_rate_min, learning_rate_max and learning_rate_step cannot be null!")

    try:
        learning_rate_min = float(learning_rate_min)
        learning_rate_max = float(learning_rate_max)
        learning_rate_step = float(learning_rate_step)
    except ValueError:
        raise ValueError("All learning_rate_min, learning_rate_max and learning_rate_step needs to be a float!")

    n_estimators_min = input_params_metrics.metadata.get("n_estimators_min")
    n_estimators_max = input_params_metrics.metadata.get("n_estimators_max")
    n_estimators_step = input_params_metrics.metadata.get("n_estimators_step")

    if n_estimators_min is None or n_estimators_max is None or n_estimators_step is None:
        raise ValueError("All n_estimators_min, n_estimators_max and n_estimators_step cannot be null!")

    try:
        n_estimators_min = int(n_estimators_min)
        n_estimators_max = int(n_estimators_max)
        n_estimators_step = int(n_estimators_step)
    except ValueError:
        raise ValueError("All n_estimators_min, n_estimators_max and n_estimators_step needs to be a float!")

    parameters = [
        V1beta1ParameterSpec(
            name="lr",
            parameter_type="double",
            feasible_space=V1beta1FeasibleSpace(
                min=str(learning_rate_min),
                max=str(learning_rate_max), 
                step=str(learning_rate_step)
            ),
        ), 
        V1beta1ParameterSpec(
            name="ne",
            parameter_type="int",
            feasible_space=V1beta1FeasibleSpace(
                min=str(n_estimators_min),
                max=str(n_estimators_max), 
                step=str(n_estimators_step)
            ),
        )
    ]

    docker_image_name = input_params_metrics.metadata.get("docker_image_name")
    if docker_image_name is None:
        raise ValueError("Docker image name cannot be null!")

    random_state = input_params_metrics.metadata.get("random_state")
    if random_state is None:
        random_state = 42
    else:
        try:
            random_state = int(random_state)
        except ValueError:
            raise ValueError("Random state needs to be an int!")
        
    x_train_path = input_params_metrics.metadata.get("x_train_path")
    x_test_path = input_params_metrics.metadata.get("x_test_path")
    y_train_path = input_params_metrics.metadata.get("y_train_path")
    y_test_path = input_params_metrics.metadata.get("y_test_path")

    train_container = {
        "name": "training-container",
        "image": f"docker.io/{docker_image_name}",
        "command": [
            "python3",
            "/opt/xgboost/train.py",
            "--lr=${trialParameters.learningRate}",
            "--ne=${trialParameters.nEstimators}",
            f"--rs={random_state}",
            f"--esp=100000",
            f"--booster=gbtree",
            f"--x_train_path={x_train_path}",
            f"--x_test_path={x_test_path}",
            f"--y_train_path={y_train_path}",
            f"--y_test_path={y_test_path}",
            f"--save_model=false",
            f"--model_folder_path=models"
        ]
    }

    template_spec = {
        "containers": [
            train_container
        ],
        "restartPolicy": "Never"
    }

    volumes = []
    volumeMounts = []

    datasets_from_pvc = input_params_metrics.metadata.get("datasets_from_pvc")
    datasets_pvc_name = input_params_metrics.metadata.get("datasets_pvc_name")
    datasets_pvc_mount_path = input_params_metrics.metadata.get("datasets_pvc_mount_path")

    datasets_from_pvc = bool(datasets_from_pvc)
    
    if datasets_from_pvc is True:
        if datasets_pvc_name is None or datasets_pvc_mount_path is None:
            raise ValueError("Both datasets_pvc_name and datasets_pvc_mount_path cannot be null")

        volumes.append({
            "name": "datasets", 
            "persistentVolumeClaim": {
                "claimName": datasets_pvc_name
            }
        })
        volumeMounts.append({
            "name": "datasets", 
            "mountPath": datasets_pvc_mount_path
        })

    '''
    if save_model is True:
        volumes.append({
            "name": "models", 
            "persistentVolumeClaim": {
                "claimName": models_pvc_name
            }
        })
        volumeMounts.append({
            "name": "models", 
            "mountPath": "/opt/xgboost/models"
        })

    if datasets_from_pvc is True or save_model is True:
        train_container["volumeMounts"] = volumeMounts
        template_spec["volumes"] = volumes
    '''
    if datasets_from_pvc is True:
        train_container["volumeMounts"] = volumeMounts
        template_spec["volumes"] = volumes

    trial_spec={
        "apiVersion": "batch/v1",
        "kind": "Job",
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "sidecar.istio.io/inject": "false"
                    }
                },
                "spec": template_spec
            }
        }
    }

    trial_template=V1beta1TrialTemplate(
        primary_container_name="training-container",
        trial_parameters=[
            V1beta1TrialParameterSpec(
                name="learningRate",
                description="Learning rate for the training model",
                reference="lr"
            ), 
            V1beta1TrialParameterSpec(
                name="nEstimators",
                description="N estimators for the training model",
                reference="ne"
            )
        ],
        trial_spec=trial_spec,
        retain=True
    )

    max_trial_counts = input_params_metrics.metadata.get("max_trial_counts")
    max_failed_trial_counts = input_params_metrics.metadata.get("max_failed_trial_counts")
    parallel_trial_counts = input_params_metrics.metadata.get("parallel_trial_counts")

    if max_failed_trial_counts is None or max_failed_trial_counts is None or parallel_trial_counts is None:
        raise ValueError("All max_trial_counts, max_failed_trial_counts and parallel_trial_counts cannot be null!")
    
    try:
        max_trial_counts = int(max_trial_counts)
        max_failed_trial_counts = int(max_failed_trial_counts)
        parallel_trial_counts = int(parallel_trial_counts)
    except ValueError:
        raise ValueError("All max_trial_counts, max_failed_trial_counts and needs to be an int!")

    experiment = V1beta1Experiment(
        api_version="kubeflow.org/v1beta1",
        kind="Experiment",
        metadata=metadata,
        spec=V1beta1ExperimentSpec(
            max_trial_count=max_trial_counts,
            parallel_trial_count=parallel_trial_counts,
            max_failed_trial_count=max_failed_trial_counts,
            algorithm=algorithm_spec,
            objective=objective_spec,
            parameters=parameters,
            trial_template=trial_template,
        )
    )

    client_namespace = input_params_metrics.metadata.get("client_namespace")
    if client_namespace is None:
        raise ValueError("Client namespace cannot be null!")

    client = KatibClient(namespace=client_namespace)
    client.create_experiment(experiment=experiment)
    client.wait_for_experiment_condition(name=experiment_name, namespace=experiment_namespace, timeout=3600)

    result = client.get_optimal_hyperparameters(name=experiment_name, namespace=experiment_namespace).to_dict()

    best_params_list = result["parameter_assignments"]

    for params in best_params_list:
        name = params["name"]
        value = params["value"]

        if name == "lr":
            value = float(value)
        elif name == "ne":
            value = int(value)
            
        best_params_metrics.log_metric(metric=name, value=value)

'''
==================================
This is for seperating the code.
Don't remove it.
Thx!

Random forest katib below.
==================================
'''

#random_forest_katib
@dsl.component(
    base_image='python:3.10-slim', 
    packages_to_install=[
        'kubeflow-katib==0.17.0'
    ]
)
def run_random_forest_katib_experiment(
    input_params_metrics: Input[Metrics], 
    best_params_metrics: Output[Metrics]
):
    from kubeflow.katib import KatibClient
    from kubernetes.client import V1ObjectMeta
    from kubeflow.katib import V1beta1Experiment
    from kubeflow.katib import V1beta1AlgorithmSpec
    from kubeflow.katib import V1beta1ObjectiveSpec
    from kubeflow.katib import V1beta1FeasibleSpace
    from kubeflow.katib import V1beta1ExperimentSpec
    from kubeflow.katib import V1beta1ObjectiveSpec
    from kubeflow.katib import V1beta1ParameterSpec
    from kubeflow.katib import V1beta1TrialTemplate
    from kubeflow.katib import V1beta1TrialParameterSpec

    from datetime import datetime, timezone, timedelta

    dt_str = datetime.now(timezone(timedelta(hours=8))).strftime("%-Y-%m-%d-%H-%M-%S")

    experiment_name = "random-forest-" + dt_str.replace("_", "-")
    experiment_namespace = input_params_metrics.metadata.get("experiment_namespace")

    if experiment_name is None or experiment_namespace is None:
        raise ValueError("Both experiment_name and experiment namespace needs to be a string!")

    metadata = V1ObjectMeta(
        name=experiment_name, 
        namespace=experiment_namespace
    )

    algorithm_spec = V1beta1AlgorithmSpec(
        algorithm_name="random"
    )

    objective_spec = V1beta1ObjectiveSpec(
        type="maximize",
        goal= 0.99,
        objective_metric_name="accuracy",
    )

    n_estimators_min = input_params_metrics.metadata.get("n_estimators_min")
    n_estimators_max = input_params_metrics.metadata.get("n_estimators_max")
    n_estimators_step = input_params_metrics.metadata.get("n_estimators_step")

    if n_estimators_min is None or n_estimators_max is None or n_estimators_step is None:
        raise ValueError("All n_estimators_min, n_estimators_max and n_estimators_step cannot be null!")

    try:
        n_estimators_min = int(n_estimators_min)
        n_estimators_max = int(n_estimators_max)
        n_estimators_step = int(n_estimators_step)
    except ValueError:
        raise ValueError("All n_estimators_min, n_estimators_max and n_estimators_step needs to be a float!")

    parameters = [
        V1beta1ParameterSpec(
            name="ne",
            parameter_type="int",
            feasible_space=V1beta1FeasibleSpace(
                min=str(n_estimators_min),
                max=str(n_estimators_max), 
                step=str(n_estimators_step)
            ),
        )
    ]

    docker_image_name = input_params_metrics.metadata.get("docker_image_name")
    if docker_image_name is None:
        raise ValueError("Docker image name cannot be null!")

    random_state = input_params_metrics.metadata.get("random_state")
    if random_state is None:
        random_state = 42
    else:
        try:
            random_state = int(random_state)
        except ValueError:
            raise ValueError("Random state needs to be an int!")
        
    x_train_path = input_params_metrics.metadata.get("x_train_path")
    x_test_path = input_params_metrics.metadata.get("x_test_path")
    y_train_path = input_params_metrics.metadata.get("y_train_path")
    y_test_path = input_params_metrics.metadata.get("y_test_path")

    train_container = {
        "name": "training-container",
        "image": f"docker.io/{docker_image_name}",
        "command": [
            "python3",
            "/opt/rfc/train.py",
            "--ne=${trialParameters.nEstimators}",
            f"--rs={random_state}",
            f"--x_train_path={x_train_path}",
            f"--x_test_path={x_test_path}",
            f"--y_train_path={y_train_path}",
            f"--y_test_path={y_test_path}",
            f"--save_model=false",
            f"--model_folder_path=models"
        ]
    }

    template_spec = {
        "containers": [
            train_container
        ],
        "restartPolicy": "Never"
    }

    volumes = []
    volumeMounts = []

    datasets_from_pvc = input_params_metrics.metadata.get("datasets_from_pvc")
    datasets_pvc_name = input_params_metrics.metadata.get("datasets_pvc_name")
    datasets_pvc_mount_path = input_params_metrics.metadata.get("datasets_pvc_mount_path")

    datasets_from_pvc = bool(datasets_from_pvc)
    
    if datasets_from_pvc is True:
        if datasets_pvc_name is None or datasets_pvc_mount_path is None:
            raise ValueError("Both datasets_pvc_name and datasets_pvc_mount_path cannot be null")

        volumes.append({
            "name": "datasets", 
            "persistentVolumeClaim": {
                "claimName": datasets_pvc_name
            }
        })
        volumeMounts.append({
            "name": "datasets", 
            "mountPath": datasets_pvc_mount_path
        })

    '''
    if save_model is True:
        volumes.append({
            "name": "models", 
            "persistentVolumeClaim": {
                "claimName": models_pvc_name
            }
        })
        volumeMounts.append({
            "name": "models", 
            "mountPath": "/opt/rfc/models"
        })

    if datasets_from_pvc is True or save_model is True:
        train_container["volumeMounts"] = volumeMounts
        template_spec["volumes"] = volumes
    '''
    if datasets_from_pvc is True:
        train_container["volumeMounts"] = volumeMounts
        template_spec["volumes"] = volumes

    trial_spec={
        "apiVersion": "batch/v1",
        "kind": "Job",
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "sidecar.istio.io/inject": "false"
                    }
                },
                "spec": template_spec
            }
        }
    }

    trial_template=V1beta1TrialTemplate(
        primary_container_name="training-container",
        trial_parameters=[
            V1beta1TrialParameterSpec(
                name="nEstimators",
                description="N estimators for the training model",
                reference="ne"
            )
        ],
        trial_spec=trial_spec,
        retain=True
    )

    max_trial_counts = input_params_metrics.metadata.get("max_trial_counts")
    max_failed_trial_counts = input_params_metrics.metadata.get("max_failed_trial_counts")
    parallel_trial_counts = input_params_metrics.metadata.get("parallel_trial_counts")

    if max_failed_trial_counts is None or max_failed_trial_counts is None or parallel_trial_counts is None:
        raise ValueError("All max_trial_counts, max_failed_trial_counts and parallel_trial_counts cannot be null!")
    
    try:
        max_trial_counts = int(max_trial_counts)
        max_failed_trial_counts = int(max_failed_trial_counts)
        parallel_trial_counts = int(parallel_trial_counts)
    except ValueError:
        raise ValueError("All max_trial_counts, max_failed_trial_counts and needs to be an int!")

    experiment = V1beta1Experiment(
        api_version="kubeflow.org/v1beta1",
        kind="Experiment",
        metadata=metadata,
        spec=V1beta1ExperimentSpec(
            max_trial_count=max_trial_counts,
            parallel_trial_count=parallel_trial_counts,
            max_failed_trial_count=max_failed_trial_counts,
            algorithm=algorithm_spec,
            objective=objective_spec,
            parameters=parameters,
            trial_template=trial_template,
        )
    )

    client_namespace = input_params_metrics.metadata.get("client_namespace")
    if client_namespace is None:
        raise ValueError("Client namespace cannot be null!")

    client = KatibClient(namespace=client_namespace)
    client.create_experiment(experiment=experiment)
    client.wait_for_experiment_condition(name=experiment_name, namespace=experiment_namespace, timeout=3600)

    result = client.get_optimal_hyperparameters(name=experiment_name, namespace=experiment_namespace).to_dict()

    best_params_list = result["parameter_assignments"]

    for params in best_params_list:
        name = params["name"]
        value = params["value"]

        if name == "ne":
            value = int(value)
            
        best_params_metrics.log_metric(metric=name, value=value)

'''
==================================
This is for seperating the code.
Don't remove it.
Thx!

KNN katib below.
==================================
'''

#knn_katib
@dsl.component(
    base_image='python:3.10-slim', 
    packages_to_install=[
        'kubeflow-katib==0.17.0'
    ]
)
def run_knn_katib_experiment(
    input_params_metrics: Input[Metrics], 
    best_params_metrics: Output[Metrics]
):
    from kubeflow.katib import KatibClient
    from kubernetes.client import V1ObjectMeta
    from kubeflow.katib import V1beta1Experiment
    from kubeflow.katib import V1beta1AlgorithmSpec
    from kubeflow.katib import V1beta1ObjectiveSpec
    from kubeflow.katib import V1beta1FeasibleSpace
    from kubeflow.katib import V1beta1ExperimentSpec
    from kubeflow.katib import V1beta1ObjectiveSpec
    from kubeflow.katib import V1beta1ParameterSpec
    from kubeflow.katib import V1beta1TrialTemplate
    from kubeflow.katib import V1beta1TrialParameterSpec

    from datetime import datetime, timezone, timedelta

    dt_str = datetime.now(timezone(timedelta(hours=8))).strftime("%-Y-%m-%d-%H-%M-%S")

    experiment_name = "knn-" + dt_str.replace("_", "-")
    experiment_namespace = input_params_metrics.metadata.get("experiment_namespace")

    if experiment_name is None or experiment_namespace is None:
        raise ValueError("Both experiment_name and experiment namespace needs to be a string!")

    metadata = V1ObjectMeta(
        name=experiment_name, 
        namespace=experiment_namespace
    )

    algorithm_spec = V1beta1AlgorithmSpec(
        algorithm_name="random"
    )

    objective_spec = V1beta1ObjectiveSpec(
        type="maximize",
        goal= 0.99,
        objective_metric_name="accuracy",
    )

    n_neighbors_min = input_params_metrics.metadata.get("n_neighbors_min")
    n_neighbors_max = input_params_metrics.metadata.get("n_neighbors_max")
    n_neighbors_step = input_params_metrics.metadata.get("n_neighbors_step")

    if n_neighbors_min is None or n_neighbors_max is None or n_neighbors_step is None:
        raise ValueError("All n_neighbors_min, n_neighbors_max and n_neighbors_step cannot be null!")

    try:
        n_neighbors_min = int(n_neighbors_min)
        n_neighbors_max = int(n_neighbors_max)
        n_neighbors_step = int(n_neighbors_step)
    except ValueError:
        raise ValueError("All n_neighbors_min, n_neighbors_max and n_neighbors_step needs to be a int!")
    
    if n_neighbors_min % 2 != 1 or n_neighbors_max % 2 != 1 or n_neighbors_step % 2 != 0:
        raise ValueError("N neighbors needs to be an odd number!")

    parameters = [
        V1beta1ParameterSpec(
            name="nn",
            parameter_type="int",
            feasible_space=V1beta1FeasibleSpace(
                min=str(n_neighbors_min),
                max=str(n_neighbors_max), 
                step=str(n_neighbors_step)
            )
        )
    ]

    docker_image_name = input_params_metrics.metadata.get("docker_image_name")
    if docker_image_name is None:
        raise ValueError("Docker image name cannot be null!")

    random_state = input_params_metrics.metadata.get("random_state")
    if random_state is None:
        random_state = 42
    else:
        try:
            random_state = int(random_state)
        except ValueError:
            raise ValueError("Random state needs to be an int!")
        
    x_train_path = input_params_metrics.metadata.get("x_train_path")
    x_test_path = input_params_metrics.metadata.get("x_test_path")
    y_train_path = input_params_metrics.metadata.get("y_train_path")
    y_test_path = input_params_metrics.metadata.get("y_test_path")

    train_container = {
        "name": "training-container",
        "image": f"docker.io/{docker_image_name}",
        "command": [
            "python3",
            "/opt/knn/train.py",
            "--nn=${trialParameters.nNeighbors}",
            f"--rs={random_state}",
            f"--x_train_path={x_train_path}",
            f"--x_test_path={x_test_path}",
            f"--y_train_path={y_train_path}",
            f"--y_test_path={y_test_path}",
            f"--save_model=false",
            f"--model_folder_path=models"
        ]
    }

    template_spec = {
        "containers": [
            train_container
        ],
        "restartPolicy": "Never"
    }

    volumes = []
    volumeMounts = []

    datasets_from_pvc = input_params_metrics.metadata.get("datasets_from_pvc")
    datasets_pvc_name = input_params_metrics.metadata.get("datasets_pvc_name")
    datasets_pvc_mount_path = input_params_metrics.metadata.get("datasets_pvc_mount_path")

    datasets_from_pvc = bool(datasets_from_pvc)
    
    if datasets_from_pvc is True:
        if datasets_pvc_name is None or datasets_pvc_mount_path is None:
            raise ValueError("Both datasets_pvc_name and datasets_pvc_mount_path cannot be null")

        volumes.append({
            "name": "datasets", 
            "persistentVolumeClaim": {
                "claimName": datasets_pvc_name
            }
        })
        volumeMounts.append({
            "name": "datasets", 
            "mountPath": datasets_pvc_mount_path
        })

    '''
    if save_model is True:
        volumes.append({
            "name": "models", 
            "persistentVolumeClaim": {
                "claimName": models_pvc_name
            }
        })
        volumeMounts.append({
            "name": "models", 
            "mountPath": "/opt/rfc/models"
        })

    if datasets_from_pvc is True or save_model is True:
        train_container["volumeMounts"] = volumeMounts
        template_spec["volumes"] = volumes
    '''
    if datasets_from_pvc is True:
        train_container["volumeMounts"] = volumeMounts
        template_spec["volumes"] = volumes

    trial_spec={
        "apiVersion": "batch/v1",
        "kind": "Job",
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "sidecar.istio.io/inject": "false"
                    }
                },
                "spec": template_spec
            }
        }
    }

    trial_template=V1beta1TrialTemplate(
        primary_container_name="training-container",
        trial_parameters=[
            V1beta1TrialParameterSpec(
                name="nNeighbors",
                description="N neighbors for the training model",
                reference="nn"
            )
        ],
        trial_spec=trial_spec,
        retain=True
    )

    max_trial_counts = input_params_metrics.metadata.get("max_trial_counts")
    max_failed_trial_counts = input_params_metrics.metadata.get("max_failed_trial_counts")
    parallel_trial_counts = input_params_metrics.metadata.get("parallel_trial_counts")

    if max_failed_trial_counts is None or max_failed_trial_counts is None or parallel_trial_counts is None:
        raise ValueError("All max_trial_counts, max_failed_trial_counts and parallel_trial_counts cannot be null!")
    
    try:
        max_trial_counts = int(max_trial_counts)
        max_failed_trial_counts = int(max_failed_trial_counts)
        parallel_trial_counts = int(parallel_trial_counts)
    except ValueError:
        raise ValueError("All max_trial_counts, max_failed_trial_counts and needs to be an int!")

    experiment = V1beta1Experiment(
        api_version="kubeflow.org/v1beta1",
        kind="Experiment",
        metadata=metadata,
        spec=V1beta1ExperimentSpec(
            max_trial_count=max_trial_counts,
            parallel_trial_count=parallel_trial_counts,
            max_failed_trial_count=max_failed_trial_counts,
            algorithm=algorithm_spec,
            objective=objective_spec,
            parameters=parameters,
            trial_template=trial_template,
        )
    )

    client_namespace = input_params_metrics.metadata.get("client_namespace")
    if client_namespace is None:
        raise ValueError("Client namespace cannot be null!")

    client = KatibClient(namespace=client_namespace)
    client.create_experiment(experiment=experiment)
    client.wait_for_experiment_condition(name=experiment_name, namespace=experiment_namespace, timeout=3600)

    result = client.get_optimal_hyperparameters(name=experiment_name, namespace=experiment_namespace).to_dict()

    best_params_list = result["parameter_assignments"]

    for params in best_params_list:
        name = params["name"]
        value = params["value"]

        if name == "nn":
            value = int(value)
            
        best_params_metrics.log_metric(metric=name, value=value)

'''
==================================
This is for seperating the code.
Don't remove it.
Thx!

Logistic Regression katib below.
==================================
'''

#Logistic Regression_katib
@dsl.component(
    base_image='python:3.10-slim', 
    packages_to_install=[
        'kubeflow-katib==0.17.0'
    ]
)
def run_lr_katib_experiment(
    input_params_metrics: Input[Metrics], 
    best_params_metrics: Output[Metrics]
):
    from kubeflow.katib import KatibClient
    from kubernetes.client import V1ObjectMeta
    from kubeflow.katib import V1beta1Experiment
    from kubeflow.katib import V1beta1AlgorithmSpec
    from kubeflow.katib import V1beta1ObjectiveSpec
    from kubeflow.katib import V1beta1FeasibleSpace
    from kubeflow.katib import V1beta1ExperimentSpec
    from kubeflow.katib import V1beta1ObjectiveSpec
    from kubeflow.katib import V1beta1ParameterSpec
    from kubeflow.katib import V1beta1TrialTemplate
    from kubeflow.katib import V1beta1TrialParameterSpec

    from datetime import datetime, timezone, timedelta

    dt_str = datetime.now(timezone(timedelta(hours=8))).strftime("%-Y-%m-%d-%H-%M-%S")

    experiment_name = "lr-" + dt_str.replace("_", "-")
    experiment_namespace = input_params_metrics.metadata.get("experiment_namespace")

    if experiment_name is None or experiment_namespace is None:
        raise ValueError("Both experiment_name and experiment namespace needs to be a string!")

    metadata = V1ObjectMeta(
        name=experiment_name, 
        namespace=experiment_namespace
    )

    algorithm_spec = V1beta1AlgorithmSpec(
        algorithm_name="random"
    )

    objective_spec = V1beta1ObjectiveSpec(
        type="maximize",
        goal= 0.99,
        objective_metric_name="accuracy",
    )

    iterators_min = input_params_metrics.metadata.get("iterators_min")
    iterators_max = input_params_metrics.metadata.get("iterators_max")
    iterators_step = input_params_metrics.metadata.get("iterators_step")

    if iterators_min is None or iterators_max is None or iterators_step is None:
        raise ValueError("All iterators_min, iterators_max and iterators_step cannot be null!")

    try:
        iterators_min = int(iterators_min)
        iterators_max = int(iterators_max)
        iterators_step = int(iterators_step)
    except ValueError:
        raise ValueError("All iterators_min, iterators_max and iterators_step needs to be a int!")

    parameters = [
        V1beta1ParameterSpec(
            name="it",
            parameter_type="int",
            feasible_space=V1beta1FeasibleSpace(
                min=str(iterators_min),
                max=str(iterators_max), 
                step=str(iterators_step)
            )
        )
    ]

    docker_image_name = input_params_metrics.metadata.get("docker_image_name")
    if docker_image_name is None:
        raise ValueError("Docker image name cannot be null!")

    random_state = input_params_metrics.metadata.get("random_state")
    if random_state is None:
        random_state = 42
    else:
        try:
            random_state = int(random_state)
        except ValueError:
            raise ValueError("Random state needs to be an int!")
        
    x_train_path = input_params_metrics.metadata.get("x_train_path")
    x_test_path = input_params_metrics.metadata.get("x_test_path")
    y_train_path = input_params_metrics.metadata.get("y_train_path")
    y_test_path = input_params_metrics.metadata.get("y_test_path")

    train_container = {
        "name": "training-container",
        "image": f"docker.io/{docker_image_name}",
        "command": [
            "python3",
            "/opt/lr/train.py",
            "--it=${trialParameters.iterators}",
            f"--rs={random_state}",
            f"--x_train_path={x_train_path}",
            f"--x_test_path={x_test_path}",
            f"--y_train_path={y_train_path}",
            f"--y_test_path={y_test_path}",
            f"--save_model=false",
            f"--model_folder_path=models"
        ]
    }

    template_spec = {
        "containers": [
            train_container
        ],
        "restartPolicy": "Never"
    }

    volumes = []
    volumeMounts = []

    datasets_from_pvc = input_params_metrics.metadata.get("datasets_from_pvc")
    datasets_pvc_name = input_params_metrics.metadata.get("datasets_pvc_name")
    datasets_pvc_mount_path = input_params_metrics.metadata.get("datasets_pvc_mount_path")

    datasets_from_pvc = bool(datasets_from_pvc)
    
    if datasets_from_pvc is True:
        if datasets_pvc_name is None or datasets_pvc_mount_path is None:
            raise ValueError("Both datasets_pvc_name and datasets_pvc_mount_path cannot be null")

        volumes.append({
            "name": "datasets", 
            "persistentVolumeClaim": {
                "claimName": datasets_pvc_name
            }
        })
        volumeMounts.append({
            "name": "datasets", 
            "mountPath": datasets_pvc_mount_path
        })

    '''
    if save_model is True:
        volumes.append({
            "name": "models", 
            "persistentVolumeClaim": {
                "claimName": models_pvc_name
            }
        })
        volumeMounts.append({
            "name": "models", 
            "mountPath": "/opt/lr/models"
        })
    '''

    if datasets_from_pvc is True:
        train_container["volumeMounts"] = volumeMounts
        template_spec["volumes"] = volumes

    trial_spec={
        "apiVersion": "batch/v1",
        "kind": "Job",
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "sidecar.istio.io/inject": "false"
                    }
                },
                "spec": template_spec
            }
        }
    }

    trial_template=V1beta1TrialTemplate(
        primary_container_name="training-container",
        trial_parameters=[
            V1beta1TrialParameterSpec(
                name="iterators",
                description="iterators for the training model",
                reference="it"
            )
        ],
        trial_spec=trial_spec,
        retain=True
    )

    max_trial_counts = input_params_metrics.metadata.get("max_trial_counts")
    max_failed_trial_counts = input_params_metrics.metadata.get("max_failed_trial_counts")
    parallel_trial_counts = input_params_metrics.metadata.get("parallel_trial_counts")

    if max_failed_trial_counts is None or max_failed_trial_counts is None or parallel_trial_counts is None:
        raise ValueError("All max_trial_counts, max_failed_trial_counts and parallel_trial_counts cannot be null!")
    
    try:
        max_trial_counts = int(max_trial_counts)
        max_failed_trial_counts = int(max_failed_trial_counts)
        parallel_trial_counts = int(parallel_trial_counts)
    except ValueError:
        raise ValueError("All max_trial_counts, max_failed_trial_counts and needs to be an int!")

    experiment = V1beta1Experiment(
        api_version="kubeflow.org/v1beta1",
        kind="Experiment",
        metadata=metadata,
        spec=V1beta1ExperimentSpec(
            max_trial_count=max_trial_counts,
            parallel_trial_count=parallel_trial_counts,
            max_failed_trial_count=max_failed_trial_counts,
            algorithm=algorithm_spec,
            objective=objective_spec,
            parameters=parameters,
            trial_template=trial_template,
        )
    )

    client_namespace = input_params_metrics.metadata.get("client_namespace")
    if client_namespace is None:
        raise ValueError("Client namespace cannot be null!")

    client = KatibClient(namespace=client_namespace)
    client.create_experiment(experiment=experiment)
    client.wait_for_experiment_condition(name=experiment_name, namespace=experiment_namespace, timeout=3600)

    result = client.get_optimal_hyperparameters(name=experiment_name, namespace=experiment_namespace).to_dict()

    best_params_list = result["parameter_assignments"]

    for params in best_params_list:
        name = params["name"]
        value = params["value"]

        if name == "it":
            value = int(value)
            
        best_params_metrics.log_metric(metric=name, value=value)

@dsl.component(
    base_image='python:3.10-slim', 
    packages_to_install=['pandas', 'xgboost', 'scikit-learn', 'joblib']
)
def run_xgboost_train(
    best_params_metrics: Input[Metrics], 
    x_train: Input[Dataset], 
    x_test: Input[Dataset], 
    y_train: Input[Dataset], 
    y_test: Input[Dataset], 
    model: Output[Model], 
    file: Output[Artifact]
):
    import pandas as pd
    import xgboost as xgb
    from xgboost import XGBClassifier
    import joblib
    import json

    from sklearn.metrics import accuracy_score

    learning_rate = best_params_metrics.metadata.get("lr")
    n_estimators = best_params_metrics.metadata.get("ne")

    x_train_df = pd.read_csv(x_train.path)
    y_train_df = pd.read_csv(y_train.path)
    x_test_df = pd.read_csv(x_test.path)
    y_test_df = pd.read_csv(y_test.path)

    dtrain = xgb.DMatrix(x_train_df.values, label=y_train_df.values)
    dtest = xgb.DMatrix(x_test_df.values, label=y_test_df.values)
  
    xgb_model = XGBClassifier(
        n_estimators=n_estimators, 
        learning_rate= learning_rate
    )
    
    xgb_model.fit(x_train_df, y_train_df.values.ravel())
    
    preds = xgb_model.predict(dtest)

    predictions = [round(value) for value in preds]
    xgb_accuracy = accuracy_score(y_test_df.values, predictions)
    print('XGBoost Test accuracy:', xgb_accuracy)
    
    # Save the model
    joblib.dump(xgb_model, model.path)

     # Save the accuracy
    data = {}
    data['accuracy'] = xgb_accuracy
    data['model_path'] = model.path

    with open(file=file.path, mode='w', encoding='utf8') as file:
        json.dump(data, file, indent=4)

@dsl.component(
    base_image='python:3.10-slim', 
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)
def run_random_forest_train(
    best_params_metrics: Input[Metrics], 
    x_train: Input[Dataset], 
    x_test: Input[Dataset], 
    y_train: Input[Dataset], 
    y_test: Input[Dataset], 
    model: Output[Model], 
    file: Output[Artifact]
):
    import pandas as pd
    import joblib
    import json

    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier

    n_estimators = best_params_metrics.metadata.get("ne")

    x_train_df = pd.read_csv(x_train.path)
    y_train_df = pd.read_csv(y_train.path)
    x_test_df = pd.read_csv(x_test.path)
    y_test_df = pd.read_csv(y_test.path)

    rfc = RandomForestClassifier(n_estimators=n_estimators)
    rfc.fit(x_train_df.values, y_train_df.values.ravel())

    rfc.predict(x_test_df.values)
    rfc_accuracy = rfc.score(x_test_df.values, y_test_df.values)

    # Save the model
    joblib.dump(rfc, model.path)

    data = {}
    data['accuracy'] = rfc_accuracy
    data['model_path'] = model.path

    with open(file=file.path, mode='w', encoding='utf8') as file:
        json.dump(data, file, indent=4)

@dsl.component(
    base_image='python:3.10-slim', 
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)
def run_knn_train(
    best_params_metrics: Input[Metrics], 
    x_train: Input[Dataset], 
    x_test: Input[Dataset], 
    y_train: Input[Dataset], 
    y_test: Input[Dataset], 
    model: Output[Model], 
    file: Output[Artifact]
):
    import pandas as pd
    import joblib
    import json

    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    n_neighbors = best_params_metrics.metadata.get("nn")

    x_train_df = pd.read_csv(x_train.path)
    y_train_df = pd.read_csv(y_train.path)
    x_test_df = pd.read_csv(x_test.path)
    y_test_df = pd.read_csv(y_test.path)

    knn_model = KNeighborsClassifier(
        n_neighbors=n_neighbors
    )
    knn_model.fit(x_train_df.values, y_train_df.values.ravel())

    y_pred = knn_model.predict(x_test_df.values)
    accuracy = accuracy_score(y_test_df.values, y_pred)

    # Save the model
    joblib.dump(knn_model, model.path)

    data = {}
    data['accuracy'] = accuracy
    data['model_path'] = model.path

    with open(file=file.path, mode='w', encoding='utf8') as file:
        json.dump(data, file, indent=4)

@dsl.component(
    base_image='python:3.10-slim', 
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)
def run_lr_train(
    best_params_metrics: Input[Metrics], 
    x_train: Input[Dataset], 
    x_test: Input[Dataset], 
    y_train: Input[Dataset], 
    y_test: Input[Dataset], 
    model: Output[Model], 
    file: Output[Artifact]
):
    import pandas as pd
    import joblib
    import json

    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression

    iterators = best_params_metrics.metadata.get("it")

    x_train_df = pd.read_csv(x_train.path)
    y_train_df = pd.read_csv(y_train.path)
    x_test_df = pd.read_csv(x_test.path)
    y_test_df = pd.read_csv(y_test.path)

    lr_model = LogisticRegression(
        random_state=0, 
        max_iter=iterators
    )
    lr_model.fit(x_train_df.values, y_train_df.values.ravel())

    y_pred = lr_model.predict(x_test_df.values)
    accuracy = accuracy_score(y_test_df.values, y_pred)

    # Save the model
    joblib.dump(lr_model, model.path)

    data = {}
    data['accuracy'] = accuracy
    data['model_path'] = model.path

    with open(file=file.path, mode='w', encoding='utf8') as file:
        json.dump(data, file, indent=4)

@dsl.component(
    base_image='python:3.10-slim',
    packages_to_install=['joblib==1.4.2', 'scikit-learn==1.5.1', 'xgboost==2.0.3']# 
)
def choose_model(
    LogisticRegression_model: Input[Model],
    XGBoost_model: Input[Model],
    RandomForest_model: Input[Model],
    KNN_model: Input[Model],
    lr_file: Input[Artifact],
    xgb_file: Input[Artifact],
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
    data = {}
    data['accuracy'] = accuracy[best_model_name]
    data['model_path'] = final_model.path

    with open(file=result.path, mode='w', encoding='utf8') as file:
        json.dump(data, file, indent=4)

@dsl.component(
    base_image='python:3.10-slim',
    packages_to_install=['joblib==1.4.2', 'scikit-learn==1.5.1', 'xgboost==2.0.3']
)
def change_model(
    old_model_path: str, 
    old_model_file_path: str, 
    new_model: Input[Model],
    new_model_file: Input[Artifact], 
):
    import joblib
    import json
    import os

    with open(new_model_file.path, 'r') as f:
        data_new = json.load(f)
    new_model_accuracy = data_new['accuracy']

    new_model_model = joblib.load(new_model.path)

    if not (os.path.exists(old_model_path) and os.path.exists(old_model_file_path)):
        joblib.dump(new_model_model, old_model_path)
        with open(old_model_file_path, 'w', encoding='utf-8') as file:
            json.dump(data_new, file, indent=4)
        result_message = f"New model saved to NAS. Accuracy: {new_model_accuracy}"
        return None
    
    try:
        with open(old_model_file_path, 'r') as f:
            data_old = json.load(f)
        old_model_accuracy = data_old['accuracy']

        if new_model_accuracy > old_model_accuracy:
            joblib.dump(new_model_model, old_model_path)
            with open(old_model_file_path, 'w', encoding='utf-8') as file:
                json.dump(data_new, file, indent=4)
            result_message = f"Model updated. New accuracy: {new_model_accuracy}, Old accuracy: {old_model_accuracy}"
        else:
            result_message = f"Existing model retained. Existing accuracy: {old_model_accuracy}, New accuracy: {new_model_accuracy}"
    except (json.JSONDecodeError, UnicodeDecodeError):
        joblib.dump(new_model_model, old_model_path)
        with open(old_model_file_path, 'w', encoding='utf-8') as file:
            json.dump(data_new, file, indent=4)
        result_message = f"New model saved to NAS. Accuracy: {new_model_accuracy}"

    print(result_message)

@dsl.pipeline(
    name="compose", 
    description="Compose of kubeflow, katib and spark"
)
def compose_pipeline(
    params_pvc_name: str = "params-pvc", 
    params_json_file_path: str = "/mnt/params/params_heart_disease.json", 
    models_pvc_name: str = "models-pvc"
):
    sparkapplication_dict = get_spark_job_definition()

    k8s_apply_op = components.load_component_from_file("k8s-apply-component.yaml")
    apply_sparkapplication_task = k8s_apply_op(object=json.dumps(sparkapplication_dict))
    apply_sparkapplication_task.set_caching_options(enable_caching=False)

    check_sparkapplication_status_op = components.load_component_from_file("checkSparkapplication.yaml")
    check_sparkapplication_status_task = check_sparkapplication_status_op(
        name=sparkapplication_dict["metadata"]["name"],
        namespace=sparkapplication_dict["metadata"]["namespace"]
    ).after(apply_sparkapplication_task)
    check_sparkapplication_status_task.set_caching_options(enable_caching=False)

    load_datasets_task = load_file_from_nas_to_minio(
        x_train_input_path="/mnt/datasets/heart_disease/x_train.csv", 
        x_test_input_path="/mnt/datasets/heart_disease/x_test.csv", 
        y_train_input_path="/mnt/datasets/heart_disease/y_train.csv", 
        y_test_input_path="/mnt/datasets/heart_disease/y_test.csv", 
    ).after(check_sparkapplication_status_task)
    load_datasets_task.set_caching_options(enable_caching=False)

    kubernetes.mount_pvc(
        task=load_datasets_task, 
        pvc_name="datasets-pvc", 
        mount_path="/mnt/datasets"
    )

    parse_input_json_task = parse_input_json(
        json_file_path=params_json_file_path
    ).after(load_datasets_task)
    parse_input_json_task.set_caching_options(enable_caching=False)

    kubernetes.mount_pvc(
        task=parse_input_json_task, 
        pvc_name=params_pvc_name, 
        mount_path="/mnt/params"
    )

    xgboost_katib_experiment_task = run_xgboost_katib_experiment(
        input_params_metrics=parse_input_json_task.outputs["xgboost_input_metrics"]
    )

    random_forest_katib_experiment_task = run_random_forest_katib_experiment(
        input_params_metrics=parse_input_json_task.outputs["random_forest_input_metrics"]
    )

    knn_katib_experiment_task = run_knn_katib_experiment(
        input_params_metrics=parse_input_json_task.outputs["knn_input_metrics"]
    )

    lr_katib_experiment_task = run_lr_katib_experiment(
        input_params_metrics=parse_input_json_task.outputs["lr_input_metrics"]
    )

    xgboost_train_task = run_xgboost_train(
        best_params_metrics=xgboost_katib_experiment_task.outputs['best_params_metrics'], 
        x_train=load_datasets_task.outputs['x_train_output'], 
        x_test=load_datasets_task.outputs['x_test_output'], 
        y_train=load_datasets_task.outputs['y_train_output'], 
        y_test=load_datasets_task.outputs['y_test_output']
    )

    random_forest_train_task = run_random_forest_train(
        best_params_metrics=random_forest_katib_experiment_task.outputs['best_params_metrics'], 
        x_train=load_datasets_task.outputs['x_train_output'], 
        x_test=load_datasets_task.outputs['x_test_output'], 
        y_train=load_datasets_task.outputs['y_train_output'], 
        y_test=load_datasets_task.outputs['y_test_output']
    )

    knn_train_task = run_knn_train(
        best_params_metrics=knn_katib_experiment_task.outputs['best_params_metrics'], 
        x_train=load_datasets_task.outputs['x_train_output'], 
        x_test=load_datasets_task.outputs['x_test_output'], 
        y_train=load_datasets_task.outputs['y_train_output'], 
        y_test=load_datasets_task.outputs['y_test_output']
    )

    lr_train_task = run_lr_train(
        best_params_metrics=lr_katib_experiment_task.outputs['best_params_metrics'], 
        x_train=load_datasets_task.outputs['x_train_output'], 
        x_test=load_datasets_task.outputs['x_test_output'], 
        y_train=load_datasets_task.outputs['y_train_output'], 
        y_test=load_datasets_task.outputs['y_test_output']
    )

    choose_model_task = choose_model(
        LogisticRegression_model=lr_train_task.outputs['model'],
        XGBoost_model=xgboost_train_task.outputs['model'],
        RandomForest_model=random_forest_train_task.outputs['model'],
        KNN_model=knn_train_task.outputs['model'],
        lr_file=lr_train_task.outputs['file'],
        xgb_file=xgboost_train_task.outputs['file'],
        rf_file=random_forest_train_task.outputs['file'],
        knn_file=knn_train_task.outputs['file']
    )

    change_model_task = change_model(
        old_model_path="/mnt/models/heart_disease_model.pkl", 
        old_model_file_path="/mnt/models/heart_disease_model.json", 
        new_model=choose_model_task.outputs["final_model"], 
        new_model_file=choose_model_task.outputs["result"]
    )

    kubernetes.mount_pvc(
        task=change_model_task, 
        pvc_name=models_pvc_name, 
        mount_path="/mnt/models"
    )

if __name__ == "__main__":
    compiler.Compiler().compile(compose_pipeline, "../compose_pipeline.yaml")