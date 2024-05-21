import os
import json
import boto3
import sagemaker
import sagemaker.session

from sagemaker import utils
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer

from sagemaker.model_metrics import MetricsSource, ModelMetrics, FileSource
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    TransformStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline_context import PipelineSession

# Importing new steps and helper functions

from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig,
)
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.clarify import BiasConfig, DataConfig, ModelConfig

#Create the SageMaker Session
region = sagemaker.Session().boto_region_name
sm_client = boto3.client("sagemaker")
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.session.Session(boto_session=boto_session, sagemaker_client=sm_client)
pipeline_session = PipelineSession()
prefix = "model-monitor-clarify-step-pipeline"

#Define variables and parameters
# role = sagemaker.get_execution_role()
# BUCKET_NAME = os.environ['BUCKET_NAME']
# PREFIX = os.environ['PREFIX']
# REGION = os.environ['AWS_DEFAULT_REGION']
# Replace with your IAM role arn that has enough access (e.g. SageMakerFullAccess)
role = os.environ['IAM_ROLE_NAME']
GITHUB_SHA = os.environ['GITHUB_SHA']
default_bucket = "sagemaker-pipeline-githubactions"
base_job_prefix = "train-monitor-clarify-pipeline"
model_package_group_name = "model-monitor-clarify-group"
pipeline_name = "model-monitor-clarify-pipeline-rad"
print(region)
print(role)
print(default_bucket)
print(base_job_prefix)


#Define pipeline parameters
processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
model_approval_status = ParameterString(
    name="ModelApprovalStatus", default_value="PendingManualApproval"
)
# The dataset used here is the open source Abalone dataset that can be found
# here - https://archive.ics.uci.edu/ml/datasets/abalone
input_data = ParameterString(
    name="InputDataUrl",
    default_value=f"s3://sagemaker-example-files-prod-{region}/datasets/tabular/uci_abalone/abalone.csv",
)

# for data quality check step
skip_check_data_quality = ParameterBoolean(name="SkipDataQualityCheck", default_value=False)
register_new_baseline_data_quality = ParameterBoolean(
    name="RegisterNewDataQualityBaseline", default_value=False
)
supplied_baseline_statistics_data_quality = ParameterString(
    name="DataQualitySuppliedStatistics", default_value=""
)
supplied_baseline_constraints_data_quality = ParameterString(
    name="DataQualitySuppliedConstraints", default_value=""
)

# for data bias check step
skip_check_data_bias = ParameterBoolean(name="SkipDataBiasCheck", default_value=False)
register_new_baseline_data_bias = ParameterBoolean(
    name="RegisterNewDataBiasBaseline", default_value=False
)
supplied_baseline_constraints_data_bias = ParameterString(
    name="DataBiasSuppliedBaselineConstraints", default_value=""
)

# for model quality check step
skip_check_model_quality = ParameterBoolean(name="SkipModelQualityCheck", default_value=False)
register_new_baseline_model_quality = ParameterBoolean(
    name="RegisterNewModelQualityBaseline", default_value=False
)
supplied_baseline_statistics_model_quality = ParameterString(
    name="ModelQualitySuppliedStatistics", default_value=""
)
supplied_baseline_constraints_model_quality = ParameterString(
    name="ModelQualitySuppliedConstraints", default_value=""
)

# for model bias check step
skip_check_model_bias = ParameterBoolean(name="SkipModelBiasCheck", default_value=False)
register_new_baseline_model_bias = ParameterBoolean(
    name="RegisterNewModelBiasBaseline", default_value=False
)
supplied_baseline_constraints_model_bias = ParameterString(
    name="ModelBiasSuppliedBaselineConstraints", default_value=""
)

# for model explainability check step
skip_check_model_explainability = ParameterBoolean(
    name="SkipModelExplainabilityCheck", default_value=False
)
register_new_baseline_model_explainability = ParameterBoolean(
    name="RegisterNewModelExplainabilityBaseline", default_value=False
)
supplied_baseline_constraints_model_explainability = ParameterString(
    name="ModelExplainabilitySuppliedBaselineConstraints", default_value=""
)


#Preprocess Step
sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    instance_type="ml.m5.xlarge",
    instance_count=processing_instance_count,
    base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
    sagemaker_session=pipeline_session,
    role=role,
)
processor_args = sklearn_processor.run(
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
    ],
    code="scripts/code/preprocess.py",
    arguments=["--input-data", input_data],
)
step_process = ProcessingStep(name="PreprocessAbaloneData", step_args=processor_args)

#====================================Data quality check=================
check_job_config = CheckJobConfig(
    role=role,
    instance_count=1,
    instance_type="ml.c5.xlarge",
    volume_size_in_gb=120,
    sagemaker_session=sagemaker_session,
)

data_quality_check_config = DataQualityCheckConfig(
    baseline_dataset=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
    dataset_format=DatasetFormat.csv(header=False, output_columns_position="START"),
    output_s3_uri=Join(
        on="/",
        values=[
            "s3:/",
            default_bucket,
            base_job_prefix,
            ExecutionVariables.PIPELINE_EXECUTION_ID,
            "dataqualitycheckstep",
        ],
    ),
)

data_quality_check_step = QualityCheckStep(
    name="DataQualityCheckStep",
    skip_check=skip_check_data_quality,
    register_new_baseline=register_new_baseline_data_quality,
    quality_check_config=data_quality_check_config,
    check_job_config=check_job_config,
    supplied_baseline_statistics=supplied_baseline_statistics_data_quality,
    supplied_baseline_constraints=supplied_baseline_constraints_data_quality,
    model_package_group_name=model_package_group_name,
)

#============data bais step===========================================
data_bias_analysis_cfg_output_path = (
    f"s3://{default_bucket}/{base_job_prefix}/databiascheckstep/analysis_cfg"
)

data_bias_data_config = DataConfig(
    s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs[
        "train"
    ].S3Output.S3Uri,
    s3_output_path=Join(
        on="/",
        values=[
            "s3:/",
            default_bucket,
            base_job_prefix,
            ExecutionVariables.PIPELINE_EXECUTION_ID,
            "databiascheckstep",
        ],
    ),
    label=0,
    dataset_type="text/csv",
    s3_analysis_config_output_path=data_bias_analysis_cfg_output_path,
)


data_bias_config = BiasConfig(
    label_values_or_threshold=[15.0], facet_name=[8], facet_values_or_threshold=[[0.5]]
)

data_bias_check_config = DataBiasCheckConfig(
    data_config=data_bias_data_config,
    data_bias_config=data_bias_config,
)

data_bias_check_step = ClarifyCheckStep(
    name="DataBiasCheckStep",
    clarify_check_config=data_bias_check_config,
    check_job_config=check_job_config,
    skip_check=skip_check_data_bias,
    register_new_baseline=register_new_baseline_data_bias,
    supplied_baseline_constraints=supplied_baseline_constraints_data_bias,
    model_package_group_name=model_package_group_name,
)

#=========================Train step======================
model_path = f"s3://{default_bucket}/{base_job_prefix}/AbaloneTrain"
image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.0-1",
    py_version="py3",
    instance_type="ml.m5.xlarge",
)

xgb_train = Estimator(
    image_uri=image_uri,
    instance_type=training_instance_type,
    instance_count=1,
    output_path=model_path,
    base_job_name=f"{base_job_prefix}/abalone-train",
    sagemaker_session=pipeline_session,
    role=role,
)

xgb_train.set_hyperparameters(
    objective="reg:linear",
    num_round=50,
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
    silent=0,
)

train_args = xgb_train.fit(
    inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "validation": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "validation"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
    },
)
step_train = TrainingStep(
    name="TrainAbaloneModel",
    step_args=train_args,
    depends_on=[data_bias_check_step.name, data_quality_check_step.name],
)

#==================Create the model====================
model = Model(
    image_uri=image_uri,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=pipeline_session,
    role=role,
)

step_create_model = ModelStep(
    name="AbaloneCreateModel",
    step_args=model.create(instance_type="ml.m5.large", accelerator_type="ml.eia1.medium"),
)


#==============================Transform Step==============================
transformer = Transformer(
    model_name=step_create_model.properties.ModelName,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    accept="text/csv",
    assemble_with="Line",
    output_path=f"s3://{default_bucket}/AbaloneTransform",
)

step_transform = TransformStep(
    name="AbaloneTransform",
    transformer=transformer,
    inputs=TransformInput(
        data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
        input_filter="$[1:]",
        join_source="Input",
        output_filter="$[0,-1]",
        content_type="text/csv",
        split_type="Line",
    ),
)


#===========================Check the Model Quality============================
model_quality_check_config = ModelQualityCheckConfig(
    baseline_dataset=step_transform.properties.TransformOutput.S3OutputPath,
    dataset_format=DatasetFormat.csv(header=False),
    output_s3_uri=Join(
        on="/",
        values=[
            "s3:/",
            default_bucket,
            base_job_prefix,
            ExecutionVariables.PIPELINE_EXECUTION_ID,
            "modelqualitycheckstep",
        ],
    ),
    problem_type="Regression",
    inference_attribute="_c0",  # use auto-populated headers since we don't have headers in the dataset
    ground_truth_attribute="_c1",  # use auto-populated headers since we don't have headers in the dataset
)

model_quality_check_step = QualityCheckStep(
    name="ModelQualityCheckStep",
    skip_check=skip_check_model_quality,
    register_new_baseline=register_new_baseline_model_quality,
    quality_check_config=model_quality_check_config,
    check_job_config=check_job_config,
    supplied_baseline_statistics=supplied_baseline_statistics_model_quality,
    supplied_baseline_constraints=supplied_baseline_constraints_model_quality,
    model_package_group_name=model_package_group_name,
)


#=======================Check for Model Bias==================================
model_bias_analysis_cfg_output_path = (
    f"s3://{default_bucket}/{base_job_prefix}/modelbiascheckstep/analysis_cfg"
)

model_bias_data_config = DataConfig(
    s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs[
        "train"
    ].S3Output.S3Uri,
    s3_output_path=Join(
        on="/",
        values=[
            "s3:/",
            default_bucket,
            base_job_prefix,
            ExecutionVariables.PIPELINE_EXECUTION_ID,
            "modelbiascheckstep",
        ],
    ),
    s3_analysis_config_output_path=model_bias_analysis_cfg_output_path,
    label=0,
    dataset_type="text/csv",
)

model_config = ModelConfig(
    model_name=step_create_model.properties.ModelName,
    instance_count=1,
    instance_type="ml.m5.xlarge",
)

# We are using this bias config to configure Clarify to detect bias based on the first feature in the featurized vector for Sex
model_bias_config = BiasConfig(
    label_values_or_threshold=[15.0], facet_name=[8], facet_values_or_threshold=[[0.5]]
)

model_bias_check_config = ModelBiasCheckConfig(
    data_config=model_bias_data_config,
    data_bias_config=model_bias_config,
    model_config=model_config,
    model_predicted_label_config=ModelPredictedLabelConfig(),
)

model_bias_check_step = ClarifyCheckStep(
    name="ModelBiasCheckStep",
    clarify_check_config=model_bias_check_config,
    check_job_config=check_job_config,
    skip_check=skip_check_model_bias,
    register_new_baseline=register_new_baseline_model_bias,
    supplied_baseline_constraints=supplied_baseline_constraints_model_bias,
    model_package_group_name=model_package_group_name,
)


#============================Check Model Explainability================
model_explainability_analysis_cfg_output_path = "s3://{}/{}/{}/{}".format(
    default_bucket, base_job_prefix, "modelexplainabilitycheckstep", "analysis_cfg"
)

model_explainability_data_config = DataConfig(
    s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs[
        "train"
    ].S3Output.S3Uri,
    s3_output_path=Join(
        on="/",
        values=[
            "s3:/",
            default_bucket,
            base_job_prefix,
            ExecutionVariables.PIPELINE_EXECUTION_ID,
            "modelexplainabilitycheckstep",
        ],
    ),
    s3_analysis_config_output_path=model_explainability_analysis_cfg_output_path,
    label=0,
    dataset_type="text/csv",
)
shap_config = SHAPConfig(seed=123, num_samples=10)
model_explainability_check_config = ModelExplainabilityCheckConfig(
    data_config=model_explainability_data_config,
    model_config=model_config,
    explainability_config=shap_config,
)
model_explainability_check_step = ClarifyCheckStep(
    name="ModelExplainabilityCheckStep",
    clarify_check_config=model_explainability_check_config,
    check_job_config=check_job_config,
    skip_check=skip_check_model_explainability,
    register_new_baseline=register_new_baseline_model_explainability,
    supplied_baseline_constraints=supplied_baseline_constraints_model_explainability,
    model_package_group_name=model_package_group_name,
)


#==================evaluate step==================================
script_eval = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name=f"{base_job_prefix}/script-abalone-eval",
    sagemaker_session=pipeline_session,
    role=role,
)
evaluation_report = PropertyFile(
    name="AbaloneEvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)

eval_args = script_eval.run(
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
    ],
    code="scripts/code/evaluate.py",
)
step_eval = ProcessingStep(
    name="EvaluateAbaloneModel",
    step_args=eval_args,
    property_files=[evaluation_report],
)


#====Define the metrics to be registered with the model in the Model Registry===

model_metrics = ModelMetrics(
    model_data_statistics=MetricsSource(
        s3_uri=data_quality_check_step.properties.CalculatedBaselineStatistics,
        content_type="application/json",
    ),
    model_data_constraints=MetricsSource(
        s3_uri=data_quality_check_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    ),
    bias_pre_training=MetricsSource(
        s3_uri=data_bias_check_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    ),
    model_statistics=MetricsSource(
        s3_uri=model_quality_check_step.properties.CalculatedBaselineStatistics,
        content_type="application/json",
    ),
    model_constraints=MetricsSource(
        s3_uri=model_quality_check_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    ),
    bias_post_training=MetricsSource(
        s3_uri=model_bias_check_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    ),
    explainability=MetricsSource(
        s3_uri=model_explainability_check_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    ),
)

drift_check_baselines = DriftCheckBaselines(
    model_data_statistics=MetricsSource(
        s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
        content_type="application/json",
    ),
    model_data_constraints=MetricsSource(
        s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
        content_type="application/json",
    ),
    bias_pre_training_constraints=MetricsSource(
        s3_uri=data_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
        content_type="application/json",
    ),
    bias_config_file=FileSource(
        s3_uri=model_bias_check_config.monitoring_analysis_config_uri,
        content_type="application/json",
    ),
    model_statistics=MetricsSource(
        s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
        content_type="application/json",
    ),
    model_constraints=MetricsSource(
        s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
        content_type="application/json",
    ),
    bias_post_training_constraints=MetricsSource(
        s3_uri=model_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
        content_type="application/json",
    ),
    explainability_constraints=MetricsSource(
        s3_uri=model_explainability_check_step.properties.BaselineUsedForDriftCheckConstraints,
        content_type="application/json",
    ),
    explainability_config_file=FileSource(
        s3_uri=model_explainability_check_config.monitoring_analysis_config_uri,
        content_type="application/json",
    ),
)

#=====================Register step==================
model_metrics = ModelMetrics(
    model_data_statistics=MetricsSource(
        s3_uri=data_quality_check_step.properties.CalculatedBaselineStatistics,
        content_type="application/json",
    ),
    model_data_constraints=MetricsSource(
        s3_uri=data_quality_check_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    ),
    bias_pre_training=MetricsSource(
        s3_uri=data_bias_check_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    ),
    model_statistics=MetricsSource(
        s3_uri=model_quality_check_step.properties.CalculatedBaselineStatistics,
        content_type="application/json",
    ),
    model_constraints=MetricsSource(
        s3_uri=model_quality_check_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    ),
    bias_post_training=MetricsSource(
        s3_uri=model_bias_check_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    ),
    explainability=MetricsSource(
        s3_uri=model_explainability_check_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    ),
)

drift_check_baselines = DriftCheckBaselines(
    model_data_statistics=MetricsSource(
        s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
        content_type="application/json",
    ),
    model_data_constraints=MetricsSource(
        s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
        content_type="application/json",
    ),
    bias_pre_training_constraints=MetricsSource(
        s3_uri=data_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
        content_type="application/json",
    ),
    bias_config_file=FileSource(
        s3_uri=model_bias_check_config.monitoring_analysis_config_uri,
        content_type="application/json",
    ),
    model_statistics=MetricsSource(
        s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
        content_type="application/json",
    ),
    model_constraints=MetricsSource(
        s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
        content_type="application/json",
    ),
    bias_post_training_constraints=MetricsSource(
        s3_uri=model_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
        content_type="application/json",
    ),
    explainability_constraints=MetricsSource(
        s3_uri=model_explainability_check_step.properties.BaselineUsedForDriftCheckConstraints,
        content_type="application/json",
    ),
    explainability_config_file=FileSource(
        s3_uri=model_explainability_check_config.monitoring_analysis_config_uri,
        content_type="application/json",
    ),
)


#====================Register the model============================
register_args = model.register(
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=model_package_group_name,
    approval_status=model_approval_status,
    model_metrics=model_metrics,
    drift_check_baselines=drift_check_baselines,
)

step_register = ModelStep(name="RegisterAbaloneModel", step_args=register_args)

#==================Condition step=============================
# condition step for evaluating model quality and branching execution
cond_lte = ConditionLessThanOrEqualTo(
    left=JsonGet(
        step_name=step_eval.name,
        property_file=evaluation_report,
        json_path="regression_metrics.mse.value",
    ),
    right=6.0,
)
step_cond = ConditionStep(
    name="CheckMSEAbaloneEvaluation",
    conditions=[cond_lte],
    if_steps=[step_register],
    else_steps=[],
)


#==================Pipeline============================
# pipeline instance
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        processing_instance_count,
        training_instance_type,
        model_approval_status,
        input_data,
        skip_check_data_quality,
        register_new_baseline_data_quality,
        supplied_baseline_statistics_data_quality,
        supplied_baseline_constraints_data_quality,
        skip_check_data_bias,
        register_new_baseline_data_bias,
        supplied_baseline_constraints_data_bias,
        skip_check_model_quality,
        register_new_baseline_model_quality,
        supplied_baseline_statistics_model_quality,
        supplied_baseline_constraints_model_quality,
        skip_check_model_bias,
        register_new_baseline_model_bias,
        supplied_baseline_constraints_model_bias,
        skip_check_model_explainability,
        register_new_baseline_model_explainability,
        supplied_baseline_constraints_model_explainability,
    ],
    steps=[
        step_process,
        data_quality_check_step,
        data_bias_check_step,
        step_train,
        step_create_model,
        step_transform,
        model_quality_check_step,
        model_bias_check_step,
        model_explainability_check_step,
        step_eval,
        step_cond,
    ],
    sagemaker_session=pipeline_session,
)


import json

definition = json.loads(pipeline.definition())

pipeline.upsert(role_arn=role)

execution = pipeline.start(
    parameters=dict(
        SkipDataQualityCheck=True,
        RegisterNewDataQualityBaseline=True,
        SkipDataBiasCheck=True,
        RegisterNewDataBiasBaseline=True,
        SkipModelQualityCheck=True,
        RegisterNewModelQualityBaseline=True,
        SkipModelBiasCheck=True,
        RegisterNewModelBiasBaseline=True,
        SkipModelExplainabilityCheck=True,
        RegisterNewModelExplainabilityBaseline=True,
    )
)


execution.wait()


















































































# import boto3
# import sagemaker
# from sagemaker import get_execution_role
# from sagemaker.sklearn.processing import SKLearnProcessor
# import json
# from sagemaker.s3 import S3Downloader
# from sagemaker.processing import ProcessingInput, ProcessingOutput
# from sagemaker.inputs import TrainingInput
# from sagemaker.sklearn.estimator import SKLearn
# from sagemaker.workflow.steps import ProcessingStep
# from sagemaker.workflow.steps import TrainingStep
# from sagemaker.workflow.pipeline import Pipeline
# from sagemaker.workflow.step_collections import RegisterModel
# import os 
# from sagemaker.workflow.pipeline_context import PipelineSession
# from sagemaker.workflow.check_job_config import CheckJobConfig
# from sagemaker.workflow.quality_check_step import (
#     DataQualityCheckConfig,
#     ModelQualityCheckConfig,
#     QualityCheckStep,
# )
# from sagemaker.workflow.clarify_check_step import (
#     DataBiasCheckConfig,
#     ClarifyCheckStep,
#     ClarifyCheckConfig,
#     ModelBiasCheckConfig,
#     ModelPredictedLabelConfig,
#     ModelExplainabilityCheckConfig,
#     SHAPConfig
# )
# from sagemaker.clarify import (
#     BiasConfig,
#     DataConfig,
#     ModelConfig
# )
# from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
# from sagemaker.workflow.condition_step import (
#     ConditionStep
# )
# from sagemaker.workflow.functions import (
#     JsonGet
# )
# from sagemaker.workflow.properties import PropertyFile
# from sagemaker.model_metrics import MetricsSource, ModelMetrics 
# from sagemaker.workflow.functions import Join
# from sagemaker.workflow.execution_variables import ExecutionVariables
# from sagemaker.model_monitor.dataset_format import DatasetFormat
# from sagemaker.drift_check_baselines import DriftCheckBaselines
# from time import gmtime, strftime
# from sagemaker.lambda_helper import Lambda
# from sagemaker.workflow.lambda_step import (
#     LambdaStep,
#     LambdaOutput,
#     LambdaOutputTypeEnum,
# )
# from sagemaker.model import Model
# from sagemaker.workflow.model_step import ModelStep
# from sagemaker.workflow.steps import CreateModelStep
# import io
# import os
# import pandas as pd
# import sys
# import time
# from time import gmtime, strftime, sleep
# from sagemaker.feature_store.feature_group import FeatureGroup
# from sagemaker.feature_store.inputs import FeatureParameter

# session = sagemaker.session.Session()
# region = os.environ['AWS_DEFAULT_REGION']
# role = os.environ['IAM_ROLE_NAME']
# # bucket = os.environ['BUCKET_NAME']
# # prefix = os.environ['PREFIX']
# bucket="sagemaker-pipeline-githubactions"
# prefix="pipeline-featurestore-final"
# model_package_group_name = "github-Churn-xgboost-model-grp-1"  # Model name in model registry

# pipeline_name = "ChurnPipeline"
# print(region)
# print(role)
# print(bucket)
# print(prefix)

# current_working_directory = os.getcwd()
# print(current_working_directory)

# churn_feature_group_name="2churndata-feature-group-07-08-22-37"
# sagemaker_session = sagemaker.Session()
# churn_feature_group = FeatureGroup(
#     name=churn_feature_group_name, sagemaker_session=sagemaker_session
# )

# feature_query = churn_feature_group.athena_query()
# churn_table_name=feature_query.table_name
# print(churn_table_name)
# query_string=f"SELECT * FROM \"sagemaker_featurestore\".\"{churn_table_name}\""
# # print(len(query_string))
# query_result_folder="query_results1"+ strftime("%d-%H-%M-%S", gmtime())
# feature_query.run(query_string=query_string, output_location='s3://'+bucket+'/'+query_result_folder+'/')
# feature_query.wait()
# dataset = feature_query.as_dataframe()
# dataset = dataset.drop(["api_invocation_time", "eventtime", "write_time", "is_deleted"], axis=1)
# print(dataset.shape)



# #save the dataset to data folder
# dataset[:400:].to_csv("scripts/data/large/churn-dataset.csv",index=False)
# dataset[401::].to_csv("scripts/data/small/churn-dataset.csv",index=False)
# # dataset[401::].to_csv("scripts/data/test.csv",index=False)
# # Upload the csv files to S3
# large_input_data_uri = session.upload_data(path="scripts/data/large/churn-dataset.csv",bucket=bucket, key_prefix=prefix + "/data/large")
# small_input_data_uri = session.upload_data(path="scripts/data/small/churn-dataset.csv",bucket=bucket, key_prefix=prefix + "/data/small")
# test_data_uri = session.upload_data(path="scripts/data/test.csv",bucket=bucket, key_prefix=prefix + "/data/test")

# print("Large data set uploaded to ", large_input_data_uri)
# print("Small data set uploaded to ", small_input_data_uri)
# print("Test data set uploaded to ", test_data_uri)

# from sagemaker.workflow.parameters import (
#     ParameterInteger,
#     ParameterString,
# )

# # How many instances to use when processing
# processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)

# # What instance type to use for processing
# processing_instance_type = ParameterString(
#     name="ProcessingInstanceType", default_value="ml.m5.large"
# )

# # What instance type to use for training
# training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")

# # Where the input data is stored
# input_data = ParameterString(
#     name="InputData",
#     default_value=large_input_data_uri,
# )

# test_data = ParameterString(
#     name="testData",
#     default_value=test_data_uri,
# )

# # What is the default status of the model when registering with model registry.
# model_approval_status = ParameterString(
#     name="ModelApprovalStatus", default_value="PendingManualApproval"
# )



# from sagemaker.sklearn.processing import SKLearnProcessor
# from sagemaker.processing import ProcessingInput, ProcessingOutput
# from sagemaker.workflow.steps import ProcessingStep
# from sagemaker.workflow.functions import Join
# from sagemaker.workflow.execution_variables import ExecutionVariables

# # Create SKlearn processor object,
# # The object contains information about what instance type to use, the IAM role to use etc.
# # A managed processor comes with a preconfigured container, so only specifying version is required.
# sklearn_processor = SKLearnProcessor(
#     framework_version="0.23-1",
#     role=role,
#     instance_type=processing_instance_type,
#     instance_count=processing_instance_count,
#     base_job_name="churn-processing-job",
# )

# # Use the sklearn_processor in a Sagemaker pipelines ProcessingStep
# step_preprocess_data = ProcessingStep(
#     name="Preprocess-Churn-Data",
#     processor=sklearn_processor,
#     inputs=[
#         ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
#     ],
#     outputs=[
#         ProcessingOutput(
#             output_name="train",
#             source="/opt/ml/processing/train",
#             destination=Join(
#                 on="/",
#                 values=[
#                     "s3://{}".format(bucket),
#                     prefix,
#                     ExecutionVariables.PIPELINE_EXECUTION_ID,
#                     "train",
#                 ],
#             ),
#         ),
#         ProcessingOutput(
#             output_name="validation",
#             source="/opt/ml/processing/validation",
#             destination=Join(
#                 on="/",
#                 values=[
#                     "s3://{}".format(bucket),
#                     prefix,
#                     ExecutionVariables.PIPELINE_EXECUTION_ID,
#                     "validation",
#                 ],
#             ),
#         ),
#         ProcessingOutput(
#             output_name="test",
#             source="/opt/ml/processing/test",
#             destination=Join(
#                 on="/",
#                 values=[
#                     "s3://{}".format(bucket),
#                     prefix,
#                     ExecutionVariables.PIPELINE_EXECUTION_ID,
#                     "test",
#                 ],
#             ),
#         ),
#     ],
#     code="scripts/preprocess.py",
# )


# #================================================train==================================

# from sagemaker.inputs import TrainingInput
# from sagemaker.workflow.steps import TrainingStep
# from sagemaker.estimator import Estimator
# model_path=f"s3://{bucket}/{prefix}/churnmodel"
# # Fetch container to use for training
# image_uri = sagemaker.image_uris.retrieve(
#     framework="xgboost",
#     region=region,
#     version="1.2-2",
#     py_version="py3",
#     instance_type="ml.m5.xlarge",
# )

# # Create XGBoost estimator object
# # The object contains information about what container to use, what instance type etc.
# xgb_estimator = Estimator(
#     image_uri=image_uri,
#     instance_type=training_instance_type,
#     instance_count=1,
#     role=role,
#     disable_profiler=True,
#     output_path=model_path

# )

# xgb_estimator.set_hyperparameters(
#     max_depth=5,
#     eta=0.2,
#     gamma=4,
#     min_child_weight=6,
#     subsample=0.8,
#     objective="binary:logistic",
#     num_round=25,
# )

# # Use the xgb_estimator in a Sagemaker pipelines ProcessingStep.
# # NOTE how the input to the training job directly references the output of the previous step.
# step_train_model = TrainingStep(
#     name="Train-Churn-Model",
#     estimator=xgb_estimator,
#     inputs={
#         "train": TrainingInput(
#             s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
#                 "train"
#             ].S3Output.S3Uri,
#             content_type="text/csv",
#         ),
#         "validation": TrainingInput(
#             s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
#                 "validation"
#             ].S3Output.S3Uri,
#             content_type="text/csv",
#         ),
#     },
# )

# #===================================================evaluate=====================================

# from sagemaker.processing import ScriptProcessor
# from sagemaker.workflow.properties import PropertyFile

# # Create ScriptProcessor object.
# # The object contains information about what container to use, what instance type etc.
# evaluate_model_processor = ScriptProcessor(
#     image_uri=image_uri,
#     command=["python3"],
#     instance_type=processing_instance_type,
#     instance_count=processing_instance_count,
#     base_job_name="script-churn-eval",
#     role=role,
# )

# # Create a PropertyFile
# # A PropertyFile is used to be able to reference outputs from a processing step, for instance to use in a condition step.
# # For more information, visit https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
# evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
# print(f"bucket uri before evaluate step: s3://{bucket}/{prefix}")
# # Use the evaluate_model_processor in a Sagemaker pipelines ProcessingStep.
# step_evaluate_model = ProcessingStep(
#     name="Evaluate-Churn-Model",
#     processor=evaluate_model_processor,
#     inputs=[
#         ProcessingInput(
#             source=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
#             destination="/opt/ml/processing/model",
#         ),
#         ProcessingInput(
#             source=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
#                 "test"
#             ].S3Output.S3Uri,
#               # Use pre-created test data instead of output from processing step
#             destination="/opt/ml/processing/test",
#         ),
#     ],
#     outputs=[
#         ProcessingOutput(
#             output_name="evaluation",
#             source="/opt/ml/processing/evaluation",
#             destination=Join(
#                 on="/",
#                 values=[
#                     "s3://{}".format(bucket),
#                     "modeloutput",
#                     ExecutionVariables.PIPELINE_EXECUTION_ID,
#                     "evaluation-report",
#                 ],
#             ),
#         ),
#     ],
#     code="scripts/evaluate.py",
#     property_files=[evaluation_report],
# )


# #==================================== Register model==================================
# from sagemaker.model_metrics import MetricsSource, ModelMetrics
# from sagemaker.workflow.step_collections import RegisterModel

# # Create ModelMetrics object using the evaluation report from the evaluation step
# # A ModelMetrics object contains metrics captured from a model.
# model_metrics = ModelMetrics(
#     model_statistics=MetricsSource(
#         s3_uri=Join(
#             on="/",
#             values=[
#                 step_evaluate_model.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
#                     "S3Uri"
#                 ],
#                 "evaluation.json",
#             ],
#         ),
#         content_type="application/json",
#     )
# )

# # Crete a RegisterModel step, which registers the model with Sagemaker Model Registry.
# step_register_model = RegisterModel(
#     name="Register-Churn-Model",
#     estimator=xgb_estimator,
#     model_data=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
#     content_types=["text/csv"],
#     response_types=["text/csv"],
#     inference_instances=["ml.t2.medium", "ml.m5.xlarge", "ml.m5.large"],
#     transform_instances=["ml.m5.xlarge"],
#     model_package_group_name=model_package_group_name,
#     approval_status=model_approval_status,
#     model_metrics=model_metrics,
# )

# #================================Condition Step====================================
# from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
# from sagemaker.workflow.condition_step import ConditionStep
# from sagemaker.workflow.functions import JsonGet

# # Create accuracy condition to ensure the model meets performance requirements.
# # Models with a test accuracy lower than the condition will not be registered with the model registry.
# cond_gte = ConditionGreaterThanOrEqualTo(
#     left=JsonGet(
#         step_name=step_evaluate_model.name,
#         property_file=evaluation_report,
#         json_path="binary_classification_metrics.accuracy.value",
#     ),
#     right=0.7,
# )

# # Create a Sagemaker Pipelines ConditionStep, using the condition above.
# # Enter the steps to perform if the condition returns True / False.
# step_cond = ConditionStep(
#     name="Accuracy-Condition",
#     conditions=[cond_gte],
#     if_steps=[step_register_model],
#     else_steps=[],
# )
# print("============================condition tested=====================================")
# print("Condition step")
# print(step_cond)
# #==================================pipeline==========================================
# from sagemaker.workflow.pipeline import Pipeline

# # Create a Sagemaker Pipeline.
# # Each parameter for the pipeline must be set as a parameter explicitly when the pipeline is created.
# # Also pass in each of the steps created above.
# # Note that the order of execution is determined from each step's dependencies on other steps,
# # not on the order they are passed in below.
# pipeline = Pipeline(
#     name=pipeline_name,
#     parameters=[
#         processing_instance_type,
#         processing_instance_count,
#         training_instance_type,
#         model_approval_status,
#         input_data,
#     ],
#     steps=[step_preprocess_data, step_train_model, step_evaluate_model, step_cond],
# )

# print("============================pipeline triggered=====================================")
# # # Submit pipline
# # pipeline.upsert(role_arn=role)

# # # Execute pipeline using the default parameters.
# # execution = pipeline.start()

# # execution.wait()

# # # List the execution steps to check out the status and artifacts:
# # execution.list_steps()
# print("============================pipeline execution completed=====================================")

























