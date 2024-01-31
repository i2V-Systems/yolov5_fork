import os
import mlflow

import sys
sys.path.append("../")
from utils.general import colorstr
from utilsv5 import convert_model

def set_logging_url(url):
    global logging_url
    logging_url = url

def on_pretrain_routine_start():
    prefix = colorstr('MLFlow: ')
    print(f"{prefix}MLFlow logging enabled, view at {logging_url}")
    
def on_train_start(train_opts):
    experiment_name = train_opts.project_name
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=train_opts.exp_name)
    mlflow.log_param('Epochs', train_opts.epochs)
    mlflow.log_param('Early stopping patience', train_opts.patience)
    mlflow.log_param('Batch size', train_opts.batch_size)
    mlflow.log_param('Image size', train_opts.image_size)
    mlflow.log_artifact(os.path.join("./",experiment_name,train_opts.exp_name, 'opt.yaml'))

def on_fit_epoch_end(metrics, epoch):
    fixed_metrics = {}
    for metric, val in metrics.items():
        metric_name = metric.replace('/', '_')
        metric_name = metric.replace(':', '_')
        fixed_metrics[metric_name] = val
    mlflow.log_metrics(fixed_metrics, step=epoch+1)

def on_batch_end():
    pass

def on_train_end(save_dir, last, best, epoch, final_results, train_opts):
    
    import cpuinfo
    import torch
    import numpy as np

    # logging plots
    images = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.png')]
    for image in images:
        mlflow.log_artifact(image)

    # logging pt models
    mlflow.log_artifact(str(best))
    mlflow.log_artifact(str(last))    
    
    # logging hardware details
    cpu_name = cpuinfo.get_cpu_info()['brand_raw']
    num_cores = cpuinfo.get_cpu_info()["count"]

    gpu_name = torch.cuda.get_device_name(torch.device(int(train_opts.device))) if torch.cuda.is_available() else 'No GPU Card Available'
    gpu_memory = np.round(torch.cuda.get_device_properties(0).total_memory / 1024** 3, 3) if torch.cuda.is_available() else 'NA'

    cpu_info = f"{cpu_name} ({num_cores} cores)"
    gpu_info = f"{gpu_name} ({gpu_memory} GB)"
    
    mlflow.log_param('CPU', cpu_info)
    mlflow.log_param('GPU', gpu_info)
    
    # logging inference times
    inf_times = convert_model(str(best), train_opts)
    
    for key, value in inf_times.items():
        print(f"{key}: {value} ms")
    
    for param_name, param_value in inf_times.items():
        mlflow.log_param(param_name, f"{param_value} ms")
        
    # logging onnx and openvino models(fp32, int8)
    model_name = os.path.splitext(os.path.basename(str(best)))[0]
    model_dir = os.path.dirname(str(best))
    onnx_path = os.path.join(model_dir, (model_name + '.onnx'))
    ov_path = os.path.join(model_dir, (model_name + '_openvino_model'))    
    ov_int8_path = os.path.join(model_dir, (model_name + '_openvino_int8_model'))
    mlflow.log_artifact(onnx_path)
    mlflow.log_artifacts(ov_path, 'OpenVINO Model (FP32)')
    mlflow.log_artifacts(ov_int8_path, 'OpenVINO Model (INT8)')
    
    mlflow.end_run()
    prefix = colorstr('MLFlow: ')
    print(f'{prefix}Run completed, view at {logging_url}')