import os
import sys
sys.path.append("../")
from utils.general import colorstr
from pytorchutils import benchmark_pytorch_model
from onnxutils import benchmark_onnx_model
from ovutils import benchmark_openvino_model, quantise_ov_model
import json

class MyObject:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

def get_arguments(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    custom_aug = data["model_parameters"]["custom_augmentation"]
    
    if custom_aug:
        return MyObject({**data['model_parameters'], **data['logging_parameters'], **data['augmentation_parameters']['custom_augmentation_parameters']})
    else:
        return MyObject({**data['model_parameters'], **data['logging_parameters'], **data['augmentation_parameters']['default_augmentation_parameters']})


def convert_model(model_path, train_opts):
    img_size = train_opts.image_size
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_dir = os.path.dirname(model_path)
    onnx_path = os.path.join(model_dir, (model_name + '.onnx'))
    onnx_dynamic_path = os.path.join(model_dir, (model_name + '_dynamic_bs' + '.onnx'))
    ov_path = os.path.join(model_dir, (model_name + '_openvino_model'))
    ov_int8_path = os.path.join(model_dir, (model_name + '_openvino_int8_model'))
    
    try:
        from ultralytics import YOLO
        print()
        prefix = colorstr('-->')
        print(f'{prefix} Starting model conversion for {model_path}')
        print()
        
        tmp_model = YOLO(model_path)
        tmp_model.export(format='onnx', imgsz=img_size, dynamic=True)
        os.rename(onnx_path, onnx_dynamic_path)
        tmp_model.export(format='onnx', imgsz=img_size)
        tmp_model.export(format='openvino', imgsz=img_size)
        prefix = colorstr('Benchmarking pytorch models ')
        print(prefix, model_path)
        pytorch_cpu_time, pytorch_gpu_time = benchmark_pytorch_model(model_path, img_size, f"cuda:{train_opts.device}")
        prefix = colorstr('Benchmarking onnx models ')
        print(prefix, onnx_path)
        onnx_cpu_time, onnx_gpu_time = benchmark_onnx_model(onnx_path, img_size, f"cuda:{train_opts.device}")
        prefix = colorstr('Benchmarking OpenVINO models ')
        print(prefix, ov_path)
        ov_time = benchmark_openvino_model(os.path.join(ov_path, (model_name+'.xml')), img_size)
        inference_times = {
            'Pytorch CPU Inference Time': f'{pytorch_cpu_time}',
            'Pytorch GPU Inference Time': f'{pytorch_gpu_time}',
            'Onnx CPU Inference Time': f'{onnx_cpu_time}',
            'Onnx GPU Inference Time': f'{onnx_gpu_time}',
            'OpenVINO Inference Time FP32': f'{ov_time}'
        }
    except Exception as E:
        print(f'Error in exporting=> {E}')
    
    try:
        prefix = colorstr('Attempting to quantise OpenVINO model to INT8')
        print(prefix, train_opts.data_path)
        quantise_ov_model(os.path.join(ov_path, (model_name+'.xml')), train_opts, model_path, f'{ov_int8_path}/{model_name}.xml')
        ov_int8_time = benchmark_openvino_model(f'{ov_int8_path}/{model_name}.xml', img_size)
        inference_times['OpenVINO Inference Time INT8'] = f'{ov_int8_time}'
        prefix = colorstr('Quantisation of OpenVINO model to INT8 successful')  
        print(prefix, ov_int8_path)
    except Exception as E:
        prefix = colorstr('Error in quantisation process: ')
        print(prefix)
        print(E)
        
    return inference_times