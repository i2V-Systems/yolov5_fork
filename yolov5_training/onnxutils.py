def benchmark_onnx_model(onnx_model_path, image_size, device):
    import onnxruntime as ort
    import numpy as np
    import time
    
    session_cpu = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    session_gpu = ort.InferenceSession(onnx_model_path, providers=[{'provider': 'CUDAExecutionProvider', 'device_id': device}])

    input_tensor = np.random.rand(image_size, image_size, 3).astype(np.float32)
    input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, 0)
    
    model_inputs = session_cpu.get_inputs()
    model_output = session_cpu.get_outputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    output_names = [model_output[i].name for i in range(len(model_output))]

    model_inputs_ = session_gpu.get_inputs()
    model_output_ = session_gpu.get_outputs()
    input_names_ = [model_inputs_[i].name for i in range(len(model_inputs_))]
    output_names_ = [model_output_[i].name for i in range(len(model_output_))]

    # cpu
    warmup_runs = 10
    for i in range(warmup_runs):
        outputs = session_cpu.run(output_names, {input_names[0]: input_tensor})
    runs = 50
    total_time = 0
    for i in range(runs):
        start_time = time.perf_counter()
        outputs = session_cpu.run(output_names, {input_names[0]: input_tensor})
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        
    cpu_total_time = np.round(((total_time/runs)*1000), 2)

    if device == 'cpu':
        return cpu_total_time, 'NA'

    # gpu
    warmup_runs = 10
    for i in range(warmup_runs):
        outputs = session_gpu.run(output_names_, {input_names_[0]: input_tensor})
    runs = 50
    total_time = 0
    for i in range(runs):
        start_time = time.perf_counter()
        outputs = session_gpu.run(output_names_, {input_names_[0]: input_tensor})
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        
    gpu_total_time = np.round(((total_time/runs)*1000), 2)
    
    return cpu_total_time, gpu_total_time