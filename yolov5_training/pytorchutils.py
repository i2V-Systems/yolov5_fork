def benchmark_pytorch_model(model_path, input_size, device):
    import torch
    import numpy as np
    import time

    model = torch.load(model_path)
    model = model['model']
    model.to(device)
    model_dtype = 'half' if torch.float16 in [param.dtype for param in model.parameters()] else 'full'
    input_tensor = torch.randn(1, 3, input_size, input_size).to(device, dtype=torch.half if model_dtype=='half' else torch.float)

    warmup_runs = 10
    for i in range(warmup_runs):
        _ = model(input_tensor)

    runs = 50
    total_time = 0
    for i in range(runs):
        start_time = time.perf_counter()
        _ = model(input_tensor)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)

    total_time_ms = np.round((total_time / runs) * 1000, 2)
    
    if device == 'cpu':
        return total_time_ms, 'NA'
    
    
    # cpu
    model.to('cpu', dtype=torch.float)
    
    input_tensor = torch.randn(1, 3, input_size, input_size, dtype=torch.float)
    
    warmup_runs = 10
    for i in range(warmup_runs):
        _ = model(input_tensor)

    runs = 50
    total_time = 0
    for i in range(runs):
        start_time = time.perf_counter()
        _ = model(input_tensor)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)

    total_time_ms_cpu = np.round((total_time / runs) * 1000, 2)
    
    return total_time_ms_cpu, total_time_ms