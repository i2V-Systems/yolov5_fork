def benchmark_openvino_model(xml_path, img_size):
    import numpy as np
    from openvino.runtime import Core
    import time
    
    input_tensor = np.random.rand(img_size, img_size, 3)
    input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, 0)

    core = Core()
    det_ov_model = core.read_model(xml_path)
    device = "CPU" 
    det_compiled_model = core.compile_model(det_ov_model, device)

    warmup_runs = 10
    for i in range(warmup_runs):
        result = det_compiled_model(input_tensor)

    runs = 50
    total_time = 0
    for i in range(runs):
        start_time = time.perf_counter()
        result = det_compiled_model(input_tensor)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    inference_time_ms = np.round(((total_time/runs) * 1000), 2)
    # print(f"Average inference time on OpenVINO: {inference_time_ms} ms")
    return inference_time_ms


def quantise_ov_model(xml_file_path, train_opts, model_path, save_path):
    from utils.general import colorstr
    print(colorstr("red", "Quantization of openVINO model is not implemented yet.."))