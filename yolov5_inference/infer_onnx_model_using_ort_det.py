import torch
import onnxruntime

import os
import time
import cv2
import numpy as np

from tqdm import tqdm

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


""" 
# Export yolo to onnx..
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt") 
    model.export(format="onnx", imgsz=[640,640], opset=12)
"""

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def preprocess(image, input_shape, dynamic_input_shape=False):
    im = letterbox(image, input_shape, stride=32, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    input_image = im / 255.0

    model_input_shape = input_image.shape[1:]
    
    # B*C*H*W
    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

    if dynamic_input_shape:
        return input_tensor, model_input_shape

    return input_tensor

def get_detections(image, ort_session, input_names, input_shape, output_names, dynamic_input_shape=False):
    image_height, image_width = image.shape[:2]
    
    if dynamic_input_shape:
        input_tensor, model_input_shape = preprocess(image, input_shape, dynamic_input_shape=dynamic_input_shape)
        input_height, input_width = model_input_shape[:]
    else:
        input_tensor = preprocess(image, input_shape, dynamic_input_shape=dynamic_input_shape)
        input_height, input_width = input_shape[:2]

    outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]

    predictions = np.squeeze(outputs)

    conf_thresold = 0.5
    # Filter out object confidence scores below threshold
    scores = predictions[:, 4]
    predictions = predictions[scores > conf_thresold, :]
    scores = scores[scores > conf_thresold]

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 5:], axis=1)

    # Get bounding boxes for each object
    boxes = predictions[:, :4]

    #rescale box
    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_width, image_height, image_width, image_height])
    boxes = boxes.astype(np.int32)
    return boxes, scores, class_ids

def get_roi(image):
    image_height, image_width = image.shape[:2]
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI", image_width, image_height)
    
    # Display the first frame
    cv2.imshow("Select ROI", image)
    roi_points = []

    # Callback function for mouse events
    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_points

        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
            if len(roi_points) > 1:
                cv2.line(image, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
            cv2.imshow("Select ROI", image)

    # Set mouse callback for the window
    cv2.setMouseCallback("Select ROI", mouse_callback)

    # Wait for the user to finish selecting the ROI
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 27:  # Enter key or Esc key to finish selecting ROI
            break

    # Close the window
    cv2.destroyAllWindows()

    # Convert the list of points to a NumPy array
    roi_points = np.array(roi_points, dtype=np.int32)

    return roi_points

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def xyxy2xywh(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] + (x[..., 2] - x[..., 0]) / 2
    y[..., 1] = x[..., 1] + (x[..., 3] - x[..., 1]) / 2
    y[..., 2] = (x[..., 2] - x[..., 0])
    y[..., 3] = (x[..., 3] - x[..., 1])
    return y

if __name__ == "__main__":
    
    INPUT_MODE = 'image' # {'image', 'video', 'live'}
    SAVE_ANNOTATIONS = False
    
    if SAVE_ANNOTATIONS:
        CONSIDER_ROI = False # make it true, if you want to save annotations which are within ROI only
        save_predictions_score = False
    
    if INPUT_MODE == 'video':
        num_frames_to_skip = 4
        save_output_frames = True
        save_output_video = False
    
    if INPUT_MODE == 'live':
        camera_ip = 0
        num_frames_to_skip = 0
        save_output_frames = True
        save_output_video = False
        view_output_frames = True
        
    dynamic_input_model = True
    
    model_path = '/home/i2v-admin/Documents/Raushan/ultralytics/yolov5/runs/train/yolov5n-exp-2-rotation90-416/weights/face-detection-yolov5n-exp-2-rotation90-416-best_opset12-dynamic.onnx'
    input_dir = '/media/i2v-admin/2tb_drive/Raushan/Face_Detection/frvt/tmp' #'/media/i2v-admin/2tb_drive/Raushan/Face_Detection/frvt/common/images/face'
    output_dir = '/media/i2v-admin/2tb_drive/Raushan/Face_Detection/frvt_outputs'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    opt_session = onnxruntime.SessionOptions()
    opt_session.enable_mem_pattern = False
    opt_session.enable_cpu_mem_arena = False
    opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    ExecutionProviders_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    ort_session = onnxruntime.InferenceSession(model_path, providers=ExecutionProviders_list)

    model_inputs = ort_session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]

    model_output = ort_session.get_outputs()
    output_names = [model_output[i].name for i in range(len(model_output))]
    
    input_shape = (416, 416) # (H, W)
    
    CLASSES = ['face']
    
    
    if INPUT_MODE == 'image':
        total_frames = sum([len(files) for root, dirs, files in os.walk(input_dir)])
        # pbar = tqdm(total=total_frames, unit="frame")
    
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                
                image = cv2.imread(os.path.join(root, file))  
            
                boxes, scores, class_ids = get_detections(image, ort_session, input_names, input_shape, output_names, dynamic_input_shape=dynamic_input_model)
                
                indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.3) # (bboxes, scores, score_threshold, nms_threshold)
                # indices = cv2.dnn.NMSBoxesBatched(boxes, scores, class_ids, 0.5, 0.3)

                image_draw = image.copy()
                count = 1
                object_count_per_img = 0
                for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
                    bbox = bbox.round().astype(np.int32).tolist()
                    cls_id = int(label)
                    cls = CLASSES[cls_id]
                    color = (0,255,0)
                    cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                    cv2.putText(image_draw,
                                f'{cls}:{int(score*100)}', (bbox[0], bbox[1]+30 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.60, [0, 0, 255],
                                thickness=2)
                    
                    object_count_per_img += 1
                    
                    # if score < 0.5:
                    #     print(int(score*100), file)
                
                if object_count_per_img == 0:
                    print(f"{file} having 0 objects")
                
                out_dir = os.path.join(output_dir, INPUT_MODE)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                
                cv2.imwrite(os.path.join(out_dir, file), image_draw)
                
                # pbar.update(1)
    
    if INPUT_MODE == 'video':
        output_vid_ext = '.mp4'
        
        total_videos = len([file for root, dirs, files in os.walk(input_dir) for file in files if file.endswith(('.avi', '.ts', '.mp4'))])
        
        for root, dirs, files in os.walk(input_dir):
            vid_idx = 0
            for file in files:
                vid_name = ".".join(file.split(".")[0:-1])
                if save_output_video:
                    output_vid_name = vid_name + output_vid_ext
                    
                    
                if file.endswith(('.avi', '.ts', '.mp4')):
                    vid_idx += 1
                    
                    print(f"========== Processing video {vid_idx}/{total_videos} ==================")
                    print(f"input_vid_path : {os.path.join(root, file)}") 
                    
                    video_capture = cv2.VideoCapture(os.path.join(root, file))
                    
                    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    pbar = tqdm(total=total_frames)
                    
                    # Get video properties (frame width, frame height, and frames per second)
                    frame_width = 1920  # int(video_capture.get(3))
                    frame_height = 1080 # int(video_capture.get(4))
                    fps = int(video_capture.get(5))
                    
                    # Define the codec and create VideoWriter object
                    if save_output_video:
                        out_vid_path = os.path.join(output_dir, INPUT_MODE, 'videos')
                        if not os.path.exists(out_vid_path):
                            os.makedirs(out_vid_path)
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' codec on Windows. For macOS or Linux, try 'MJPG'.
                        out = cv2.VideoWriter(os.path.join(out_vid_path, output_vid_name), fourcc, fps, (frame_width, frame_height))
                        
                    if save_output_frames:
                        out_dir = os.path.join(output_dir, INPUT_MODE, 'frames', str(vid_idx))
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                            
                    if SAVE_ANNOTATIONS:
                        save_annotations_path = os.path.join(output_dir, INPUT_MODE, 'annotations', 'labels', str(vid_idx))
                        save_imgs_path = os.path.join(output_dir, INPUT_MODE, 'annotations', 'images', str(vid_idx))
                        if not os.path.exists(save_annotations_path):
                            os.makedirs(save_annotations_path)
                            
                        if not os.path.exists(save_imgs_path):
                            os.makedirs(save_imgs_path)
                    
                    
                    frame_counter = 0
                    
                    ret = True
                    while ret:
                        ret, frame_raw = video_capture.read()
                        frame_counter += 1
                        
                        # take roi..
                        if CONSIDER_ROI:
                            if vid_idx == 1 and frame_counter == 1:
                                # roi_points = get_roi(frame_raw)
                                roi_points = np.array([[   4, 1076],[   1,   91],[ 163,   74],[ 179,   42],[ 421,   44],[1816,  853],[1845,  652],[ 983,  308],[ 786,  115],[1910,  217],[1915, 1075],[   4, 1077]])
                        
                        if frame_counter % (num_frames_to_skip + 1) == 0 and not (frame_raw is None):
                            if save_output_frames:
                                output_frame = os.path.join(out_dir, "{}_{}_{}.jpg".format(vid_name, vid_idx, frame_counter))
                                
                            if SAVE_ANNOTATIONS:
                                image_name = "{}_{}_{}.jpg".format(vid_name, vid_idx, frame_counter)
                                cv2.imwrite(os.path.join(save_imgs_path, image_name), frame_raw)
                                output_txt = os.path.join(save_annotations_path, image_name.split('.jpg')[0] + '.txt')
                                f = open(output_txt, 'w')
                            
                            image = frame_raw.copy() 
                        
                            boxes, scores, class_ids = get_detections(image, ort_session, input_names, input_shape, output_names, dynamic_input_shape=dynamic_input_model)
                            
                            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.3) # (bboxes, scores, score_threshold, nms_threshold)
                            # indices = cv2.dnn.NMSBoxesBatched(boxes, scores, class_ids, 0.5, 0.3)

                            image_draw = frame_raw.copy()
                            count = 1
                            for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
                                bbox = bbox.round().astype(np.int32).tolist()
                                cls_id = int(label)
                                cls = CLASSES[cls_id]
                                color = (0,255,0)
                                cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                                cv2.putText(image_draw,
                                            f'{cls}:{int(score*100)}', (bbox[0], bbox[1] - 2),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.60, [0, 0, 255],
                                            thickness=2)
                                
                                ##save annotations here..
                                if SAVE_ANNOTATIONS:
                                    
                                    xc = bbox[0] + (bbox[2] - bbox[0]) / 2
                                    yc = bbox[1] + (bbox[3] - bbox[1]) / 2
                                    
                                    if CONSIDER_ROI:
                                        point = Point(xc, yc)
                                        polygon = Polygon(roi_points)
                                        object_center_within_roi = polygon.contains(point)
                                    
                                    if CONSIDER_ROI and (not object_center_within_roi): #and (cls == 'axle'):
                                        pass
                                    else:
                                        image_height, image_width = image.shape[:2]
                                        input_height, input_width = input_shape[:2]
                                        image_shape = np.array([image_width, image_height, image_width, image_height])
                                        bbox = np.divide(bbox, image_shape, dtype=np.float32)
                                        bbox = xyxy2xywh(np.array(bbox))
                                        if save_predictions_score:
                                            f.write(f"{cls_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {score}\n")
                                        else:
                                            f.write(f"{cls_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                                    
                            if SAVE_ANNOTATIONS:
                                f.close()      

                            if save_output_frames:
                                cv2.imwrite(output_frame, image_draw)
                
                            image_draw = cv2.resize(image_draw, (frame_width, frame_height))
                            
                            if save_output_video:
                                out.write(image_draw)
                                
                            pbar.update(num_frames_to_skip + 1)
    
                    pbar.close()
    
    if INPUT_MODE == 'live':
        output_vid_ext = '.mp4'
        
        vid_name = "live"
        if save_output_video:
            output_vid_name = vid_name + output_vid_ext
        
        stream = cv2.VideoCapture(camera_ip)
        
        # Get video properties (frame width, frame height, and frames per second)
        frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)) # 1920
        frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 1080
        fps = int(stream.get(cv2.CAP_PROP_FPS))
        
        # Define the codec and create VideoWriter object
        if save_output_video:
            out_vid_path = os.path.join(output_dir, INPUT_MODE, 'videos')
            if not os.path.exists(out_vid_path):
                os.makedirs(out_vid_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' codec on Windows. For macOS or Linux, try 'MJPG'.
            out = cv2.VideoWriter(os.path.join(out_vid_path, output_vid_name), fourcc, fps, (frame_width, frame_height))
            
        if save_output_frames:
            out_dir = os.path.join(output_dir, INPUT_MODE, 'frames')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
        if SAVE_ANNOTATIONS:
            save_annotations_path = os.path.join(output_dir, INPUT_MODE, 'annotations', 'labels')
            save_imgs_path = os.path.join(output_dir, INPUT_MODE, 'annotations', 'images')
            if not os.path.exists(save_annotations_path):
                os.makedirs(save_annotations_path)
                
            if not os.path.exists(save_imgs_path):
                os.makedirs(save_imgs_path)
        
        frame_counter = 0

        print("========== Inference started ==================")
        print("Press ctrl+c to stop inference")

        try:
            while True:
                ret, frame_raw = stream.read()

                if not ret:
                    print('Trying to reconnect..')
                    time.sleep(1)
                    stream = cv2.VideoCapture(camera_ip)
                    continue

                frame_counter += 1
                
                if SAVE_ANNOTATIONS:
                    # take roi..
                    if CONSIDER_ROI:
                        if frame_counter == 1:
                            # roi_points = get_roi(frame_raw)
                            roi_points = np.array([[   4, 1076],[   1,   91],[ 163,   74],[ 179,   42],[ 421,   44],[1816,  853],[1845,  652],[ 983,  308],[ 786,  115],[1910,  217],[1915, 1075],[   4, 1077]])
                
                if frame_counter % (num_frames_to_skip + 1) == 0 and not (frame_raw is None):
                    if save_output_frames:
                        output_frame = os.path.join(out_dir, "{}_{}.jpg".format(vid_name, frame_counter))
                        
                    if SAVE_ANNOTATIONS:
                        image_name = "{}_{}.jpg".format(vid_name, frame_counter)
                        cv2.imwrite(os.path.join(save_imgs_path, image_name), frame_raw)
                        output_txt = os.path.join(save_annotations_path, image_name.split('.jpg')[0] + '.txt')
                        f = open(output_txt, 'w')
                    
                    image = frame_raw.copy() 
                
                    image = frame_raw.copy() 
                        
                    boxes, scores, class_ids = get_detections(image, ort_session, input_names, input_shape, output_names, dynamic_input_shape=dynamic_input_model)
                    
                    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.3) # (bboxes, scores, score_threshold, nms_threshold)
                    # indices = cv2.dnn.NMSBoxesBatched(boxes, scores, class_ids, 0.5, 0.3)

                    image_draw = frame_raw.copy()
                    count = 1
                    for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
                        bbox = bbox.round().astype(np.int32).tolist()
                        cls_id = int(label)
                        cls = CLASSES[cls_id]
                        color = (0,255,0)
                        cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                        cv2.putText(image_draw,
                                    f'{cls}:{int(score*100)}', (bbox[0], bbox[1] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.60, [0, 0, 255],
                                    thickness=2)
                        
                        ##save annotations here..
                        if SAVE_ANNOTATIONS:
                            
                            xc = bbox[0] + (bbox[2] - bbox[0]) / 2
                            yc = bbox[1] + (bbox[3] - bbox[1]) / 2
                            
                            if CONSIDER_ROI:
                                point = Point(xc, yc)
                                polygon = Polygon(roi_points)
                                object_center_within_roi = polygon.contains(point)
                            
                            if CONSIDER_ROI and (not object_center_within_roi): #and (cls == 'axle'):
                                pass
                            else:
                                image_height, image_width = image.shape[:2]
                                input_height, input_width = input_shape[:2]
                                image_shape = np.array([image_width, image_height, image_width, image_height])
                                bbox = np.divide(bbox, image_shape, dtype=np.float32)
                                bbox = xyxy2xywh(np.array(bbox))
                                if save_predictions_score:
                                    f.write(f"{cls_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {score}\n")
                                else:
                                    f.write(f"{cls_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                            
                    if SAVE_ANNOTATIONS:
                        f.close()      

                    if save_output_frames:
                        cv2.imwrite(output_frame, image_draw)
        
                    image_draw = cv2.resize(image_draw, (frame_width, frame_height))
                    
                    if save_output_video:
                        out.write(image_draw)
                    
                    if view_output_frames:
                        cv2.imshow('inference_result', image_draw)
                        cv2.waitKey(1)

        except KeyboardInterrupt:
            if view_output_frames:
                cv2.destroyAllWindows()