import os
import sys
from utilsv5 import get_arguments
sys.path.append("../")
from train import run
from utils.general import colorstr


def main(train_json_path):
    
    os.environ["train_json_path"] = train_json_path
    
    train_opts = get_arguments(train_json_path)

    if train_opts.mlflow_logging:
        
        from mlflowutils import set_logging_url
        import mlflow
        
        global logging_url
        logging_url = train_opts.mlflow_logging_url
        mlflow.set_tracking_uri(logging_url) 
        set_logging_url(logging_url)
        
        os.environ["AWS_ACCESS_KEY_ID"] = "admin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = train_opts.mlflow_logging_url 

    if train_opts.task == 'detect':
        run(
            model=train_opts.model_path,
            data=train_opts.data_path,
            epochs=train_opts.epochs,
            batch_size=train_opts.batch_size,
            patience=train_opts.patience,
            nosave= train_opts.nosave,
            save_period= train_opts.save_period,
            cache= train_opts.cache,
            workers= train_opts.workers,
            exist_ok= train_opts.exist_ok,
            weights= train_opts.pretrained,
            optimizer= train_opts.optimizer,
            verbose= train_opts.verbose,
            seed= train_opts.seed,
            deterministic= train_opts.deterministic,
            single_cls= train_opts.single_cls,
            rect= train_opts.rect,
            cos_lr= train_opts.cos_lr,
            close_mosaic= train_opts.close_mosaic,
            resume= train_opts.resume,
            amp= train_opts.amp,
            fraction= train_opts.fraction,
            profile= train_opts.profile,
            dfl= train_opts.dfl,
            pose= train_opts.pose,
            kobj= train_opts.kobj,
            label_smoothing= train_opts.label_smoothing,
            nbs= train_opts.nbs,
            overlap_mask= train_opts.overlap_mask,
            mask_ratio= train_opts.mask_ratio,
            dropout= train_opts.dropout,
            noval= train_opts.noval,
            multi_scale= train_opts.multi_scale,
            imgsz=train_opts.image_size,
            device=train_opts.device,
            project=train_opts.project_name,
            name=train_opts.exp_name,
            cfg=train_opts.cfg,
            hyp=train_opts.hyp,
            evolve=train_opts.evolve,
            freeze=train_opts.freeze,
            noplots=train_opts.noplots,
            sync_bn=train_opts.sync_bn,
            image_weights=train_opts.image_weights,
            quad=train_opts.quad,
            noautoanchor=train_opts.noautoanchor,
        )
        
    elif train_opts.task == 'segment':
        print(colorstr("red", "segmentation training is not implemented yet.."))
        
    elif train_opts.task == 'pose':
        print(colorstr("red", "pose estimation training is not implemented yet..")) 
        
    elif train_opts.task == 'classify':
        print(colorstr("red", "classification training is not implemented yet.."))
        
    else:
        print(f"Please select a proper task type : {{'detect', 'classify' 'segment', 'pose'}} \nExiting..")
        sys.exit()
        
    
if __name__ == '__main__':
    
    train_json_path = "./config_detection.json"  # choose "./config_classification.json" or "./config_detection.json" or "./config_segmentation.json" or "./config_pose.json" based on task type
    main(train_json_path)