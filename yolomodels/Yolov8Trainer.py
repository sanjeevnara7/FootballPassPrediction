'''
Simple Training Script for Yolov8. Yolov8 configs are added to custom config file configs/yolo/hyp-yolov8.yaml for overriding defaults.
'''
import torch
from ultralytics import YOLO

def main():
    CONFIG_FILE_PATH = 'configs/yolo/hyp-yolov8.yaml'
    # Load Yolo model pretrained on SoccerNet Tracking Frames
    model = YOLO('runs/YOLOv8/yolov8l_trackletframes-2.5k_fixed_res_1080/weights/best.pt')
    
    #model = YOLO('configs/yolo/yolov8l.yaml').load('yolov8l.pt') #Load Yolov8 with MS COCO weights
    
    # Get Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('*'*60)
    print('GPU USAGE:')
    print('Device: ', device)
    print(torch.cuda.get_device_name(0))
    print()
    print('*'*60)
    
    # Train the model
    model.train(cfg=CONFIG_FILE_PATH)
    print('*'*60)
    print('*'*60)
    # Check results on validation
    results = model.val(imgsz=1080)
    print(results)

if __name__ == '__main__':
    main()
