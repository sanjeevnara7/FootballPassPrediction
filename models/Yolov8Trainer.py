'''
Training Script for Yolov8. Yolov8 configs are added to custom config file configs/yolo/hyp-yolov8.yaml for overriding defaults.
'''
import torch
from ultralytics import YOLO

def main():
    # Load Yolo model with custom config file
    model = YOLO('configs/yolo/yolov8l.yaml').load('yolov8l.pt')  # build from YAML and transfer weights
    
    # Get Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('*'*60)
    print('GPU USAGE:')
    print('Device: ', device)
    print(torch.cuda.get_device_name(0))
    print()
    print('*'*60)
    
    # Train the model
    model.train(cfg="configs/yolo/hyp-yolov8.yaml")
    
    # Check results on validation
    results = model.val()
    print(results)

if __name__ == '__main__':
    main()
