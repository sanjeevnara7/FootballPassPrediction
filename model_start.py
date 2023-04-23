import sys
sys.path.append('yolov7-main')

import torch
from models.yolo import Model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())

def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

if __name__ == '__main__':
    print('*'*100)
    print('Initializing Yolo model...')
    print('yolov7-main/cfg/training/yolov7-d6.yaml')
    print('*'*100)
    model = Model(cfg='yolov7-main/cfg/training/yolov7-d6.yaml',ch=3, nc=2)
    trainable_params, total_params = count_parameters(model)
    print('Trainable Params: ',trainable_params)
    print('Total Params: ', total_params)
    ckpt = torch.load('downloads/yolov7-d6.pt', map_location='cpu')
    state_dict = ckpt['model'].float().state_dict()
    state_dict = intersect_dicts(state_dict, model.state_dict())
    model.load_state_dict(state_dict, strict=False)
    print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), 'downloads/yolov7-d6.pt'))