import os
import torch


class Args:
  def __init__(self):
    # self.initialized = False
    self.current_path = os.path.join(os.getcwd(), 'perspective_transform')
    self.base_path = os.getcwd() 
    self.isTrain = False
    
    #perspective_transform
    self.batchSize=1
    self.loadSize=1024
    self.fineSize=1024
    self.input_nc=3
    self.output_nc=1
    self.ngf=64
    self.ndf=64
    self.which_model_netD='basic'
    self.which_model_netG='unet_256'
    self.n_layers_D=3
    self.dataset_mode='single'
    self.model='two_pix2pix'
    self.which_direction='AtoB'
    self.nThreads=1
    self.checkpoints_dir=os.path.join(self.base_path, 'weights/pytorch-two-GAN-models/soccer_seg_detection_pix2pix')
    self.norm='batch'
    self.serial_batches=True
    self.display_winsize=256
    self.display_id=1
    self.display_port=8097
    self.no_dropout=True
    self.max_dataset_size=float("inf")
    self.resize_or_crop='resize_and_crop'
    self.no_flip=True
    self.init_type='normal'
    self.name=os.path.join(self.base_path, 'PerspectiveTransform/weights/pytorch-two-GAN-models/soccer_seg_detection_pix2pix')

    # Test Options
    self.ntest=float("inf")
    self.aspect_ratio=1.0
    self.phase='test'
    self.which_epoch='latest'
    self.how_many=1

    self.initialized = True
    self.continue_train=False
    
    if torch.cuda.is_available():
        self.gpu_ids=[0]
    else:
        self.gpu_ids=[]
