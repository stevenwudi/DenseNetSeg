
cmd             = 'train'  # ['train', 'test']
data_dir        = '/media/samsumg_1tb/CITYSCAPE'
num_class       = 19  # CITYSCAPE number of classes

# Training parameters
arch            = 'densenet121'  # ['densenet121', 'densenet169', 'densenet201', 'densenet161']
num_workers     = 4
crop_size       = 1024
batch_size      = 2
epochs          = 500
lr              = 0.01
momentum        = 0.9
weight_decay    = 1e-4

pretrained      = None  #'./model_save_dir/checkpoint_070.pth.tar' # './model_save_dir/checkpoint_250.pth.tar'  # 'path to latest checkpoint (default: none)'
resume          = None # './model_save_dir/checkpoint_010.pth.tar'
lr_mode         = 'poly'  # ['step', 'poly']
# data augmentation
random_scale    = 2
random_rotate   = 10  # in degrees
random_horizontal_flip = 0.5

# densenet architectures
transition_layer  = (3, 5, 7, 9, 11)
conv_num_features = (64, 256, 512, 1024, 1024)
out_channels_num  = (1, 2, 4, 8, 16)  # (1, 2, 4, 8, 16)  (1, 1, 1, 1, 1)
dilation          = (2, 2, 4, 8, 16)  # (1, 2, 4, 8, 16)  (1, 1, 1, 1, 1)
ppl_out_channels_num = (32, 64, 128, 256)

# Validation/Test parameters
multi_scale     = False
evaluate        = False

