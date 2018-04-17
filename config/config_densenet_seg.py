
cmd             = 'train'  # ['train', 'test']
data_dir        = '/media/samsumg_1tb/CITYSCAPE'
num_class       = 19  # CITYSCAPE number of classes

# Training parameters
arch            = 'densenet121'  # ['densenet121', 'densenet169', 'densenet201', 'densenet161']
num_workers     = 4
crop_size       = 1024
batch_size      = 3
epochs          = 250
lr              = 0.01
momentum        = 0.9
weight_decay    = 1e-4

pretrained      = None  # './model_save_dir/checkpoint_latest.pth.tar'  # 'path to latest checkpoint (default: none)'
resume          = './model_save_dir/checkpoint_latest.pth.tar'
lr_mode         = 'poly'  # ['step', 'poly']
# data augmentation
random_scale    = 2
random_rotate   = 10  # in degrees
random_horizontal_flip = 0.5

# Validation/Test parameters
multi_scale     = False
evaluate        = False

