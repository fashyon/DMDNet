from os.path import join

# experiment specifics

name = 'DMDNet' #'name of the experiment. It decides where to store samples and models'
gpu_ids = '3' #'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
checkpoints_dir = './checkpoints' #'models are saved here'
seed = 2018 #'random seed to use. Default=2018'
isTrain = True

# Pretrained model path for depth estimation
depth_estimation_model = './pretrained/dpt_next_vit_large_384.pt'
vgg_model = './pretrained/vgg19-dcbb9e9d.pth'

# modify the following code to
datadir_VOC2012 = '../data/mytrain/train/VOCdevkit/VOC2012/VOC2012_crop352'
datadir_VOC2012_txt = './VOC2012_352_train_png.txt'
datadir = '../data/mytrain/train/'
datadir_test = '../data/mytrain/test'
datadir_real = join(datadir, 'real')
datadir_nature = join(datadir, 'nature')



dims = [48,96,192,384,768]

num_experts = 4
top_experts = 2

model_type = 'dpt_next_vit_large_384'

resume_epoch = None
# for training
lr = 1e-4 #initial learning rate for adam
wd = 0 #weight decay for adam
low_sigma = 2 #min sigma in synthetic dataset
high_sigma = 5 #max sigma in synthetic dataset
low_gamma = 1.3 #max gamma in synthetic dataset
high_gamma = 1.3 #max gamma in synthetic dataset

# data augmentation
batchsize = 1 #input batch size
patchsize = (352,352) #patchsize
check_size = 32
# for setting input
serial_batches = False #'if true, takes images in order to make batches, otherwise takes them randomly'
nThreads = 8 #'# threads for loading data'
# nThreads = 0 #'# threads for loading data'
max_dataset_size = None #'Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.'

total_epochs = 60

# loss weight
lambda_L1_T = 1
lambda_L1_R = 1
lambda_vgg_T = 0.02
lambda_triplet_T = 0.1
lambda_triplet_R = 0.05
lambda_align_T = 0.1
lambda_align_R = 0.05
lambda_load_T = 0.008
lambda_load_R = 0.008


# for network


# for displays
no_html = False #do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/
save_epoch_freq = 5 #frequency of saving checkpoints at the end of epochs
# for display
no_log = False #'disable tf logger?'
display_winsize = 256 #'display window size'
display_port = 8097 #'visdom port of the web display'
display_id = 0 #'window id of the web display (use 0 to disable visdom)'
display_single_pane_ncols = 0 #'if positive, display all images in a single visdom web panel with certain number of images per row.'

