mode = 'test' # train or test

epochs = 100
# optimizer: Adam 
lr = 2e-4
weight_decay = 1e-5
gamma = 0.5
weight_decay_step = 30

train_batch_size = 4
val_batch_size = 4
test_batch_size = 4
save_freq = 2
val_freq = 2
strat_save_epoch = 2

#train_data_dir = '/var/wangbinbin/data/wow_1.0/train/'
train_data_dir = '/var/wangbinbin/data/wow_1.0-100/train/'

#val_data_dir = '/var/wangbinbin/data/wow_1.0/validation/'
val_data_dir = '/var/wangbinbin/data/wow_1.0-100/validation/'

#test_data_dir = '/var/wangbinbin/data/wow_1.0/test/'
#test_data_dir = '/var/wangbinbin/HiNet/image/coco-val2017-100/test'
#test_data_dir = '/var/wangbinbin/HiNet/image/coco-val2017-500/test'
#test_data_dir = '/var/wangbinbin/HiNet/image/coco-val2017-1000/test'
#test_data_dir = '/var/wangbinbin/HiNet/image/coco-val2017-1500/test'
#test_data_dir = '/var/wangbinbin/HiNet/image/coco-val2017-2000/test'
test_data_dir = '/var/wangbinbin/HiNet/image/coco-val2017-2500/test'


stego_img_height = 256 # stego_img_height == stego_img_width
stego_img_channel = 3 # 
'''
Dataset structure      
train_data_dir/       |     val_data_dir/        |      the structure of test_data_dir/ is the same as that of train_data_dir and val_data_dir
    cover/            |         cover/
        xxx1.png      |             xxx1.png
        xxx2.png      |             xxx2.png
        ...           |             ...
    stego/            |         stego/
        xxx1.png      |             xxx1.png
        xxx2.png      |             xxx2.png
        ...           |             ...
'''

pre_trained_srnet_path = '/var/wangbinbin/Deep-Steganalysis/checkpoints/SRNet/checkpoint_100.pt'
pre_trained_yenet_path = 'checkpoints/YeNet/checkpoint_100.pt'
pre_trained_stegnet_path = 'checkpoints/StegNet/checkpoint_002.pt'
pre_trained_siastegnet_path = '/var/wangbinbin/Deep-Steganalysis/checkpoints/SiaStegNet/checkpoint_018.pt'
pre_trained_xunet_path = 'checkpoints/XuNet/checkpoint_008.pt'
pre_trained_zhunet_path = '/var/wangbinbin/Deep-Steganalysis/checkpoints/ZhuNet/checkpoint_010.pt'
