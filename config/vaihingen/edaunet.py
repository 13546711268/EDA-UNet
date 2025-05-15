from torch.utils.data import DataLoader
from network.losses import *
from network.datasets.vaihingen_dataset import *
from network.models.edaunet import *
from catalyst.contrib.nn import Lookahead
from catalyst import utils


# training hparam
max_epoch = 50
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
lr = 1e-3
weight_decay = 0.01
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES
token_length = num_classes
weights_name = "edaunet"
weights_path = "checkpoints/vaihingen/{}".format(weights_name)
test_weights_name = weights_name
log_name = 'vaihingen/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = [0]  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None #"checkpoints/vaihingen/delta-0817l0.2/delta-0817l0.2.ckpt"  # whether continue training with the checkpoint, default None
strategy = None

#  define the network
#  define the network
net = EDANet(num_classes=num_classes, in_channels=3, width=16, middle_blk_num=1,
                enc_blk_nums=[2,2,2,2], dec_blk_nums=[2,2,2,2])

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)

use_aux_loss = False

# define the dataloader

train_dataset = VaihingenDataset(data_root='data/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(transform=val_aug)
test_dataset = VaihingenDataset(data_root='data/vaihingen/test',
                                transform=test_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
base_optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

