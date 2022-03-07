import time
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from metrics import AverageMeter, accuracy
from i3d_m2 import I3DCustom
from dataset import HMDB51
import torch.nn.functional as F

TEST_PRINT_FREQ = 250
WORKERS = 4
TRAIN_SPLIT = 'test_train_splits'
SPLIT_ORD = 1
NUM_CLASS = 51
RGB_ROOT = 'jpegs_256'
FLOW_ROOT = 'tvl1_flow'
EVAL_FREQ = 1
test_transforms = transforms.Compose([transforms.CenterCrop((224, 224))])
val_loader = torch.utils.data.DataLoader(
    HMDB51(TRAIN_SPLIT, SPLIT_ORD, 'test', RGB_ROOT, FLOW_ROOT, 'rgb', NUM_CLASS, test_transforms),
    batch_size=1,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=True
)
batch_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
criterion = torch.nn.CrossEntropyLoss().cuda()

channels = 3
checkpoint = 'models/rgb_split1_epoch099.pt'
rgb_model = I3DCustom(NUM_CLASS, in_channels=channels)
rgb_model = rgb_model.cuda()
rgb_model.load_state_dict(torch.load(checkpoint))
rgb_model.eval()

channels = 3
checkpoint = 'models/rgb_flow_split1_epoch099.pt'
flow_model = I3DCustom(NUM_CLASS, in_channels=channels)
flow_model = flow_model.cuda()
flow_model.load_state_dict(torch.load(checkpoint)['MARS_model'])
flow_model.eval()

with torch.no_grad():
    end = time.time()
    for step, (rgb, target) in enumerate(val_loader):
        rgb = rgb.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        rgb_output, _, _, _ = rgb_model(rgb)
        flow_output, _, _, _ = flow_model(rgb)
        output = rgb_output + flow_output
        loss = criterion(output, target)

        prec1, prec5 = accuracy(F.softmax(output, dim=1), target, topk=(1, 5))

        losses.update(loss.item(), rgb.size(0))
        top1.update(prec1[0], rgb.size(0))
        top5.update(prec5[0], rgb.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if step % TEST_PRINT_FREQ == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                step, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))