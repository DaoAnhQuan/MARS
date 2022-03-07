import os
import time
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from metrics import AverageMeter, accuracy
from i3d_m2 import I3DCustom
from dataset import HMDB51
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

TRAIN_GRAD_ACCUM_STEPS = 1
TRAIN_PRINT_FREQ = 100
TEST_PRINT_FREQ = 250
TRAIN_MODALITY = 'rgb_flow'
TRAIN_BATCH_SIZE = 12
SEED = 789
LOG_DIR = 'weight_runs'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
WORKERS = 4
TRAIN_SPLIT = 'test_train_splits'
SPLIT_ORD = 1
NUM_CLASS = 51
RGB_ROOT = 'jpegs_256'
FLOW_ROOT = 'tvl1_flow'
CHECKPOINT_DIR = 'weight_models'
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)
MAX_EPOCHS = 40
EVAL_FREQ = 5


class WeightI3D(nn.Module):
    def __init__(self):
        super(WeightI3D, self).__init__()
        flow_model = I3DCustom(51, in_channels=2)
        flow_model.load_state_dict(torch.load('models/flow_split1_epoch099.pt'))
        rgb_model = I3DCustom(51, in_channels=3)
        rgb_model.load_state_dict(torch.load('models/rgb_split1_epoch099.pt'))

        for param in flow_model.parameters():
            param.requires_grad = False
        for param in rgb_model.parameters():
            param.requires_grad = False

        flow_model.eval()
        rgb_model.eval()
        self.flow_model = flow_model
        self.rgb_model = rgb_model
        self.w1 = nn.Parameter(torch.rand((1, NUM_CLASS)))
        self.w2 = nn.Parameter(torch.rand((1, NUM_CLASS)))

    def forward(self, rgb, flow):
        rgb_output, _, _, _ = self.rgb_model(rgb)
        flow_output, _, _, _ = self.flow_model(flow)

        rgb_output = torch.sigmoid(rgb_output)
        flow_output = torch.sigmoid(flow_output)

        output = self.w1 * rgb_output + self.w2 * flow_output
        return output


def train(train_loader, model, criterion, optimizer, epoch, scheduler, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for step, (rgb, flow, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        rgb = rgb.cuda(non_blocking=True)
        flow = flow.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(rgb, flow)

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(F.softmax(output, dim=1), target, topk=(1, 5))
        losses.update(loss.item(), rgb.size(0))
        top1.update(prec1[0], rgb.size(0))
        top5.update(prec5[0], rgb.size(0))

        loss = loss / TRAIN_GRAD_ACCUM_STEPS

        loss.backward()

        if step % TRAIN_GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % TRAIN_PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, step, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))

    if writer:
        writer.add_scalar('train/loss', losses.avg, epoch + 1)
        writer.add_scalar('train/top1', top1.avg, epoch + 1)
        writer.add_scalar('train/top5', top5.avg, epoch + 1)


def validate(val_loader, model, criterion, epoch, writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    # model.eval()

    with torch.no_grad():
        end = time.time()
        for step, (rgb, flow, target) in enumerate(val_loader):
            rgb = rgb.cuda(non_blocking=True)
            flow = flow.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(rgb, flow)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(F.softmax(output, dim=1), target, topk=(1, 5))

            losses.update(loss.item(), rgb.size(0))
            top1.update(prec1[0], rgb.size(0))
            top5.update(prec5[0], rgb.size(0))

            # measure elapsed time
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

        if writer:
            writer.add_scalar('val/loss', losses.avg, epoch + 1)
            writer.add_scalar('val/top1', top1.avg, epoch + 1)
            writer.add_scalar('val/top5', top5.avg, epoch + 1)

    return losses.avg


def run():
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(seed=SEED)

    # Log to tensorboard
    writer = SummaryWriter(log_dir=LOG_DIR)

    # Setup dataloaders
    train_transforms = transforms.Compose([transforms.RandomCrop((224, 224)),
                                           transforms.RandomHorizontalFlip(),
                                           ])
    test_transforms = transforms.Compose([transforms.CenterCrop((224, 224))])

    train_loader = torch.utils.data.DataLoader(
        HMDB51(TRAIN_SPLIT, SPLIT_ORD, 'train', RGB_ROOT, FLOW_ROOT, TRAIN_MODALITY, NUM_CLASS, train_transforms),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        HMDB51(TRAIN_SPLIT, SPLIT_ORD, 'test', RGB_ROOT, FLOW_ROOT, TRAIN_MODALITY, NUM_CLASS, test_transforms),
        batch_size=1,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True
    )

    model = WeightI3D()
    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(
        [model.w1, model.w2],
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0000007
    )

    for epoch in range(MAX_EPOCHS):
        train(train_loader,
              model,
              criterion,
              optimizer,
              epoch,
              None,
              writer
              )
        if (epoch + 1) % EVAL_FREQ == 0 or epoch == MAX_EPOCHS - 1:
            val_loss = validate(val_loader, model, criterion, epoch, writer)
            torch.save(
                {'w1': model.w1, 'w2': model.w2},
                CHECKPOINT_DIR + '/' + TRAIN_MODALITY + '_split' + str(SPLIT_ORD) + '_epoch' + str(epoch).zfill(
                    3) + '.pt'
            )
            torch.save(optimizer.state_dict(),
                       CHECKPOINT_DIR + '/optimizer_split' + str(SPLIT_ORD) + '_epoch' + str(epoch).zfill(3) + '.pt')
    writer.close()


if __name__ == "__main__":
    run()