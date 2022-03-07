import os
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from metrics import AverageMeter, accuracy
from i3d_m2 import I3DCustom
from dataset import HMDB51
import numpy as np

TRAIN_PRINT_FREQ = 100
TEST_PRINT_FREQ = 250
TRAIN_MODALITY = 'rgb_flow'
TRAIN_BATCH_SIZE = 8
SEED = 789
LOG_DIR = 'mars_avg_runs'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
WORKERS = 4
TRAIN_SPLIT = 'test_train_splits'
SPLIT_ORD = 1
NUM_CLASS = 51
RGB_ROOT = 'jpegs_256'
FLOW_ROOT = 'tvl1_flow'
CHECKPOINT_DIR = 'mars_avg_models'
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)
MAX_EPOCHS = 100
EVAL_FREQ = 5


def train(train_loader, rgb_model, flow_model, criterion, mse, optimizer, epoch, scheduler=None, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    distilation_losses = AverageMeter()
    cls_losses = AverageMeter()
    total_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    rgb_model.train()

    end = time.time()
    for step, (rgb, flow, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        rgb = rgb.cuda(non_blocking=True)
        flow = flow.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        rgb_output, rgb_features, _, _ = rgb_model(rgb)
        _, flow_features, _, _ = flow_model(flow)
        flow_features.detach()
        # output = torch.mean(output, dim=2)
        cls_loss = criterion(rgb_output, target)
        distilation_loss = mse(rgb_features, flow_features)
        total_loss = cls_loss + 50 * distilation_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(rgb_output, target, topk=(1, 5))
        total_losses.update(total_loss.item(), rgb.size(0))
        cls_losses.update(cls_loss.item(), rgb.size(0))
        distilation_losses.update(distilation_loss.item(), rgb.size(0))
        top1.update(prec1[0], rgb.size(0))
        top5.update(prec5[0], rgb.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % TRAIN_PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Total loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'
                   'CLS loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                   'Distilation loss {distilation_loss.val:.4f} ({distilation_loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, step, len(train_loader), batch_time=batch_time,
                data_time=data_time, total_loss=total_losses,
                cls_loss=cls_losses, distilation_loss=distilation_losses,
                top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))

    if writer:
        writer.add_scalar('train/total_loss', total_losses.avg, epoch + 1)
        writer.add_scalar('train/cls_loss', cls_losses.avg, epoch + 1)
        writer.add_scalar('train/distilation_loss', distilation_losses.avg, epoch + 1)
        writer.add_scalar('train/top1', top1.avg, epoch + 1)
        writer.add_scalar('train/top5', top5.avg, epoch + 1)


def validate(val_loader, rgb_model, flow_model, criterion, mse, epoch, writer=None):
    batch_time = AverageMeter()
    distilation_losses = AverageMeter()
    cls_losses = AverageMeter()
    total_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    rgb_model.eval()

    with torch.no_grad():
        end = time.time()
        for step, (rgb, flow, target) in enumerate(val_loader):
            rgb = rgb.cuda(non_blocking=True)
            flow = flow.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            rgb_output, _, rgb_features, _ = rgb_model(rgb)
            _, _, flow_features, _ = flow_model(flow)
            cls_loss = criterion(rgb_output, target)
            distilation_loss = mse(rgb_features, flow_features)
            total_loss = cls_loss + 50 * distilation_loss

            # measure accuracy and record loss
            prec1, prec5 = accuracy(rgb_output, target, topk=(1, 5))

            total_losses.update(total_loss.item(), rgb.size(0))
            cls_losses.update(cls_loss.item(), rgb.size(0))
            distilation_losses.update(distilation_loss.item(), rgb.size(0))
            top1.update(prec1[0], rgb.size(0))
            top5.update(prec5[0], rgb.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % TEST_PRINT_FREQ == 0:
                print(('Test: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Total loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'
                       'CLS loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                       'Distilation loss {distilation_loss.val:.4f} ({distilation_loss.avg:.4f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    step, len(val_loader), batch_time=batch_time, total_loss=total_losses,
                    cls_loss=cls_losses, distilation_loss=distilation_losses,
                    top1=top1, top5=top5)))

        print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} '
               'Total Loss {total_loss.avg:.5f} CLS Loss {cls_loss.avg:.5f} '
               'Distilation Loss {distilation_loss.avg:.5f}'
               .format(top1=top1, top5=top5, total_loss=total_losses, cls_loss=cls_losses,
                       distilation_loss=distilation_losses)))

        if writer:
            writer.add_scalar('val/total_loss', total_losses.avg, epoch + 1)
            writer.add_scalar('val/cls_loss', cls_losses.avg, epoch + 1)
            writer.add_scalar('val/distilation_loss', distilation_losses.avg, epoch + 1)
            writer.add_scalar('val/top1', top1.avg, epoch + 1)
            writer.add_scalar('val/top5', top5.avg, epoch + 1)

    return total_losses.avg


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

    rgb_model = I3DCustom(NUM_CLASS, in_channels=3)
    flow_model = I3DCustom(NUM_CLASS, in_channels=2)

    flow_model.load_state_dict(torch.load('models/flow_split1_epoch099.pt'))
    rgb_model.load_state_dict(torch.load('models/rgb_split1_epoch099.pt'))

    rgb_model = rgb_model.cuda()
    flow_model = flow_model.cuda()

    for param in flow_model.parameters():
        param.requires_grad = False
    flow_model.eval()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    mse = torch.nn.MSELoss().cuda()

    optimizer = optim.SGD(
        rgb_model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-5)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, threshold=1e-5)

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    for epoch in range(0, MAX_EPOCHS):
        train(train_loader,
              rgb_model,
              flow_model,
              criterion,
              mse,
              optimizer,
              epoch,
              None,
              writer
              )
        if (epoch + 1) % EVAL_FREQ == 0 or epoch == MAX_EPOCHS - 1:
            state_dict = {
                'epoch': epoch,
                'MARS_model': rgb_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(
                state_dict,
                CHECKPOINT_DIR + '/' + TRAIN_MODALITY + '_split' + str(SPLIT_ORD) + '_epoch' + str(epoch).zfill(
                    3) + '.pt'
            )
        val_loss = validate(val_loader, rgb_model, flow_model, criterion, mse, epoch, writer)
        scheduler.step(val_loss)
    writer.close()


if __name__ == "__main__":
    run()