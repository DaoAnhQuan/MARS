import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from metrics import AverageMeter, accuracy
from i3d_m2 import I3DCustom
from dataset import HMDB51

TRAIN_GRAD_ACCUM_STEPS = 4
TRAIN_PRINT_FREQ = 100
TEST_PRINT_FREQ = 250
TRAIN_MODALITY = 'rgb'
TRAIN_BATCH_SIZE = 8
LOG_DIR = 'rgb_runs'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
WORKERS = 4
TRAIN_SPLIT = 'test_train_splits'
SPLIT_ORD = 1
NUM_CLASS = 51
RGB_ROOT = 'jpegs_256'
FLOW_ROOT = 'tvl1_flow'
CHECKPOINT_DIR = 'rgb_models'
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)
MAX_EPOCHS = 100
EVAL_FREQ = 1


def train(train_loader, model, criterion, optimizer, epoch, scheduler=None, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for step, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output, _, _, _ = model(input)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(F.softmax(output, dim=1), target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        loss = loss / TRAIN_GRAD_ACCUM_STEPS

        loss.backward()

        if step % TRAIN_GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

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

    model.eval()

    with torch.no_grad():
        end = time.time()
        for step, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output, _, _, _ = model(input)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(F.softmax(output, dim=1), target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

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
    print("Training ", TRAIN_MODALITY, " model.")
    print("Batch size:", TRAIN_BATCH_SIZE, " Gradient accumulation steps:", TRAIN_GRAD_ACCUM_STEPS)

    writer = SummaryWriter(log_dir=LOG_DIR)

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

    if TRAIN_MODALITY == "rgb":
        channels = 3
        checkpoint = 'models/rgb_imagenet.pt'
    elif TRAIN_MODALITY == "flow":
        channels = 2
        checkpoint = 'models/flow_imagenet.pt'
    else:
        raise ValueError("Modality must be RGB or flow")

    i3d_model = I3DCustom(NUM_CLASS, in_channels=channels, i3d_load=checkpoint, dropout_p=0.36)

    i3d_model = i3d_model.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(
        i3d_model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0000007
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, threshold=1e-5)
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    for epoch in range(MAX_EPOCHS):
        train(train_loader,
              i3d_model,
              criterion,
              optimizer,
              epoch,
              None,
              writer
              )
        if (epoch + 1) % EVAL_FREQ == 0 or epoch == MAX_EPOCHS - 1:
            val_loss = validate(val_loader, i3d_model, criterion, epoch, writer)
            scheduler.step(val_loss)
            torch.save(
                i3d_model.state_dict(),
                CHECKPOINT_DIR + '/' + TRAIN_MODALITY + '_split' + str(SPLIT_ORD) + '_epoch' + str(epoch).zfill(
                    3) + '.pt'
            )
            torch.save(optimizer.state_dict(),
                       CHECKPOINT_DIR + '/optimizer_split' + str(SPLIT_ORD) + '_epoch' + str(epoch).zfill(3) + '.pt')
            torch.save(scheduler.state_dict(),
                       CHECKPOINT_DIR + '/scheduler_split' + str(SPLIT_ORD) + '_epoch' + str(epoch).zfill(3) + '.pt')

    writer.close()


if __name__ == "__main__":
    run()