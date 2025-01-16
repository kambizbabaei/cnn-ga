"""
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import data_loader
import os
from datetime import datetime
import multiprocessing
from utils import StatusUpdateTool

class BasicBlock(nn.Module):
    expansion = 1

    # ADDED: groups parameter with default=1
    def __init__(self, in_planes, planes, stride=1, groups=1):
        super(BasicBlock, self).__init__()

        # UPDATED: include groups in the Conv2d layers
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False, groups=groups
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False, groups=groups
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GroupedPointwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=2, interleave=True):
        super(GroupedPointwiseBlock, self).__init__()
        self.groups = groups
        self.interleave = interleave

        # Ensure divisibility
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        # Create parallel 1x1 conv branches
        self.branch_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels // groups,
                out_channels // groups,
                kernel_size=1,
                bias=False
            ) for _ in range(groups)
        ])

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 1) Split input by channels into 'groups'
        splits = torch.chunk(x, self.groups, dim=1)

        # 2) Pass each split through its corresponding 1x1 conv
        out_splits = [conv(split) for conv, split in zip(self.branch_convs, splits)]

        # 3) Concatenate outputs along channel dimension
        out = torch.cat(out_splits, dim=1)  # shape: (N, out_channels, H, W)

        # 4) Optionally apply channel shuffle
        if self.interleave and self.groups > 1:
            out = self.channel_shuffle(out, self.groups)

        # 5) Batch norm + ReLU
        out = self.bn(out)
        out = F.relu(out)
        return out

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        # reshape: (batch, groups, channels_per_group, height, width)
        x = x.view(batch_size, groups, channels_per_group, height, width)
        # transpose to shuffle
        x = x.transpose(1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)
        return x


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        # The GA code generator will append lines here
        # to create conv/pool/linear modules.
        #generated_init

    def forward(self, x):
        # The GA code generator will append lines here
        # to create forward pass logic for each module.
        #generate_forward

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class TrainModel(object):
    def __init__(self):
        # Setup dataset paths and loaders
        data_dir = os.path.expanduser('./dataset')
        trainloader, validate_loader = data_loader.get_train_valid_loader(
            data_dir, batch_size=128, augment=True,
            valid_size=0.1, shuffle=True,
            random_seed=2312390, show_sample=False,
            num_workers=1, pin_memory=True
        )
        # testloader = data_loader.get_test_loader(
        #     data_dir, batch_size=128, shuffle=False,
        #     num_workers=1, pin_memory=True
        # )

        net = EvoCNNModel()
        cudnn.benchmark = True
        net = net.cuda()

        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0

        self.net = net
        self.criterion = criterion
        self.best_acc = best_acc
        self.trainloader = trainloader
        self.validate_loader = validate_loader
        self.file_id = os.path.basename(__file__).split('.')[0]

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        log_dir = './log'
        os.makedirs(log_dir, exist_ok=True)
        with open(f'{log_dir}/{self.file_id}.txt', file_mode) as f:
            f.write(f'[{dt_str}]-{_str}\n')
            f.flush()

    def train(self, epoch):
        self.net.train()
        # Simple LR schedule
        if epoch == 0:
            lr = 0.01
        elif epoch > 248:
            lr = 0.001
        elif epoch > 148:
            lr = 0.01
        else:
            lr = 0.1

        optimizer = optim.SGD(
            self.net.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4
        )

        running_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        self.log_record(f'Train-Epoch:{epoch + 1:3d},  Loss: {train_loss:.3f}, Acc:{train_acc:.3f}')

    def test(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for _, data in enumerate(self.validate_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        validate_loss = test_loss / total
        validate_acc = correct / total
        if validate_acc > self.best_acc:
            self.best_acc = validate_acc

        self.log_record(f'Validate-Loss:{validate_loss:.3f}, Acc:{validate_acc:.3f}')

    def process(self):
        total_epoch = StatusUpdateTool.get_epoch_size()
        for p in range(total_epoch):
            self.train(p)
            self.test(p)
        return self.best_acc


class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        m = TrainModel()
        try:
            m.log_record(f'Used GPU#{gpu_id}, worker name:{multiprocessing.current_process().name}[{os.getpid()}]',
                        first_time=True)
            best_acc = m.process()
        except BaseException as e:
            print(f'Exception occurs, file:{file_id}, pid:{os.getpid()}...{str(e)}')
            m.log_record(f'Exception occur:{str(e)}')
        finally:
            m.log_record(f'Finished-Acc:{best_acc:.3f}')
            populations_dir = './populations'
            os.makedirs(populations_dir, exist_ok=True)
            with open(f'{populations_dir}/after_{file_id[4:6]}.txt', 'a+') as f:
                f.write(f'{file_id}={best_acc:.5f}\n')
                f.flush()
"""
