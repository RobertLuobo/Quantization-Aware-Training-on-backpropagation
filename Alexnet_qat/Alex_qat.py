import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import time
from quantizerAlex import *
from Timer import Timer_logger
from config import cfg
Timer_logger = Timer_logger()
Timer_logger.log_info("===> training ")


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.Maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.ReLU = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(192)
        self.bn3 = nn.BatchNorm2d(384)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.ReLU(self.conv1(x))
        x = self.Maxpool(x)
        x = self.bn1(x)

        x = self.ReLU(self.conv2(x))
        x = self.Maxpool(x)
        x = self.bn2(x)

        x = self.ReLU(self.conv3(x))
        x = self.bn3(x)

        x = self.ReLU(self.conv4(x))
        x = self.bn4(x)

        x = self.ReLU(self.conv5(x))
        x = self.bn5(x)
        x = self.Maxpool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.ReLU(self.fc1(x))
        x = self.dropout(x)
        x = self.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x


def visualise(x, axs):
    x = x.view(-1).cpu().numpy()
    axs.hist(x)

def print_net_struction(net):
    global input_size, device, batch_size, Timer_logger
    from torchsummary import summary
    if input_size == 224: Timer_logger.log_info(summary(model=net.to(device), input_size=(3, 224, 224), batch_size=batch_size,
                                device="cuda"))  # model, input_size(channel, H, W), batch_size, device
    elif input_size == 32: Timer_logger.log_info(summary(model=net.to(device), input_size=(3, 32, 32), batch_size=batch_size,
                                device="cuda"))  # model, input_size(channel, H, W), batch_size, device
    else:pass

def train(args, model, device, train_loader, optimizer, epoch, test_loader):
    model.train()
    Timer_logger.start()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            Timer_logger.log()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))
            test(args, model, device, test_loader)
            Timer_logger.start()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(mnist=False):
    Timer_logger.log_info("Function:%s" %  (str(main.__name__)))
    Timer_logger.start()
    batch_size = cfg.batch_size
    test_batch_size = cfg.test_batch_size
    epochs = cfg.epoch
    lr = cfg.lr
    momentum = cfg.momentum
    input_size = cfg.input_size
    seed = cfg.seed
    log_interval = cfg.log_interval
    save_model = cfg.save_model
    no_cuda = cfg.no_cuda
    dataset_root = cfg.dataset_root

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}


    transform = transforms.Compose(
        [
         transforms.Resize(input_size),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

    trainset = datasets.CIFAR10(root=dataset_root, train=True,
                                download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=0)

    testset = datasets.CIFAR10(root=dataset_root, train=False,
                               download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False, num_workers=0)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1, epochs + 1):
        Timer_logger.log_info("normal:")
        Timer_logger.start()
        train(args, model, device, train_loader, optimizer, epoch, test_loader)
        Timer_logger.log()

        Timer_logger.log_info("test")
        Timer_logger.start()
        test(args, model, device, test_loader)
        Timer_logger.log()

    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    Timer_logger.log()
    return model


# Quantization Aware Training Forward Pass
def quantAwareTrainingForward(model, x, stats, vis=False, axs=None, sym=False, num_bits=8, act_quant=False):

    conv1weight = model.conv1.weight.data
    model.conv1.weight.data = FakeQuantOp.apply(model.conv1.weight.data, num_bits)
    x = F.relu(model.conv1(x))
    x = model.bn1(x)

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv1']['ema_min'], stats['conv1']['ema_max'])

    x = F.max_pool2d(x, 3, 2)


    conv2weight = model.conv2.weight.data
    model.conv2.weight.data = FakeQuantOp.apply(model.conv2.weight.data, num_bits)
    x = F.relu(model.conv2(x))
    x = model.bn2(x)

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv2']['ema_min'], stats['conv2']['ema_max'])

    x = F.max_pool2d(x, 3, 2)


    conv3weight = model.conv3.weight.data
    model.conv3.weight.data = FakeQuantOp.apply(model.conv3.weight.data, num_bits)
    x = F.relu(model.conv3(x))
    x = model.bn3(x)

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv3')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv3']['ema_min'], stats['conv3']['ema_max'])



    conv4weight = model.conv4.weight.data
    model.conv4.weight.data = FakeQuantOp.apply(model.conv4.weight.data, num_bits)
    x = F.relu(model.conv4(x))
    x = model.bn4(x)

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv4')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv4']['ema_min'], stats['conv4']['ema_max'])

    conv5weight = model.conv5.weight.data
    model.conv5.weight.data = FakeQuantOp.apply(model.conv5.weight.data, num_bits)
    x = F.relu(model.conv5(x))
    x = model.bn5(x)

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv5')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv5']['ema_min'], stats['conv5']['ema_max'])

    x = F.max_pool2d(x, 3, 2)
    x = F.adaptive_avg_pool2d(x,(6, 6))
    x = torch.flatten(x, 1)
    # x = x.view(-1, 1250)  # CIFAR
    x = model.dropout(x)

    fc1weight = model.fc1.weight.data
    model.fc1.weight.data = FakeQuantOp.apply(model.fc1.weight.data, num_bits)
    x = F.relu(model.fc1(x))
    x = model.dropout(x)

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc1')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['fc1']['ema_min'], stats['fc1']['ema_max'])

    fc2weight = model.fc2.weight.data
    model.fc2.weight.data = FakeQuantOp.apply(model.fc2.weight.data, num_bits)
    x = F.relu(model.fc2(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc2')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['fc2']['ema_min'], stats['fc2']['ema_max'])

    x = model.fc3(x)

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc2')


    return x, \
           conv1weight, conv2weight, conv3weight, conv4weight,conv5weight,\
           fc1weight, fc2weight, stats


# Train using Quantization Aware Training

def trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant=False, num_bits=8 ):
    model.train()
    Timer_logger.start()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, \
        conv1weight, conv2weight, conv3weight, conv4weight,conv5weight,\
        fc1weight, fc2weight, stats= quantAwareTrainingForward(model, data, stats,
                                                                   num_bits=num_bits,
                                                                   act_quant=act_quant)

        model.conv1.weight.data = conv1weight
        model.conv2.weight.data = conv2weight
        model.conv3.weight.data = conv3weight
        model.conv4.weight.data = conv4weight
        model.conv5.weight.data = conv5weight
        model.fc1.weight.data = fc1weight
        model.fc2.weight.data = fc2weight

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            Timer_logger.log()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits)
            Timer_logger.start()

    return stats


def testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=4):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, \
            conv1weight, conv2weight, conv3weight, conv4weight, conv5weight, \
            fc1weight, fc2weight, stats = quantAwareTrainingForward(model, data, stats,
                                                                    num_bits=num_bits,
                                                                    act_quant=act_quant)

            model.conv1.weight.data = conv1weight
            model.conv2.weight.data = conv2weight
            model.conv3.weight.data = conv3weight
            model.conv4.weight.data = conv4weight
            model.conv5.weight.data = conv5weight
            model.fc1.weight.data = fc1weight
            model.fc2.weight.data = fc2weight

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def mainQuantAware(mnist=False):
    Timer_logger.log_info("Function: %s" % (str(mainQuantAware.__name__)))
    print(mainQuantAware.__name__)

    batch_size = cfg.batch_size
    test_batch_size = cfg.test_batch_size
    epochs = cfg.epoch
    lr = cfg.lr
    momentum = cfg.momentum
    input_size = cfg.input_size
    seed = cfg.seed
    log_interval = cfg.log_interval
    save_model = cfg.save_model
    no_cuda = cfg.no_cuda
    start_QAT_epoch = cfg.start_QAT_epoch
    num_bits = cfg.num_bits
    dataset_root = cfg.dataset_root
    Allstart_time = time.time()


    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}


    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

    trainset = datasets.CIFAR10(root=dataset_root, train=True,
                                download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=0)

    testset = datasets.CIFAR10(root=dataset_root, train=False,
                               download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False, num_workers=0)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    args = {}
    args["log_interval"] = log_interval


    stats = {}
    for epoch in range(1, epochs + 1):
        if epoch > start_QAT_epoch:
            act_quant = True
        else:
            act_quant = False

        Timer_logger.log_info("QAT_FP/BP, act_quant"+str(act_quant))
        Timer_logger.start()
        stats = trainQuantAware(args, model, device, train_loader, test_loader, optimizer, epoch, stats, act_quant,
                                num_bits=num_bits)
        scheduler.step()
        Timer_logger.log()

        Timer_logger.log_info("testQuantAware")
        Timer_logger.start()
        testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits)
        Timer_logger.log()

    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    Timer_logger.log_info("ALl time: %.4f" %(time.time() - Allstart_time))
    return model, stats


# model = main()

Timer_logger.log_info("QAT_FP/BP, act_quant")
model, old_stats = mainQuantAware()