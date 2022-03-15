from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import evaluator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
objects = pd.read_json("./dataset/objects.json", typ='series')
batch_size = 4
nz = 32
ngf = 64
ndf = 64
nc = 3
lr = 0.0003


class trainLoader(data.Dataset):
    def __init__(self):
        train_json = pd.read_json("./dataset/train.json", typ='series', dtype='list')
        self.img = []
        self.label = []
        for key in train_json.keys():
            self.img.append(key)
            tmp = np.zeros(24, dtype='float32')
            for i in train_json[key]:
                tmp[objects[i]] = 1
            self.label.append(tmp)
        print("> Found %d images..." % (len(self.img)))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        path = "./iclevr/" + self.img[index]
        img = Image.open(path).convert('RGB')
        img = transforms.Resize((64, 64))(img)
        img = transforms.ToTensor()(img)
        return img, self.label[index]


class testLoader(data.Dataset):
    def __init__(self):
        test_json = pd.read_json("./dataset/new_test.json", typ='series', dtype='list')
        self.label = []
        for i in test_json:
            tmp = np.zeros(24, dtype='float32')
            for j in i:
                tmp[objects[j]] = 1
            self.label.append(tmp)
        print("> Found %d images..." % (len(self.label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.label[index]


train_data = trainLoader()
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = testLoader()
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 25, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.squeeze()


netG = Generator().to(device)
netG.apply(weights_init)
print(netG)
netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)

criterion = nn.BCELoss()
# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


def train(niter):
    total_d =[]
    total_g = []
    for epoch in range(niter):
        for i, data in enumerate(trainloader, 0):
            label = data[1].to(device)
            real_label = torch.ones((len(label), 1), device=device)
            fake_label = torch.zeros((len(label), 1), device=device)
            label1 = torch.cat((label, real_label), dim=-1)
            label0 = torch.cat((label, fake_label), dim=-1)
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_data = data[0].to(device)
            size_now = real_data.size(0)

            output = netD(real_data)
            errD_real = 0
            errD_real = errD_real+criterion(output, label1)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(size_now, nz-24, device=device)
            noise = torch.cat((noise, label), dim=-1).view(size_now, nz, 1, 1)
            fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = 0
            errD_fake = errD_fake+criterion(output, label0)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            output = netD(fake)
            errG = criterion(output, label1)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            if i % 150 == 0:
                print('[%2d/%2d][%4d/%4d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, niter, i, len(trainloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                total_d.append(errD)
                total_g.append(errG)
        for param_group in optimizerD.param_groups:
            param_group['lr'] = param_group['lr']*0.96
        for param_group in optimizerG.param_groups:
            param_group['lr'] = param_group['lr']*0.96
            print("learning rate:%.8f" % param_group['lr'])
    torch.save(netG.state_dict(), 'my_netG_weight.pth')
    torch.save(netD.state_dict(), 'my_netD_weight.pth')
    x_axis = [(k + 1) for k in range(len(total_g))]
    plt.title("Loss", fontsize=20)
    plt.ylabel("loss", fontsize=16)
    plt.xlabel("step", fontsize=16)
    plt.plot(x_axis, total_g, label="generator", color='b', linewidth=0.3)
    plt.plot(x_axis, total_d, label="discriminator", color='r', linewidth=0.3)
    plt.legend(loc='upper right')
    plt.savefig("Loss.png", dpi=600)
    plt.clf()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", dest="mode", type=int)
    args = parser.parse_args()
    if args.mode == 0:
        train(25)
    else:
        netG.load_state_dict(torch.load("my_netG_weight.pth"))
        evaluation = evaluator.evaluation_model()
        result = torch.empty(0, 3, 64, 64).to(device)
        total_acc = 0.0
        i = 0
        for i, label in enumerate(testloader, 0):
            label = label.to(device)
            noise = torch.randn(len(label), nz-24, device=device)
            noise = torch.cat((noise, label), dim=-1).view(len(label), nz, 1, 1)
            fake = netG(noise)
            img = torch.empty(0, 3, 64, 64).to(device)
            for j in range(len(fake)):
                img = torch.cat((img,
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(fake[j]).view(1, 3, 64, 64)))
            total_acc = total_acc + evaluation.eval(img, label)
            result = torch.cat((result, fake))
        print("Accuracy = %.4f" % (total_acc/i))
        transforms.ToPILImage()(make_grid(result.cpu(), nrow=8)).show()
