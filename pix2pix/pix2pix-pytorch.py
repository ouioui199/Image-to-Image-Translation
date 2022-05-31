import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normal_init(m, mean ,std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, filters=64, kernel_size=4, strides=2, padding=1, **kwargs):
        super(Generator, self).__init__(**kwargs)

        # Unet Encoder
        self.conv1 = nn.Conv2d(3, filters, kernel_size, strides, padding=1, bias=False)

        self.conv2 = nn.Conv2d(filters, filters*2, kernel_size, strides, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(filters*2)
        
        self.conv3 = nn.Conv2d(filters*2, filters*4, kernel_size, strides, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(filters*4)

        self.conv4 = nn.Conv2d(filters*4, filters*8, kernel_size, strides, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(filters*8)

        self.conv5 = nn.Conv2d(filters*8, filters*8, kernel_size, strides, padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(filters*8)

        self.conv6 = nn.Conv2d(filters*8, filters*8, kernel_size, strides, padding=1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(filters*8)

        self.conv7 = nn.Conv2d(filters*8, filters*8, kernel_size, strides, padding=1, bias=False)
        self.conv7_bn = nn.BatchNorm2d(filters*8)

        # self.conv8 = nn.Conv2d(filters*8, filters*8, kernel_size, strides, bias=False)
        # self.conv8_bn = nn.BatchNorm2d(filters*8)

        self.bottleneck = nn.Conv2d(filters*8, filters*8, kernel_size, strides, padding=1, bias=False)

        # Unet Decoder
        self.deconv1 = nn.ConvTranspose2d(filters*8, filters*8, kernel_size, strides, padding=1, bias=False)
        self.deconv1_bn = nn.BatchNorm2d(filters*8)

        self.deconv2 = nn.ConvTranspose2d(filters*8*2, filters*8, kernel_size, strides, padding=1, bias=False)
        self.deconv2_bn = nn.BatchNorm2d(filters*8)

        self.deconv3 = nn.ConvTranspose2d(filters*8*2, filters*8, kernel_size, strides, padding=1, bias=False)
        self.deconv3_bn = nn.BatchNorm2d(filters*8)

        self.deconv4 = nn.ConvTranspose2d(filters*8*2, filters*8, kernel_size, strides, padding=1, bias=False)
        self.deconv4_bn = nn.BatchNorm2d(filters*8)

        self.deconv5 = nn.ConvTranspose2d(filters*8*2, filters*4, kernel_size, strides, padding=1, bias=False)
        self.deconv5_bn = nn.BatchNorm2d(filters*4)

        self.deconv6 = nn.ConvTranspose2d(filters*4*2, filters*2, kernel_size, strides, padding=1, bias=False)
        self.deconv6_bn = nn.BatchNorm2d(filters*2)

        self.deconv7 = nn.ConvTranspose2d(filters*2*2, filters, kernel_size, strides, padding=1, bias=False)
        self.deconv7_bn = nn.BatchNorm2d(filters)

        self.deconv8 = nn.ConvTranspose2d(filters*2, 3, kernel_size, strides, padding=1, bias=False)

    def weight_init(self, mean, std): # A enlever, ne pas init à chaque fois
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):

        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))

        b = self.bottleneck(F.leaky_relu(e7, 0.2))

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(b))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        
        d8 = self.deconv8(F.relu(d7))

        out = F.tanh(d8)

        return out

class Discriminator(nn.Module):
    def __init__(self, filters=64, kernel_size=4, strides=2, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.conv1 = nn.Conv2d(3*2, filters, kernel_size, strides, padding=1, bias=False)

        self.conv2 = nn.Conv2d(filters, filters*2, kernel_size, strides, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(filters*2)

        self.conv3 = nn.Conv2d(filters*2, filters*4, kernel_size, strides, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(filters*4)

        self.conv4 = nn.Conv2d(filters*4, filters*8, kernel_size, strides, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(filters*8)

        # self.conv5 = nn.Conv2d(filters*8, filters*8, kernel_size, strides, padding=1, bias=False)
        # self.conv5_bn = nn.BatchNorm2d(filters*8)

        self.conv6 = nn.Conv2d(filters*8, 1, kernel_size, padding=1, bias=False)

    def weight_init(self, mean, std): # il ne faut pas init a chaque fois
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, target):
        x = torch.cat([input, target], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        # x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = nn.ZeroPad2d((1,0,1,0))(x)
        x = F.sigmoid(self.conv6(x))

        return x

img = torch.randn(1,3,256,256)
gen = Generator()
output = gen(img)

dis = Discriminator()
val = dis(output, img)
print(val.shape)


