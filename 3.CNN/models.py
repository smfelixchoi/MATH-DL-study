import torch
from torch import nn

class MobileNet(nn.Module):
    def __init__(self, channels, init_weights=True):
        super(MobileNet,self).__init__()
        
        self.conv = nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        
        self.dwconv1 = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, 
                                 padding=1, groups=channels[0], bias=False)
        self.bn11 = nn.BatchNorm2d(channels[0])
        self.ptconv1 = nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(channels[1])
        
        self.dwconv2 = nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1, groups=channels[1], bias=False)
        self.bn21 = nn.BatchNorm2d(channels[1])
        self.ptconv2 = nn.Conv2d(channels[1], channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(channels[2])

        self.dwconv3 = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1, groups=channels[2], bias=False)
        self.bn31 = nn.BatchNorm2d(channels[2])
        self.ptconv3 = nn.Conv2d(channels[2], channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn32 = nn.BatchNorm2d(channels[3])

        self.dwconv4 = nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1, groups=channels[3], bias=False)
        self.bn41 = nn.BatchNorm2d(channels[3])
        self.ptconv4 = nn.Conv2d(channels[3], channels[4], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn42 = nn.BatchNorm2d(channels[4])

        self.avgpool = nn.AvgPool2d(kernel_size=(4,4))
        
        self.fc1 = nn.Linear(channels[4], channels[4]//4, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[4]//4)
        self.fc2 = nn.Linear(channels[4]//4, 10, bias=True)
        
        if init_weights:
            self.initialize_weights()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.dwconv1(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.ptconv1(x)
        x = self.bn12(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        
        x = self.dwconv2(x)
        x = self.bn21(x)
        x = self.relu(x)
        x = self.ptconv2(x)
        x = self.bn22(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        
        x = self.dwconv3(x)
        x = self.bn31(x)
        x = self.relu(x)
        x = self.ptconv3(x)
        x = self.bn32(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        
        x = self.dwconv4(x)
        x = self.bn41(x)
        x = self.relu(x)
        x = self.ptconv4(x)
        x = self.bn42(x)
        x = self.relu(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
                    
                    
class MobileNet2(nn.Module):
    def __init__(self, channels, init_weights=True):
        super(MobileNet2, self).__init__()
        
        self.conv = nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        
        self.dwconv1 = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, 
                                 padding=1, groups=channels[0], bias=False)
        self.bn11 = nn.BatchNorm2d(channels[0])
        self.ptconv1 = nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(channels[1])
        
        self.dwconv2 = nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1, groups=channels[1], bias=False)
        self.bn21 = nn.BatchNorm2d(channels[1])
        self.ptconv2 = nn.Conv2d(channels[1], channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(channels[2])

        self.dwconv3 = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1, groups=channels[2], bias=False)
        self.bn31 = nn.BatchNorm2d(channels[2])
        self.ptconv3 = nn.Conv2d(channels[2], channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn32 = nn.BatchNorm2d(channels[3])

        self.dwconv4 = nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1, groups=channels[3], bias=False)
        self.bn41 = nn.BatchNorm2d(channels[3])
        self.ptconv4 = nn.Conv2d(channels[3], channels[4], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn42 = nn.BatchNorm2d(channels[4])

        self.convpool = nn.Conv2d(channels[4], channels[4], kernel_size=4, stride=1, padding=0, groups=channels[4], bias=True)
        
        self.fc1 = nn.Linear(channels[4], channels[4]//4, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[4]//4)
        self.fc2 = nn.Linear(channels[4]//4, 10, bias=True)
        
        if init_weights:
            self.initialize_weights()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.dwconv1(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.ptconv1(x)
        x = self.bn12(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        
        x = self.dwconv2(x)
        x = self.bn21(x)
        x = self.relu(x)
        x = self.ptconv2(x)
        x = self.bn22(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        
        x = self.dwconv3(x)
        x = self.bn31(x)
        x = self.relu(x)
        x = self.ptconv3(x)
        x = self.bn32(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        
        x = self.dwconv4(x)
        x = self.bn41(x)
        x = self.relu(x)
        x = self.ptconv4(x)
        x = self.bn42(x)
        x = self.relu(x)
        
        x = self.convpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
                    
                   