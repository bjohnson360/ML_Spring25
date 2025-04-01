import torch
import torch.nn as nn
import torch.nn.functional as F

class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        #depthwise separable convolution
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        #shortcut: if dimensions match, identity; otherwise, a 1x1 conv to match dimensions
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.pointwise(self.depthwise(out))))
        out = out + self.shortcut(x)
        return out

        
class BlazeFace(nn.Module):
    def __init__(self, num_classes=2):
        super(BlazeFace, self).__init__()

        # backbone using several BlazeBlocks for downsampling
        self.layer1 = BlazeBlock(3,16,stride=2)
        self.layer2 = BlazeBlock(16, 32, stride=2)
        self.layer3 = BlazeBlock(32, 64, stride=2) 
        self.layer4 = BlazeBlock(64, 128, stride=2)

        # detection heads on the final feature map (8x8)
        # bounding box head: outputs 4 values per spatial location
        self.conv_bbox = nn.Conv2d(128, 4, kernel_size=1)
        #classification head: outputs face/background scores per spatial location
        self.conv_cls = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x): #backbone feature extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #detection heads
        bbox = self.conv_bbox(x)   
        cls = self.conv_cls(x)     
        return bbox, cls
    

if __name__ == "__main__":
    # Quick test
    model = BlazeFace()
    dummy = torch.randn(1, 3, 128, 128)
    bbox, cls = model(dummy)
    print("BBox shape:", bbox.shape)  # Expected [1, 4, 8, 8]
    print("Cls shape:", cls.shape)    # Expected [1, 2, 8, 8]