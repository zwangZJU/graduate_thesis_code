import torch
import torch.nn as nn

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        # self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:

            branch_main = [
                # pw
                nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                # pw-linear
                nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
            ]

            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:

            self.conv1 = nn.Sequential(
                # pw
                nn.Conv2d(inp, mid_channels * 2, 1, 1, 0, groups=2, bias=False),
                nn.BatchNorm2d(mid_channels * 2),
                nn.ReLU(inplace=True)
            )

            branch_main = [

                # dw
                nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                # pw-linear
                nn.Conv2d(mid_channels, oup - mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup - mid_channels),
                nn.ReLU(inplace=True),
            ]

            self.branch_proj = None

        self.branch_main = nn.Sequential(*branch_main)

    def forward(self, old_x):
        if self.stride == 1:
            # x_proj, x = self.channel_clip(old_x)
            # return torch.cat((x_proj, self.branch_main(x)), 1)
            # x = self.conv1(old_x)
            identity = old_x
            old_x = self.conv1(old_x)

            x_proj, x = self.channel_clip(old_x)
            x_out = self.channel_shuffle(torch.cat((x_proj, self.branch_main(x)), 1))
            if identity.shape[1] == x_out.shape:
                return x_out + identity
            else:
                return x_out
        elif self.stride == 2:
            x_proj = old_x
            x = old_x

            return self.channel_shuffle(torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1))

    def channel_clip(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

    def channel_shuffle(self, x, group=4):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % group == 0
        group_channels = num_channels // group

        x = x.reshape(batchsize, group_channels, group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)
        return x
