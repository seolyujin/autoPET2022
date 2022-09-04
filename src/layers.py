import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np                                                

class image_trans(nn.Module):

    def __init__(self, nf=16):
        super(image_trans, self).__init__()

        self.conv0 = nn.Conv3d(1, nf, kernel_size=3, padding=1) #64-64
        self.bn0 = nn.BatchNorm3d(nf)
        self.conv1 = nn.Conv3d(nf, nf*2, kernel_size=3, padding=1, stride=2) #64-32
        self.bn1 = nn.BatchNorm3d(nf*2)
        self.conv2 = nn.Conv3d(nf*2, nf*4, kernel_size=3, padding=1, stride=2) #32-16
        self.bn2 = nn.BatchNorm3d(nf*4)
        self.conv3 = nn.Conv3d(nf * 4, nf * 8, kernel_size=3, padding=1, stride=2)  # 16-8
        self.bn3 = nn.BatchNorm3d(nf * 8)

        self.bottleneck0 = nn.Conv3d(nf*8, nf*8, kernel_size=3, padding=1) #8-8
        self.bnb0 = nn.BatchNorm3d(nf * 8)
        self.bottleneck1 = nn.Conv3d(nf*8, nf*8, kernel_size=3, padding=1) #8-8
        self.bnb1 = nn.BatchNorm3d(nf * 8)

        self.up31 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # 8-16
        self.pad3 = nn.ConstantPad3d(1, 0)
        self.up32 = nn.Conv3d(nf * 8, nf * 4, kernel_size=3, padding=0)
        self.drop3 = nn.Dropout(0.5)

        self.up21 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) #16-32
        self.pad2 = nn.ConstantPad3d(1, 0)
        self.up22 = nn.Conv3d(nf*4 + nf*4, nf*2, kernel_size=3, padding=0)
        self.drop2 = nn.Dropout(0.5)

        self.up11 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) #32-64
        self.pad1 = nn.ConstantPad3d(1, 0)
        self.up12 = nn.Conv3d(nf*2 + nf*2, nf, kernel_size=3, padding=0)
        self.drop1 = nn.Dropout(0.5)

        self.pad0 = nn.ConstantPad3d(1, 0)
        self.output = nn.Conv3d(nf + nf, 1, kernel_size=3, padding=0)
        

    def forward(self, x):

        c0 = F.relu(self.bn0(self.conv0(x)))
        c1 = F.relu(self.bn1(self.conv1(c0)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn3(self.conv3(c2)))

        b0 = F.relu(self.bnb0(self.bottleneck0(c3)))
        b1 = F.relu(self.bnb1(self.bottleneck1(b0)))

        u3 = F.relu(self.up32(self.pad3(self.up31(b1))))
        u3cat = self.drop3(torch.cat([u3, c2], 1))
        u2 = F.relu(self.up22(self.pad2(self.up21(u3cat))))
        u2cat = self.drop2(torch.cat([u2, c1], 1))
        u1 = F.relu(self.up12(self.pad1(self.up11(u2cat))))
        u1cat = self.drop1(torch.cat([u1, c0], 1))
        out = self.output(self.pad0(u1cat)) + x

        return torch.tanh(out)

    
class AffineSTN3D(nn.Module):

    def __init__(self, input_size, device, input_channels=1, nf=2):
        super(AffineSTN3D, self).__init__()
        self.dtype = torch.float
        self.device = device
        self.nf = nf
        self.conv00 = nn.Conv3d(input_channels, self.nf, kernel_size=3, padding=1).to(self.device)
        self.bn00 = nn.BatchNorm3d(self.nf).to(self.device)
        self.conv0 = nn.Conv3d(self.nf, self.nf * 2, kernel_size=3, padding=1, stride=2).to(self.device)
        self.bn0 = nn.BatchNorm3d(self.nf * 2).to(self.device)
        self.conv1 = nn.Conv3d(self.nf * 2, self.nf * 4, kernel_size=3, padding=1, stride=2).to(self.device)
        self.bn1 = nn.BatchNorm3d(self.nf * 4).to(self.device)
        self.conv2 = nn.Conv3d(self.nf * 4, self.nf * 4, kernel_size=3, padding=1, stride=2).to(self.device)
        self.bn2 = nn.BatchNorm3d(self.nf * 4).to(self.device)
        self.conv3 = nn.Conv3d(self.nf * 4, self.nf * 4, kernel_size=3, padding=1, stride=2).to(self.device)
        self.bn3 = nn.BatchNorm3d(self.nf * 4).to(self.device)
        
        final_size = 4 * 4 * 4* self.nf * 4 #*27

        # Regressor for individual parameters
        self.translation = nn.Linear(final_size, 3).to(self.device)
        self.rotation = nn.Linear(final_size, 3).to(self.device)
        self.scaling = nn.Linear(final_size, 3).to(self.device)
        self.shearing = nn.Linear(final_size, 3).to(self.device)

        # initialize the weights/bias with identity transformation
        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.shearing.weight.data.zero_()
        self.shearing.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))

    def get_theta(self, i):
        return self.theta[i]

    def forward(self, x):
        xs = F.relu(self.bn00(self.conv00(x)))
        xs = F.relu(self.bn0(self.conv0(xs)))
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        xs = xs.view(xs.size(0), -1)

        self.affine_matrix(xs)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def gen_3d_mesh_grid(self, d, h, w):
        d_s = torch.linspace(-1, 1, d)
        h_s = torch.linspace(-1, 1, h)
        w_s = torch.linspace(-1, 1, w)

        d_s, h_s, w_s = torch.meshgrid([d_s, h_s, w_s])
        one_s = torch.ones_like(w_s)

        mesh_grid = torch.stack([w_s, h_s, d_s, one_s])
        return mesh_grid  # 4 x d x h x w

    def affine_grid(self, theta, size):
        b, c, d, h, w = size
        mesh_grid = self.gen_3d_mesh_grid(d, h, w)
        mesh_grid = mesh_grid.unsqueeze(0)

        mesh_grid = mesh_grid.repeat(b, 1, 1, 1, 1)  # channel dim = 4
        mesh_grid = mesh_grid.view(b, 4, -1)
        mesh_grid = torch.bmm(theta, mesh_grid)  # channel dim = 3
        mesh_grid = mesh_grid.permute(0, 2, 1)  # move channel to last dim
        return mesh_grid.view(b, d, h, w, 3)

    def warp_image(self, img):
        grid = self.affine_grid(self.theta[:, 0:3, :], img.size()).to(self.device)
        wrp = F.grid_sample(img, grid, align_corners=False)

        return wrp

    def warp_inv_image(self, img):
        grid = self.affine_grid(self.theta_inv[:, 0:3, :], img.size()).to(self.device)
        wrp = F.grid_sample(img, grid, align_corners=False)

        return wrp

    def affine_matrix(self, x):
        b = x.size(0)

        ### TRANSLATION ###
        trans = torch.tanh(self.translation(x)) * 0.1
        translation_matrix = torch.zeros([b, 4, 4], dtype=torch.float)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 2, 2] = 1.0
        translation_matrix[:, 0, 3] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 3] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 3] = trans[:, 2].view(-1)
        translation_matrix[:, 3, 3] = 1.0

        ### ROTATION ###
        rot = torch.tanh(self.rotation(x)) * (np.pi / 4.0)
        angle_1 = rot[:, 0].view(-1)
        rotation_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float)
        rotation_matrix_1[:, 0, 0] = torch.cos(angle_1)
        rotation_matrix_1[:, 0, 1] = -torch.sin(angle_1)
        rotation_matrix_1[:, 1, 0] = torch.sin(angle_1)
        rotation_matrix_1[:, 1, 1] = torch.cos(angle_1)
        rotation_matrix_1[:, 2, 2] = 1.0
        rotation_matrix_1[:, 3, 3] = 1.0
        
        angle_2 = rot[:, 1].view(-1)
        rotation_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float)
        rotation_matrix_2[:, 1, 1] = torch.cos(angle_2)
        rotation_matrix_2[:, 1, 2] = -torch.sin(angle_2)
        rotation_matrix_2[:, 2, 1] = torch.sin(angle_2)
        rotation_matrix_2[:, 2, 2] = torch.cos(angle_2)
        rotation_matrix_2[:, 0, 0] = 1.0
        rotation_matrix_2[:, 3, 3] = 1.0
        
        angle_3 = rot[:, 2].view(-1)
        rotation_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float)
        rotation_matrix_3[:, 0, 0] = torch.cos(angle_3)
        rotation_matrix_3[:, 0, 1] = -torch.sin(angle_3)
        rotation_matrix_3[:, 1, 0] = torch.sin(angle_3)
        rotation_matrix_3[:, 1, 1] = torch.cos(angle_3)
        rotation_matrix_3[:, 2, 2] = 1.0
        rotation_matrix_3[:, 3, 3] = 1.0

        rotation_matrix = torch.bmm(rotation_matrix_1, rotation_matrix_2)
        rotation_matrix = torch.bmm(rotation_matrix, rotation_matrix_3)

        ### SCALING ###
        scale = torch.tanh(self.scaling(x)) * 0.2
        scaling_matrix = torch.zeros([b, 4, 4], dtype=torch.float)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = torch.exp(scale[:, 2].view(-1))
        scaling_matrix[:, 3, 3] = 1.0

        ### SHEARING ###
        shear = torch.tanh(self.shearing(x)) * (np.pi / 4.0)

        shear_1 = shear[:, 0].view(-1)
        shearing_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float)
        shearing_matrix_1[:, 1, 1] = torch.cos(shear_1)
        shearing_matrix_1[:, 1, 2] = -torch.sin(shear_1)
        shearing_matrix_1[:, 2, 1] = torch.sin(shear_1)
        shearing_matrix_1[:, 2, 2] = torch.cos(shear_1)
        shearing_matrix_1[:, 0, 0] = 1.0
        shearing_matrix_1[:, 3, 3] = 1.0

        shear_2 = shear[:, 1].view(-1)
        shearing_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float)
        shearing_matrix_2[:, 0, 0] = torch.cos(shear_2)
        shearing_matrix_2[:, 0, 2] = torch.sin(shear_2)
        shearing_matrix_2[:, 2, 0] = -torch.sin(shear_2)
        shearing_matrix_2[:, 2, 2] = torch.cos(shear_2)
        shearing_matrix_2[:, 1, 1] = 1.0
        shearing_matrix_2[:, 3, 3] = 1.0

        shear_3 = shear[:, 2].view(-1)
        shearing_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float)
        shearing_matrix_3[:, 0, 0] = torch.cos(shear_3)
        shearing_matrix_3[:, 0, 1] = -torch.sin(shear_3)
        shearing_matrix_3[:, 1, 0] = torch.sin(shear_3)
        shearing_matrix_3[:, 1, 1] = torch.cos(shear_3)
        shearing_matrix_3[:, 2, 2] = 1.0
        shearing_matrix_3[:, 3, 3] = 1.0

        shearing_matrix = torch.bmm(shearing_matrix_1, shearing_matrix_2)
        shearing_matrix = torch.bmm(shearing_matrix, shearing_matrix_3)

        # Affine transform
        matrix = torch.bmm(shearing_matrix, scaling_matrix)
        matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        matrix = torch.bmm(matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        self.theta = matrix
        self.theta_inv = torch.inverse(matrix)
            
    

class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        nn.BatchNorm1d

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x


class FastSmoothSENorm(nn.Module):
    class SEWeights(nn.Module):
        def __init__(self, in_channels, reduction=2):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        def forward(self, x):
            b, c, d, h, w = x.size()
            out = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1, 1)  # output_shape: in_channels x (1, 1, 1)
            out = F.relu(self.conv1(out))
            out = self.conv2(out)
            return out

    def __init__(self, in_channels, reduction=2):
        super(FastSmoothSENorm, self).__init__()
        self.norm = nn.InstanceNorm3d(in_channels, affine=False)
        self.gamma = self.SEWeights(in_channels, reduction)
        self.beta = self.SEWeights(in_channels, reduction)

    def forward(self, x):
        gamma = torch.sigmoid(self.gamma(x))
        beta = torch.tanh(self.beta(x))
        x = self.norm(x)
        return gamma * x + beta


class FastSmoothSeNormConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super(FastSmoothSeNormConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = FastSmoothSENorm(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.norm(x)
        return x


class RESseNormConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super().__init__()
        self.conv1 = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, **kwargs)

        if in_channels != out_channels:
            self.res_conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1, padding=0)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = self.res_conv(x) if self.res_conv else x
        x = self.conv1(x)
        x += residual
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return x
