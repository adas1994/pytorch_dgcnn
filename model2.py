import os, sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x,k=20,idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    #device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature




class PointNet(nn.Module):
    def __init__(self, args, output_channels=5):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class JetVarsProcessor(nn.Module):
    def __init__(self,args, input_dim, output_dim):
        super(JetVarsProcessor, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(input_dim, input_dim, bias=False)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(input_dim, 2*input_dim, bias=False)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.bn2 = nn.BatchNorm1d(2*input_dim)
        self.dp2 = nn.Dropout()
        self.linear3 = nn.Linear(2*input_dim, 2*input_dim, bias=False)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.bn3 = nn.BatchNorm1d(2*input_dim)
        self.dp3 = nn.Dropout()
        self.linear4 = nn.Linear(2*input_dim, 32, bias=False)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.bn4 = nn.BatchNorm1d(32)
        self.dp4 = nn.Dropout()
        self.linear5 = nn.Linear(32, 12, bias=False)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2)
        self.bn5 = nn.BatchNorm1d(12)
        

    def forward(self, x):
        x = self.bn1(self.relu1(self.linear1(x)))
        #x = self.dp1(x)
        x = self.bn2(self.relu2(self.linear2(x)))
        x = self.dp2(x)
        x = self.bn3(self.relu3(self.linear3(x)))
        x = self.dp3(x)
        x = self.bn4(self.relu4(self.linear4(x)))
        #x = self.dp4(x)
        x = self.bn5(self.relu5(self.linear5(x)))
        return x



class FrameImageModel(nn.Module):
    def __init__(self, args, output_channels=12):
        super(FrameImageModel, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(48)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(192)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv_block1 = nn.Sequential(nn.Conv2d(6, 48, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(48*2, 48, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_block3 = nn.Sequential(nn.Conv2d(48*2, 96, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_block4 = nn.Sequential(nn.Conv2d(96*2, 192, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_block5 = nn.Sequential(nn.Conv1d(384, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 64, bias=False)
        self.bn6 = nn.BatchNorm1d(64)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(64, 96)
        self.bn7 = nn.BatchNorm1d(96)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(96, output_channels)
        self.feedforward_block = nn.Sequential(self.linear1,
                                               self.bn6,
                                               nn.LeakyReLU(negative_slope=0.2),
                                               self.dp1,
                                               self.linear2,
                                               self.bn7,
                                               nn.LeakyReLU(negative_slope=0.2),
                                               self.dp2,
                                               self.linear3)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv_block1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv_block2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv_block3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv_block4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv_block5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = self.feedforward_block(x)
        return x


class JetClassifier(nn.Module):
    def __init__(self, args, output_channels=12):
        super(JetClassifier, self).__init__()
        self.args = args
        self.output_channels = output_channels
        self.HiggsFrameImageModel = FrameImageModel(self.args, self.output_channels)
        self.WFrameImageModel     = FrameImageModel(self.args, self.output_channels)
        self.ZFrameImageModel     = FrameImageModel(self.args, self.output_channels)
        self.TopFrameImageModel   = FrameImageModel(self.args, self.output_channels)
        self.JetFrameImageModel   = FrameImageModel(self.args, self.output_channels)
        self.LabFrameImageModel   = FrameImageModel(self.args, self.output_channels)
        
        self.linear1 = nn.Linear(6*self.output_channels, 144, bias=False)
        self.relu1   = nn.LeakyReLU(negative_slope=0.2)
        self.bn1     = nn.BatchNorm1d(144)
        self.dp1     = nn.Dropout(p=args.dropout)
        
        self.linear2 = nn.Linear(144, 64)
        self.relu2   = nn.LeakyReLU(negative_slope=0.2)
        self.bn2     = nn.BatchNorm1d(64)
        self.dp2     = nn.Dropout(p=args.dropout)
        
        self.linear3 = nn.Linear(64, 16)
        self.relu3   = nn.LeakyReLU(negative_slope=0.2)
        self.bn3     = nn.BatchNorm1d(16)

        self.linear4 = nn.Linear(16, 5)

    def forward(self, x):
        tensor_input_HiggsFrame = x[:,:,0:50]
        tensor_input_TopFrame   = x[:,:,50:100]
        tensor_input_WFrame     = x[:,:,100:150]
        tensor_input_ZFrame     = x[:,:,150:200]
        tensor_input_jetFrame   = x[:,:,200:250]
        tensor_input_labFrame   = x[:,:,250:300]

        tensor_output_HiggsFrameModel = self.HiggsFrameImageModel(tensor_input_HiggsFrame)
        tensor_output_topFrameModel   = self.TopFrameImageModel(tensor_input_TopFrame)
        tensor_output_WFrameModel     = self.WFrameImageModel(tensor_input_WFrame)
        tensor_output_ZFrameModel     = self.ZFrameImageModel(tensor_input_ZFrame)
        tensor_output_jetFrameModel   = self.JetFrameImageModel(tensor_input_jetFrame)
        tensor_output_labFrameModel   = self.LabFrameImageModel(tensor_input_labFrame)


        combined = torch.cat((tensor_output_HiggsFrameModel, tensor_output_topFrameModel, tensor_output_WFrameModel, 
                       tensor_output_ZFrameModel, tensor_output_jetFrameModel, tensor_output_labFrameModel), dim=1)
        
        combined1 = self.bn1(self.relu1(self.linear1(combined)))
        combined1 = self.dp1(combined1)
        
        combined1 = self.bn2(self.relu2(self.linear2(combined1)))
        combined1 = self.dp2(combined1)
        combined1 = self.bn3(self.relu3(self.linear3(combined1)))
        
        prediction = F.softmax(self.linear4(combined1))
        return prediction


