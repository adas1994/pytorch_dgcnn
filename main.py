from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import label_binarize
#from new_model import PointNet, FrameImageModel, JetClassifier
from model2 import PointNet, FrameImageModel, JetClassifier
import numpy as np
from torch.utils.data import DataLoader
from util import Jet_PointCloudDataSet, cal_loss, IOStream, PointCloudDataSet
from util import make_onehot
import sklearn.metrics as metrics

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model2.py checkpoints' + '/' + args.exp_name + '/' + 'model2.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    #os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args,io):
    file_list = ['preprocess1_HHESTIA_ttbar', 'preprocess2_HHESTIA_RadionToZZ', 'preprocess_HHESTIA_QCD1800To2400', 'preprocess0_HHESTIA_HH_4B', 'preprocess2_HHESTIA_ZprimeWW']
    train_grp_list  = ['train_group' for i in range(len(file_list))]
    test_grp_list = ['test_group' for i in range(len(file_list))]
    data_dir = '/afs/crc.nd.edu/user/a/adas/DGCNN'
    pcl_train_dataset = Jet_PointCloudDataSet(file_list,train_grp_list,data_dir)
    pcl_test_dataset = Jet_PointCloudDataSet(file_list,test_grp_list,data_dir)
    train_loader = DataLoader(pcl_train_dataset, num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)


    test_loader = DataLoader(pcl_test_dataset, num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = FrameImageModel(args).to(device)
    elif args.model == 'jetClassifier':
        model = JetClassifier(args).to(device)
    else:
        raise Exception("Not implemented")

    #print(str(model))
    model = model.double()
    model = nn.DataParallel(model)
    torch.save(model.state_dict(),'model.pt')
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss
    best_test_acc = 0
    record_file = open("note.txt","w")
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        print("########## training on epoch number ------>> ",epoch)
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        batch_number = 1
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1).double()
            batch_size = data.size()[0]
            opt.zero_grad()
            #print(data.shape)
            logits = model(data)
            loss = criterion(logits, label)
            batch_loss = loss.detach().cpu().numpy()
            print("### batch number ",batch_number," ",loss)
            
            binarized_label = label_binarize(label.cpu().numpy(),classes=[0,1,2,3,4])
            
            batch_number = batch_number + 1
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            batch_label = label.detach().cpu().numpy()
            batch_preds = preds.detach().cpu().numpy()
            batch_acc = metrics.accuracy_score(batch_label, batch_preds)
            balanced_batch_acc = metrics.balanced_accuracy_score(batch_label, batch_preds)
            print("### batch accuracy scores ",batch_acc, " ",balanced_batch_acc)
            record_file.write(str(epoch)+" "+str(batch_number)+" "+str(batch_loss)+" "+str(batch_acc)+" "+str(balanced_batch_acc))
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        accuracy_score = metrics.accuracy_score(train_true, train_pred)
        balanced_accuracy_score =  metrics.balanced_accuracy_score(train_true, train_pred)
        print(accuracy_score, balanced_accuracy_score)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch, train_loss*1.0/count, accuracy_score, balanced_accuracy_score)
        io.cprint(outstr)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            #print(data.shape)
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            batch_label = label.detach().cpu().numpy()
            batch_preds = preds.detach().cpu().numpy()
            batch_acc = metrics.accuracy_score(batch_label, batch_preds)
            balanced_batch_acc = metrics.balanced_accuracy_score(batch_label, batch_preds)
            print("### test batch accuracy scores ",batch_acc, " ",balanced_batch_acc)
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
    
    
    record_file.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',help='Name of the experiment')
    parser.add_argument('--model', type=str, default='jetClassifier', metavar='N',choices=['pointnet', 'dgcnn', 'jetClassifier'],help='Model to use, [pointnet, dgcnn, jetClassifier]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N', choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=400, metavar='batch_size',help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=400, metavar='batch_size', help='Size of batch')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.004, metavar='LR',help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.8, metavar='M',help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default= False, help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=50, help='number of points to use')
    parser.add_argument('--dropout', type=float, default=0.35, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=32, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')

    args = parser.parse_args()
    
    _init_()
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    #else:
    #    test(args, io)
