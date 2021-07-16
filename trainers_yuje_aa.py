import torch
from torch._C import has_cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable

import numpy as np
import utils
import cv2
import random

import os

class Trainer(nn.Module):
    def __init__(self, network, dataloaders, optimizer, use_cuda=False):
        super(Trainer, self).__init__()
        self.network = network
        self.loader_train, self.loader_test = dataloaders
        self.optimizer = optimizer

        if use_cuda:
            self.nGPUs = torch.cuda.device_count()
            print('==> Transporting model to {} cuda device(s)..'.format(self.nGPUs))
            if self.nGPUs > 1:
                self.network = nn.DataParallel(self.network, device_ids=range(self.nGPUs))
            self.network.cuda()
            self.cuda = lambda x: x.cuda()
            cudnn.benchmark = True
        else:
            self.cuda = lambda x: x
            print('==> Keeping all on CPU..')

    def epoch(self, train=False, lr=0.1):
        if train:
            self.network.train()
            loader = self.loader_train
            forward = self.forward_train
        else:
            self.network.eval()
            loader = self.loader_test
            forward = self.forward_test

        loss_total = 0
        sample_error = 0
        sample_error5 = 0
        sample_total = 0
        progress = utils.ProgressBar(len(loader), '<progress bar is initialized.>')

        for batch_idx, (inputs, targets) in enumerate(loader):
            batchsize = targets.size(0)
            outputs, loss_batch = forward(inputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted5 = torch.topk(outputs.data, 5)
            sample_total += batchsize
            sample_error += batchsize - predicted.cpu().eq(targets).sum().item()
            loss_total += loss_batch.data.item() * batchsize
            loss = float(loss_total / sample_total)
            err = float(1. * sample_error / sample_total)
            result = predicted5[:, 0].cpu().eq(targets)
            for i in range(4):
                result += predicted5[:, i + 1].cpu().eq(targets)
            result = result.sum().item()
            sample_error5 += batchsize - result
            err5 = float(1. * sample_error5 / sample_total)

            progress.update(batch_idx,
                '{}, top1 loss: {:0.4f}, err:{:5.2f}% ({:5d}/{:5d}), top5 err:{:5.2f}% ({:5d}/{:5d}), lr:{}'.format(
                    'train' if train else ' test', loss, 100 * err,
                    int(sample_error), int(sample_total), 100 * err5,
                    int(sample_error5), int(sample_total), lr))

        return [err, loss]

    def forward_train(self, inputs, targets):
        self.optimizer.zero_grad()
        inputs = Variable(self.cuda(inputs))
        targets = Variable(self.cuda(targets))
        outputs = self.network(inputs)
        loss_batch = F.cross_entropy(outputs, targets)
        loss_batch.backward()
        self.optimizer.step()


        return outputs, loss_batch

    def forward_test(self, inputs, targets):
        with torch.no_grad():
            inputs = Variable(self.cuda(inputs))
        with torch.no_grad():
            targets = Variable(self.cuda(targets))
        with torch.no_grad():
            outputs = self.network(inputs)
        loss_batch = F.cross_entropy(outputs, targets)
        return outputs, loss_batch

class TrainerRICAP(Trainer):

    def __init__(self, network, dataloaders, optimizer, beta_of_ricap,use_cuda=False,saliency_type='mid'):
        super(TrainerRICAP,self).__init__(
            network,dataloaders,optimizer,use_cuda)
        self.beta = beta_of_ricap
        self.saliency_type = saliency_type
        self.saliency_bbox = None
        # if saliency_type == 'max':
        #     self.saliency_bbox = self.saliency_bbox_max
        # elif saliency_type == 'mid':
        #     self.saliency_bbox = self.saliency_bbox_mid
        # elif saliency_type == 'rand':
        #     self.saliency_bbox = self.saliency_bbox_rand

    def ricap(self, images, targets):

        beta = self.beta

        I_x, I_y=images.size()[2:]
        w = int(np.round(I_x * np.random.beta(beta, beta)))
        h = int(np.round(I_y * np.random.beta(beta, beta)))

        # 각 이미지의 크기
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]
                

        cropped_images = {}
        c_ = {0:[],1:[],2:[],3:[]}
        W_ = {}
        for k in range(4):
            # W_[k]=torch.tensor(((w_[k]*h_[k])/(I_x*I_y))).cuda()
            W_[k] = ((w_[k]*h_[k])/(I_x*I_y))

        patched_images_box = []
        box = []
        for p in range(4):
            box.append(self.cuda(torch.randperm(images.size(0))))
        index= [0,0,0,0]
        for i in range(images.size(0)):
            cropped_image = {}
            
            for ik in range(4):
                index[ik] = box[ik][i]

            for k in range(4):
                r = random.random()
                if r>0.5:
                    bbx1,bby1,bbx2,bby2 = self.saliency_bbox_rand(images[index[k]],w_[k],h_[k])
                # print(bbx1,bby1,bbx2,bby2)
                    cropped_image[k] = images[index[k]][:,bbx1:bbx2,bby1:bby2]
                else:
                    x_k = np.random.randint(0, I_x - w_[k] + 1)
                    y_k = np.random.randint(0, I_y - h_[k] + 1)
                    cropped_image[k] = images[index[k]][:, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                c_[k].append(targets[index[k]])
                
            patched_image = torch.cat(
                (torch.cat((cropped_image[0],cropped_image[1]),1),
                torch.cat((cropped_image[2],cropped_image[3]),1)),
                2)
            patched_images_box.append(patched_image)
        patched_images = torch.stack(patched_images_box,dim=0).cuda()
        # print(patched_images.shape)
        #c_처리
        # for k in range(4):
        #     c_[k] = torch.stack(c_[k]).cuda()
        #     W_[k] = torch.stack(W_[k]).cuda()
        for k in range(4):
            for j in range(len(c_[k])):
                c_[k][j] = c_[k][j].cpu().numpy()
            c_[k]=np.array(c_[k])
        targets = (c_,W_)
        # print(targets)
        return patched_images, targets

    def ricap_criterion(self, outputs, c_, W_):
        # print(outputs.shape)
        # print(len(c_[0]))
        # print(outputs[0].shape,c_[0][0].shape)
        # loss = 0
        # for i in range(outputs.size(0)):
        # for k in range(4):
        #     calc = torch.cuda.LongTensor(W_[k] *c_[k])
        loss = sum([W_[k] * F.cross_entropy(outputs, Variable(torch.cuda.LongTensor(c_[k]))) for k in range(4)])
        # print(loss)
        return loss
    
    def forward_train(self, inputs, targets):
        self.optimizer.zero_grad()
        inputs, targets = self.cuda(inputs), self.cuda(targets)
        inputs, (c_, W_) = self.ricap(inputs, targets)
        # for i in range(20):
        #     if not os.path.isfile('/root/volume/SICAP/checkpoint/testImage'+str(i)+'.jpg'):
        #         torchvision.utils.save_image(inputs[i],'/root/volume/SICAP/checkpoint/testImage'+str(i)+'.jpg')
        # print(inputs.shape)
        inputs = Variable(inputs)
        outputs = self.network(inputs)
        loss_batch = self.ricap_criterion(outputs, c_, W_)
        loss_batch.backward()
        self.optimizer.step()
        return outputs, loss_batch

    def find_nearest(self,array, value,W,w_,H,h_):
        array = array[:W-w_+1,:H-h_+1]
        idx = np.unravel_index((np.abs(array- value)).argmin(),array.shape)
        return idx
                
    # def find_nearest(self,array,value):
    #     idx = np.unravel_index((np.abs(array-value)).argmin(),array.shape)
    #     return idx                


    def saliency_bbox_rand(self,img,w_,h_):
        size = img.size()
        W = size[1]
        H = size[2]

        # initialize OpenCV's static fine grained saliency detector and
        # compute the saliency map
        temp_img = img.cpu().numpy().transpose(1, 2, 0)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(temp_img)
        saliencyMap = saliencyMap[:W-w_+1,:H-h_+1]
        saliencyMap = (saliencyMap * 255).astype("uint8")

        maxV = np.max(saliencyMap,axis=None)
        midV = np.median(saliencyMap, axis=None)

        rV = random.randrange(int(midV),int(maxV)+1)
        
        idx = self.find_nearest(saliencyMap,rV,W,w_,H,h_)

        random_indices = idx

        x = random_indices[0]
        y = random_indices[1]

        bbx1 = x 
        bby1 = y 
        bbx2 = x + w_
        bby2 = y + h_

        return bbx1, bby1, bbx2, bby2

def make_trainer(network, dataloaders, optimizer, use_cuda, beta_of_ricap=0.0,saliency_type= 'mid'):
    if beta_of_ricap:
        return TrainerRICAP(network, dataloaders, optimizer, beta_of_ricap, use_cuda,saliency_type)
    else:
        return Trainer(network, dataloaders, optimizer, use_cuda)