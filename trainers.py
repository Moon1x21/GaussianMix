import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import numpy as np
import utils
import cv2

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

    def __init__(self, network, dataloaders, optimizer, beta_of_ricap, use_cuda=False,saliency_type='mid'):
        super(TrainerRICAP, self).__init__(
            network, dataloaders, optimizer, use_cuda)
        self.beta = beta_of_ricap
        self.saliency_type = saliency_type

        self.saliency_bbox = None
        if saliency_type == 'max':
            self.saliency_bbox = self.saliency_bbox_max
        elif saliency_type == 'mid':
            self.saliency_bbox = self.saliency_bbox_mid
        elif saliency_type == 'rand':
            self.saliency_bbox = self.saliency_bbox_rand
    def ricap(self, images, targets):

        beta = self.beta  # hyperparameter

        # size of image
        I_x, I_y = images.size()[2:]

        # generate boundary position (w, h)
        w = int(np.round(I_x * np.random.beta(beta, beta)))
        h = int(np.round(I_y * np.random.beta(beta, beta)))
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]

        # select four images
        cropped_images = {}
        c_ = {}
        W_ = {}
        for k in range(4):
            index = self.cuda(torch.randperm(images.size(0)))
            
            bbx1, bby1, bbx2, bby2 = self.saliency_bbox(images[index[0]],w_[k],h_[k])
            cropped_images[k] = images[index][:,:,bbx1:bbx2,bby1:bby2]
            c_[k] = targets[index]
            W_[k] = (w_[k] * h_[k]) / (I_x * I_y)

        # patch cropped images
        patched_images = torch.cat(
            (torch.cat((cropped_images[0], cropped_images[1]), 2),
             torch.cat((cropped_images[2], cropped_images[3]), 2)),
            3)
        targets = (c_, W_)
        return patched_images, targets

    def ricap_criterion(self, outputs, c_, W_):
        loss = sum([W_[k] * F.cross_entropy(outputs, Variable(c_[k])) for k in range(4)])
        print(W_)
        print(loss)
        return loss

    def forward_train(self, inputs, targets):
        self.optimizer.zero_grad()
        inputs, targets = self.cuda(inputs), self.cuda(targets)
        inputs, (c_, W_) = self.ricap(inputs, targets)
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

    def saliency_bbox_max(self,img,w_,h_):
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

        maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None),saliencyMap.shape)
        print(maximum_indices)
        x = maximum_indices[0]
        y = maximum_indices[1]

        bbx1 = x 
        bby1 = y 
        bbx2 = x + w_
        bby2 = y + h_

        return bbx1, bby1, bbx2, bby2

    def saliency_bbox_mid(self,img,w_,h_):
        size = img.size()
        W = size[1]
        H = size[2]

        # initialize OpenCV's static fine grained saliency detector and
        # compute the saliency map
        temp_img = img.cpu().numpy().transpose(1, 2, 0)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(temp_img)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        idx = self.find_nearest(saliencyMap,np.median(saliencyMap, axis=None),W,w_,H,h_)

        median_indices = idx
        x = median_indices[0]
        y = median_indices[1]

        bbx1 = x 
        bby1 = y 
        bbx2 = x + w_
        bby2 = y + h_

        return bbx1, bby1, bbx2, bby2
    
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
