#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

## File is altered for binary segmentation, (Johan Bergius) ##
import datetime
import os
import time
import imp
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable
from common.avgmeter import *
from common.logger import Logger
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import *
from tasks.semantic.modules.ioueval import *
from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.modules.Lovasz_Softmax import Lovasz_softmax
import tasks.semantic.modules.adf as adf


''' INIT FUNC '''
# Reference
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)


    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net



def keep_variance_fn(x):
    return x + 1e-3

def one_hot_pred_from_label(y_pred, labels):
    y_true = torch.zeros_like(y_pred)
    ones = torch.ones_like(y_pred)
    indexes = [l for l in labels]
    y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(labels.size(0)), indexes]

    return y_true


class SoftmaxHeteroscedasticLoss(torch.nn.Module):
    def __init__(self):
        super(SoftmaxHeteroscedasticLoss, self).__init__()
        self.adf_softmax = adf.Softmax(dim=1, keep_variance_fn=keep_variance_fn)

    def forward(self, outputs, targets, eps=1e-5):
        mean, var = self.adf_softmax(*outputs)
        targets = torch.nn.functional.one_hot(targets, num_classes=2).permute(0,3,1,2).float()

        precision = 1 / (var + eps)
        return torch.mean(0.5 * precision * (targets - mean) ** 2 + 0.5 * torch.log(var + eps))


def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return


def save_checkpoint(to_save, logdir, suffix=""):
    # Save the weights
    torch.save(to_save, logdir +
               "/SalsaNext" + suffix)


class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None,uncertainty=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path
        self.uncertainty = uncertainty

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()
        self.epoch = 0

        # put logger where it belongs

        self.info = {"train_update": 0,
                     "train_loss": 0,
                     "train_acc": 0,
                     "train_iou": 0,
                     "valid_loss": 0,
                     "valid_acc": 0,
                     "valid_iou": 0,
                     "best_train_iou": 0,
                     "best_val_iou": 0}

        # get the data
        parserModule = imp.load_source("parserModule",
                                       booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                       self.DATA["name"] + '/parser.py')
        self.parser = parserModule.Parser(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"],
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=None,
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=self.ARCH["train"]["batch_size"],
                                          workers=self.ARCH["train"]["workers"],
                                          gt=True,
                                          shuffle_train=True)

        # weights for loss (and bias)

        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights
        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        #print("Loss weights from content: ", self.loss_w.data)

        with torch.no_grad():
            if not self.uncertainty:
                self.model = SalsaNext(self.parser.get_n_classes())
                init_net(self.model, init_type='kaiming', init_gain=0.02, gpu_ids=[]) #added
            else:
                self.model = SalsaNextUncertainty(self.parser.get_n_classes())

        self.tb_logger = Logger(self.log + "/tb")

        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)  # spread in gpus
            self.model = convert_model(self.model).cuda()  # sync batchnorm
            self.model_single = self.model.module  # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()



        self.criterion = nn.BCELoss(reduce=False).to(self.device)
        
        self.ls = Lovasz_softmax(ignore=0).to(self.device)
        self.SoftmaxHeteroscedasticLoss = SoftmaxHeteroscedasticLoss().to(self.device)
        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
            self.ls = nn.DataParallel(self.ls).cuda()
            self.SoftmaxHeteroscedasticLoss = nn.DataParallel(self.SoftmaxHeteroscedasticLoss).cuda()

        self.optimizer = optim.SGD([{'params': self.model.parameters()}],
                                   lr=self.ARCH["train"]["lr"],
                                   momentum=self.ARCH["train"]["momentum"],
                                   weight_decay=self.ARCH["train"]["w_decay"])


        # Use warmup learning rate
        # post decay and step sizes come in epochs and we want it in steps
        steps_per_epoch = self.parser.get_train_size()
        up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
        final_decay = self.ARCH["train"]["lr_decay"] ** (1 / steps_per_epoch)
        self.scheduler = warmupLR(optimizer=self.optimizer,
                                  lr=self.ARCH["train"]["lr"],
                                  warmup_steps=up_steps,
                                  momentum=self.ARCH["train"]["momentum"],
                                  decay=final_decay)

        if self.path is not None:
            torch.nn.Module.dump_patches = True
            w_dict = torch.load(path + "/SalsaNext",
                                map_location=lambda storage, loc: storage)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
            self.optimizer.load_state_dict(w_dict['optimizer'])
            self.epoch = w_dict['epoch'] + 1
            self.scheduler.load_state_dict(w_dict['scheduler'])
            print("dict epoch:", w_dict['epoch'])
            self.info = w_dict['info']
            print("info", w_dict['info'])


    def calculate_estimate(self, epoch, iter):
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) * \
                       (self.parser.get_train_size() * self.ARCH['train']['max_epochs'] - (
                               iter + 1 + epoch * self.parser.get_train_size()))) + \
                   int(self.batch_time_e.avg * self.parser.get_valid_size() * (
                           self.ARCH['train']['max_epochs'] - (epoch)))
        return str(datetime.timedelta(seconds=estimate))

    @staticmethod
    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    @staticmethod
    def make_log_img(depth, mask, pred, gt, color_fn):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
        # make label prediction
        pred_color = color_fn((pred * mask).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)

    @staticmethod
    def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
        # save scalars
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)

    def train(self):

        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)

        # train for n epochs
        for epoch in range(self.epoch, self.ARCH["train"]["max_epochs"]):

            # train for 1 epoch
            acc, iou, loss, update_mean,hetero_l = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                           model=self.model,
                                                           criterion=self.criterion,
                                                           optimizer=self.optimizer,
                                                           epoch=epoch,
                                                           evaluator=self.evaluator,
                                                           scheduler=self.scheduler,
                                                           color_fn=self.parser.to_color,
                                                           report=self.ARCH["train"]["report_batch"],
                                                           show_scans=self.ARCH["train"]["show_scans"])

            # update info
            self.info["train_update"] = update_mean
            self.info["train_loss"] = loss
            self.info["train_acc"] = acc
            self.info["train_iou"] = iou
            self.info["train_hetero"] = hetero_l

            # remember best iou and save checkpoint
            state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict()
                     }
            save_checkpoint(state, self.log, suffix="")

            if self.info['train_iou'] > self.info['best_train_iou']:
                print("Best mean iou in training set so far, save model!")
                self.info['best_train_iou'] = self.info['train_iou']
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_train_best")

            if epoch % self.ARCH["train"]["report_epoch"] == 0:
                # evaluate on validation set
                print("*" * 80)
                acc, iou, loss, rand_img,hetero_l = self.validate(val_loader=self.parser.get_valid_set(),
                                                         model=self.model,
                                                         criterion=self.criterion,
                                                         evaluator=self.evaluator,
                                                         class_func=self.parser.get_xentropy_class_string,
                                                         color_fn=self.parser.to_color,
                                                         save_scans=self.ARCH["train"]["save_scans"])

                # update info
                self.info["valid_loss"] = loss
                self.info["valid_acc"] = acc
                self.info["valid_iou"] = iou
                self.info['valid_heteros'] = hetero_l



            print("*" * 80)

            # save to log
            Trainer.save_to_log(logdir=self.log,
                                logger=self.tb_logger,
                                info=self.info,
                                epoch=epoch,
                                w_summary=self.ARCH["train"]["save_summary"],
                                model=self.model_single,
                                img_summary=self.ARCH["train"]["save_scans"],
                                imgs=rand_img)

        print('Finished Training')

        return

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, scheduler, color_fn, report=10,
                    show_scans=False):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        update_ratio_meter = AverageMeter()
        
        weight = torch.tensor([0.72, 0.28])

        
        # empty the cache to train now
        if self.gpu:
            torch.cuda.empty_cache()

        # switch to train mode
        model.train()
        end = time.time()
        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(train_loader):
            # measure data loading time
            self.data_time_t.update(time.time() - end)
            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
                #proj_mask = proj_mask.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()

            # compute output
            if self.uncertainty:
                #unused
                output = model(in_vol)
                '''
                output_mean, output_var = adf.Softmax(dim=1, keep_variance_fn=keep_variance_fn)(*output)
                hetero = self.SoftmaxHeteroscedasticLoss(output,proj_labels) 
                loss_m = criterion(output_mean.clamp(min=1e-8), proj_labels.long()) + hetero + self.ls(output_mean, proj_labels.long())
                
                y = proj_labels.float()
                weight_ = weight[y.data.view(-1).long()].view_as(y)
                loss_m = criterion(output[:,0,:,:], y)
                loss_m = loss_m * weight_.cuda()
                loss_m = loss_m.mean()
                
                hetero_l.update(hetero.mean().item(), in_vol.size(0))
                output = output_mean
                '''
            else:
                output = model(in_vol)
                                 
                #output = torch.add(output, 1)#TANH instead of sigmoid
                #output = torch.div(output, 2)#TANH instead of sigmoid    
                y = proj_labels.float()
                weight_ = weight[y.data.view(-1).long()].view_as(y)
                loss_m = criterion(output[:,0,:,:], y)
                loss_m = loss_m * weight_.cuda()
                loss_m = loss_m.mean()
                
            optimizer.zero_grad()
            if self.n_gpus > 1:
                idx = torch.ones(self.n_gpus).cuda()
                loss_m.backward(idx)
            else:
                loss_m.backward()
            optimizer.step()
             
            
            # measure accuracy and record loss
            loss = loss_m.mean()
            with torch.no_grad():
                evaluator.reset()

                argmax = torch.round(output) 
                evaluator.addBatch(argmax.long(), proj_labels.long())
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()

            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            # measure elapsed time
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            # get gradient updates and weights, so I can print the relationship of
            # their norms
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10) *
                                                value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))
            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            if show_scans:
                # get the first scan in batch and project points
                mask_np = proj_mask[0].cpu().numpy()
                depth_np = in_vol[0][0].cpu().numpy()
                pred_np = argmax[0].cpu().numpy()
                gt_np = proj_labels[0].cpu().numpy()
                out = Trainer.make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)

                mask_np = proj_mask[1].cpu().numpy()
                depth_np = in_vol[1][0].cpu().numpy()
                pred_np = argmax[1].cpu().numpy()
                gt_np = proj_labels[1].cpu().numpy()
                out2 = Trainer.make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)

                out = np.concatenate([out, out2], axis=0)
                cv2.imshow("sample_training", out)
                cv2.waitKey(1)
            if self.uncertainty:

                if i % self.ARCH["train"]["report_batch"] == 0:
                    print( 'Lr: {lr:.3e} | '
                          'Update: {umean:.3e} mean,{ustd:.3e} std | '
                          'Epoch: [{0}][{1}/{2}] | '
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                          'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                          'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                          'Hetero {hetero_l.val:.4f} ({hetero_l.avg:.4f}) | '
                          'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                          'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                        epoch, i, len(train_loader), batch_time=self.batch_time_t,
                        data_time=self.data_time_t, loss=losses, hetero_l=hetero_l,acc=acc, iou=iou, lr=lr,
                        umean=update_mean, ustd=update_std, estim=self.calculate_estimate(epoch, i)))

                    save_to_log(self.log, 'log.txt', 'Lr: {lr:.3e} | '
                          'Update: {umean:.3e} mean,{ustd:.3e} std | '
                          'Epoch: [{0}][{1}/{2}] | '
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                          'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                          'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                          'Hetero {hetero.val:.4f} ({hetero.avg:.4f}) | '
                          'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                          'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                        epoch, i, len(train_loader), batch_time=self.batch_time_t,
                        data_time=self.data_time_t, loss=losses, hetero=hetero_l,acc=acc, iou=iou, lr=lr,
                        umean=update_mean, ustd=update_std, estim=self.calculate_estimate(epoch, i)))
            else:
                if i % self.ARCH["train"]["report_batch"] == 0:
                    print('Lr: {lr:.3e} | '
                          'Update: {umean:.3e} mean,{ustd:.3e} std | '
                          'Epoch: [{0}][{1}/{2}] | '
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                          'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                          'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                          'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                          'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                        epoch, i, len(train_loader), batch_time=self.batch_time_t,
                        data_time=self.data_time_t, loss=losses, acc=acc, iou=iou, lr=lr,
                        umean=update_mean, ustd=update_std, estim=self.calculate_estimate(epoch, i)))

                    save_to_log(self.log, 'log.txt', 'Lr: {lr:.3e} | '
                                                     'Update: {umean:.3e} mean,{ustd:.3e} std | '
                                                     'Epoch: [{0}][{1}/{2}] | '
                                                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                                                     'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                                                     'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                                                     'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                                                     'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                        epoch, i, len(train_loader), batch_time=self.batch_time_t,
                        data_time=self.data_time_t, loss=losses, acc=acc, iou=iou, lr=lr,
                        umean=update_mean, ustd=update_std, estim=self.calculate_estimate(epoch, i)))

            # step scheduler
            scheduler.step()

        return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg,hetero_l.avg

    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        rand_imgs = []
        
        

        
        # switch to evaluate mode
        model.eval()
        evaluator.reset()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(val_loader):
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                # compute output
                if self.uncertainty:
                    log_var, output, _ = model(in_vol)
                    '''
                    log_out = torch.log(output.clamp(min=1e-8))
                    mean = output.argmax(dim=1)
                    log_var = log_var.mean(dim=1)
                    hetero = self.SoftmaxHeteroscedasticLoss(mean.float(),proj_labels.float()).mean()
                    jacc = self.ls(output, proj_labels)                       
                    wce = criterionJB(output[:,0,:,:], proj_labels.float()) #JB added 
                    loss = wce + jacc
                    hetero_l.update(hetero.mean().item(), in_vol.size(0))
                    '''
                else:
                    output = model(in_vol)
                    
                    #output = torch.add(output, 1) #TANH instead of sigmoid
                    #output = torch.div(output, 2) #TANH instead of sigmoid           
                    log_out = torch.log(output.clamp(min=1e-8))

                    
                    y = proj_labels.float()
                    weight = torch.tensor([0.72, 0.28])
                    weight_ = weight[y.data.view(-1).long()].view_as(y)
                    loss = criterion(output[:,0,:,:], y)
                    loss = loss * weight_.cuda()
                    loss = loss.mean()
 


                # measure accuracy and record loss
                argmax = torch.round(output[:,0,:,:])
                evaluator.addBatch(argmax.long(), proj_labels.long())
                losses.update(loss.mean().item(), in_vol.size(0))


                wces.update(loss.mean().item(),in_vol.size(0))



                if save_scans:
                    # get the first scan in batch and project points
                    mask_np = proj_mask[0].cpu().numpy()
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = Trainer.make_log_img(depth_np,
                                               mask_np,
                                               pred_np,
                                               gt_np,
                                               color_fn)
                    rand_imgs.append(out)

                # measure elapsed time
                self.batch_time_e.update(time.time() - end)
                end = time.time()

            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            if self.uncertainty:
                print('Validation set:\n'       
                      'Time avg per batch {batch_time.avg:.3f}\n'
                      'Loss avg {loss.avg:.4f}\n'
                      'Jaccard avg {jac.avg:.4f}\n'
                      'WCE avg {wces.avg:.4f}\n'
                      'Hetero avg {hetero.avg}:.4f\n'
                      'Acc avg {acc.avg:.3f}\n'
                      'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                     loss=losses,
                                                     jac=jaccs,
                                                     wces=wces,
                                                     hetero=hetero_l,
                                                     acc=acc, iou=iou))

                save_to_log(self.log, 'log.txt', 'Validation set:\n'
                      'Time avg per batch {batch_time.avg:.3f}\n'
                      'Loss avg {loss.avg:.4f}\n'
                      'Jaccard avg {jac.avg:.4f}\n'
                      'WCE avg {wces.avg:.4f}\n'
                      'Hetero avg {hetero.avg}:.4f\n'
                      'Acc avg {acc.avg:.3f}\n'
                      'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                     loss=losses,
                                                     jac=jaccs,
                                                     wces=wces,
                                                     hetero=hetero_l,
                                                     acc=acc, iou=iou))
                # print also classwise
                for i, jacc in enumerate(class_jaccard):
                    print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                        i=i, class_str=class_func(i), jacc=jacc))
                    save_to_log(self.log, 'log.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                        i=i, class_str=class_func(i), jacc=jacc))
                    self.info["valid_classes/"+class_func(i)] = jacc
            else:

                print('Validation set:\n'
                      'Time avg per batch {batch_time.avg:.3f}\n'
                      'Loss avg {loss.avg:.4f}\n'
                      'Jaccard avg {jac.avg:.4f}\n'
                      'WCE avg {wces.avg:.4f}\n'
                      'Acc avg {acc.avg:.3f}\n'
                      'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                     loss=losses,
                                                     jac=jaccs,
                                                     wces=wces,
                                                     acc=acc, iou=iou))

                save_to_log(self.log, 'log.txt', 'Validation set:\n'
                                                 'Time avg per batch {batch_time.avg:.3f}\n'
                                                 'Loss avg {loss.avg:.4f}\n'
                                                 'Jaccard avg {jac.avg:.4f}\n'
                                                 'WCE avg {wces.avg:.4f}\n'
                                                 'Acc avg {acc.avg:.3f}\n'
                                                 'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                                                loss=losses,
                                                                                jac=jaccs,
                                                                                wces=wces,
                                                                                acc=acc, iou=iou))
                # print also classwise
                for i, jacc in enumerate(class_jaccard):
                    print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                        i=i, class_str=class_func(i), jacc=jacc))
                    save_to_log(self.log, 'log.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                        i=i, class_str=class_func(i), jacc=jacc))
                    self.info["valid_classes/" + class_func(i)] = jacc


        return acc.avg, iou.avg, losses.avg, rand_imgs, hetero_l.avg
