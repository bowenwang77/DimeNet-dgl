import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from sklearn import metrics
from torch.optim.lr_scheduler import MultiStepLR

from nnframe.cgcnn_dgl_torch.model_cgcnn import SubstrateConvBlock
from data_origi.load_graph_data_torch import GraphLoader

best_mae_error = 1e10


def main(args, graph_loader):
    global best_mae_error

    # load data
    graph_loader.tt_split(val_ratio=args['val_ratio'],
                          test_ratio=args['test_ratio'],
                          normalize=True)
    train_loader, val_loader, test_loader = graph_loader.batch(batch_sz=args['batch_size'],
                                                               batch_sz_t=args['batch_size'])

    # obtain target value normalizer
    train_target = []
    for i in range(len(train_loader)):
        train_target += train_loader[i][1][:, 1].tolist()
    if len(train_target) < 500:
        warnings.warn('Dataset has less than 500 data points. '
                      'Lower accuracy is expected. ')
        sample_target = train_target
    else:
        sample_target = sample(train_target, 500)
    normalizer = Normalizer(torch.FloatTensor(sample_target))

    # build model
    orig_atom_fea_len = train_loader[0][0][1].shape[-1]
    edge_dist_len = train_loader[0][0][2].shape[-1]
    model = SubstrateConvBlock(i_dim=orig_atom_fea_len,
                               o_dim=args['atom_fea_len'],
                               e_dim=edge_dist_len,
                               num_layer=args['n_conv'],
                               n_h=args['n_h'],
                               h_dim=args['h_fea_len'])
    model.float()
    summary(model)
    print(model)

    if args['cuda']:
        model.cuda()

    # define loss func and optimizer
    criterion = nn.MSELoss()
    if args['optim'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), args['lr'],
                              momentum=args['momentum'],
                              weight_decay=args['weight_decay'])
    elif args['optim'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), args['lr'],
                               weight_decay=args['weight_decay'])
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args['resume']:
        if os.path.isfile(args['resume']):
            print("=> loading checkpoint '{}'".format(args['resume']))
            checkpoint = torch.load(args['resume'])
            args['start_epoch'] = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args['resume']))

    scheduler = MultiStepLR(optimizer, milestones=args['lr_milestones'],
                            gamma=0.1)

    for epoch in range(args['start_epoch'], args['epochs']):
        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        mae_error = validate(args, val_loader, model, criterion, normalizer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if args['task'] == 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': args
        }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(args, test_loader, model, criterion, normalizer, test=True)


def train(args, train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode

    model.train()

    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args['cuda']:
            x = (x[0].to('cuda:0'),
                 x[1].to('cuda:0'),
                 x[2].to('cuda:0'))
        # else:
        #     x = (x[0],
        #          Variable(x[1]),
        #          Variable(x[2]))

        target, sample_ids = torch.unsqueeze(y[:, 1], dim=1), torch.unsqueeze(y[:, 0], dim=1)
        # normalize target
        target_normed = normalizer.norm(target)
        if args['cuda']:
            target_normed = target_normed.to('cuda:0')
        # else:
        #     target_normed = Variable(target_normed)

        # compute output
        output = model(x)
        loss = criterion(output, target_normed)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.detach().cpu()), target)
        losses.update(loss.detach().cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mae_errors=mae_errors)
            )
            with open(file=f'cgcnn_train_log.txt', mode='a+') as log:
                log.writelines('Epoch: [{0}][{1}/{2}]\t'
                               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                               'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                               'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\n'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )
            for name, parms in model.named_parameters():
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                      ' -->grad_value:', parms.grad)


def validate(args, val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (x, y) in enumerate(val_loader):

        target, batch_cif_ids = torch.unsqueeze(y[:, 1], dim=1), torch.unsqueeze(y[:, 0], dim=1)
        if args['cuda']:
            x = (x[0].to('cuda:0'),
                 x[1].to('cuda:0'),
                 x[2].to('cuda:0'))
        # else:
        #     with torch.no_grad():
        #         x = (x[0],
        #              Variable(x[1]),
        #              Variable(x[2]))

        target_normed = normalizer.norm(target)
        if args['cuda']:
            # with torch.no_grad():
            target_normed = target_normed.to('cuda:0')
        # else:
        #     with torch.no_grad():
        #         target_normed = Variable(target_normed)
        with torch.no_grad():
            # compute output
            output = model(x)
            loss = criterion(output, target_normed)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.detach().cpu()), target)
        losses.update(loss.detach().cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids.view(-1).tolist()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                mae_errors=mae_errors))
            with open(file=f'cgcnn_test_log.txt', mode='a+') as log:
                log.writelines(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\n'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        mae_errors=mae_errors)
                )

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                    mae_errors=mae_errors))
    with open(file=f'cgcnn_test_log.txt', mode='a+') as log:
        log.writelines(' {star} MAE {mae_errors.avg:.3f}\n'.format(star=star_label,
                                                                   mae_errors=mae_errors))
    return mae_errors.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(args, optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args['lr'] * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    bool_expn = False
    gl = GraphLoader(graph_type='direct',
                     bool_expn=False,
                     prime_path=r'..\..\data_origi\primitive',
                     graph_path=r'..\..\data_origi',
                     one_hot=True,
                     load_ratio=1.0,
                     cgcnn_feat=True)
    args_dict = {'task': 'regression',
                 'cuda': False,
                 'workers': 0,
                 'epochs': 500,
                 'start_epoch': 0,
                 'batch_size': 16,
                 'lr': 0.001,
                 'lr_milestones': [100],
                 'momentum': 0.9,
                 'weight_decay': 0,
                 'print_freq': 10,
                 'resume': f'cgcnn_struct_expn{bool_expn}_dgl',

                 'train_ratio': 0.6,
                 'train_size': None,
                 'val_ratio': 0.2,
                 'val_size': None,
                 'test_ratio': 0.2,
                 'test_size': None,

                 'optim': 'Adam',
                 'atom_fea_len': 128,
                 'h_fea_len': 256,
                 'n_conv': 3,
                 'n_h': 1}
    main(args_dict, gl)
