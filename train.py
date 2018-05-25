import os
import torch
import datetime

from darknet import Darknet19

import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
#import cfgs.config as cfg
from random import randint

data_root_path = './data'
save_root_path = './save'
pretrained_model = './pretrained_model.npz'

def main(args):
    data_name = args.data_name
    data_type = args.data_type
    year = args.year
    batch_size = args.batch_size
    num_process = args.num_process
    """
    gpu_id = args.gpu_id
    if gpu_id>=0:
        torch.cuda.set_device(gpu_id)
    """
    gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES'])
    lr = args.lr
    lr_decay_epochs = args.lr_decay_epochs
    lr_decay_rate = args.lr_decay_rate
    epoch = args.epoch
    disp_interval = args.disp_interval
    exp_name = args.exp_name
    use_tensorboard = args.use_tensorboard

    # Data Loader
    datadir = os.path.join(data_root_path, data_name)
    im_processor = yolo_utils.preprocess_train
    if data_name=='voc':
        # data loader
        from datasets.pascal_voc import VOCDataset
        from cfgs import config_voc as cfg
        imdb = VOCDataset(data_type=data_type, year=year, datadir=datadir,
                        batch_size=batch_size, im_processor=im_processor, cfg=cfg,
                        processes=num_process, shuffle=True,
                        dst_size=cfg.multi_scale_inp_size)
    elif data_name=='mscoco':
        from datasets.coco import COCODataset
        from cfgs import config_coco as cfg
        imdb = COCODataset(data_type=data_type, year=year, datadir=datadir,
                        batch_size=batch_size, im_processor=im_processor, cfg=cfg,
                        processes=num_process, shuffle=True,
                        dst_size=cfg.multi_scale_inp_size)
    elif data_name=='open_image':
        from datasets.open_image import OpenImageDataset
        from cfgs import config_open_image as cfg
        imdb = OpenImageDataset(data_type=data_type, datadir=datadir,
                        batch_size=batch_size, im_processor=im_processor, cfg=cfg,
                        processes=num_process, shuffle=True,
                        dst_size=cfg.multi_scale_inp_size)
    elif data_name=='open_image_for_robocup':
        from datasets.open_image_for_robocup import OpenImageForRobocupDataset
        from cfgs import config_open_image_for_robocup as cfg
        imdb = OpenImageForRobocupDataset(data_type=data_type, datadir=datadir,
                        batch_size=batch_size, im_processor=im_processor, cfg=cfg,
                        processes=num_process, shuffle=True,
                        dst_size=cfg.multi_scale_inp_size)

    loader = torch.utils.data.DataLoader(imdb, batch_size=batch_size, shuffle=True, num_workers=num_process)
    save_dir = os.path.join(save_root_path, data_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print('load data succ...')

    # Network
    net = Darknet19(data_name=data_name)
    net.load_from_npz(pretrained_model, num_conv=18)
    if gpu_id>=0:
        #net.cuda(gpu_id)
        net.cuda()
    net.train()
    print('load net succ...')

    # optimizer
    start_epoch = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)

    # tensorboad
    if use_tensorboard:
        from tensorboardX import SummaryWriter
        summary_writer = SummaryWriter(os.path.join(save_dir, exp_name))
    else:
        summary_writer = None

    # training
    print('start train')
    batch_per_epoch = imdb.batch_per_epoch
    train_loss = 0
    bbox_loss, iou_loss, cls_loss = 0., 0., 0.
    cnt = 0
    t = Timer()
    step_cnt = 0
    size_index = 8
    #for step in range(0,epoch * imdb.batch_per_epoch):
    for n in range(epoch):
        for step, ind in enumerate(loader):
            t.tic()
            # batch
            #batch = imdb.next_batch(size_index)
            batch = imdb.parse(ind, size_index)
            im = batch['images']
            gt_boxes = batch['gt_boxes']
            gt_classes = batch['gt_classes']
            dontcare = batch['dontcare']
            orgin_im = batch['origin_im']
            
            #print(im.shape)
            # forward
            im_data = net_utils.np_to_variable(im, gpu_id=gpu_id,
                                               volatile=False).permute(0, 3, 1, 2)
            bbox_pred, iou_pred, prob_pred = net(im_data, gt_boxes, gt_classes, dontcare, size_index, gpu_id)
            
            # backward
            loss = net.loss
            bbox_loss += net.bbox_loss.data.cpu().numpy()
            iou_loss += net.iou_loss.data.cpu().numpy()
            cls_loss += net.cls_loss.data.cpu().numpy()
            train_loss += loss.data.cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
            step_cnt += 1
            duration = t.toc()
            if (step+1) % disp_interval == 0:
                train_loss /= cnt
                bbox_loss /= cnt
                iou_loss /= cnt
                cls_loss /= cnt
                print(('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, '
                       'cls_loss: %.3f (%.2f s/batch, rest:%s)' %
                       (n, step_cnt, batch_per_epoch, train_loss, bbox_loss,
                        iou_loss, cls_loss, duration,
                        str(datetime.timedelta(seconds=int((batch_per_epoch - step_cnt) * duration))))))  # noqa

                if use_tensorboard:
                    summary_writer.add_scalar('loss_train', train_loss, step + n * batch_per_epoch)
                    summary_writer.add_scalar('loss_bbox', bbox_loss, step + n * batch_per_epoch)
                    summary_writer.add_scalar('loss_iou', iou_loss, step + n * batch_per_epoch)
                    summary_writer.add_scalar('loss_cls', cls_loss, step + n * batch_per_epoch)
                    summary_writer.add_scalar('learning_rate', lr, step + n * batch_per_epoch)

                    # plot results
                    bbox_pred = bbox_pred.data[0:1].cpu().numpy()
                    iou_pred = iou_pred.data[0:1].cpu().numpy()
                    prob_pred = prob_pred.data[0:1].cpu().numpy()
                    image = im[0]
                    bboxes, scores, cls_inds = yolo_utils.postprocess(
                        bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh=0.3, size_index=size_index, data_name=data_name)
                    im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)
                    summary_writer.add_image('predict', im2show, step + n * batch_per_epoch)

                train_loss = 0
                bbox_loss, iou_loss, cls_loss = 0., 0., 0.
                cnt = 0
                t.clear()

        if n in lr_decay_epochs:
            lr *= lr_decay_rate
            optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                        momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay)
            
        save_name = os.path.join(save_dir, '{}/epoch_{}.h5'.format(exp_name, n))
        net_utils.save_net(save_name, net)
        print(('save model: {}'.format(save_name)))
        size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)
        print("image_size {}".format(cfg.multi_scale_inp_size[size_index]))
        step_cnt = 0

    imdb.close()

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Yolo with Pytorch')
    parser.add_argument('-dn', '--data_name', default='voc', type=str, help='the dataset name')
    parser.add_argument('-dt', '--data_type', default='train', type=str, help='the dataset type, train or val or test')
    parser.add_argument('-y',  '--year', default=2007, type=int, help='the number of year (voc:2007 or 2012, mscoco:2014, 2017)')
    parser.add_argument('-b',  '--batch_size', default=10, type=int, help='the number of batch size')
    parser.add_argument('-np', '--num_process', default=5, type=int, help='the number of processes in data loading')
    parser.add_argument('-g',  '--gpu_id', default=-1, type=int, help='the gpu id')
    parser.add_argument('-l',  '--lr', default=0.01, type=float, help='the initial learning rate')
    parser.add_argument('-lde','--lr_decay_epochs', default=[60,90], type=float, nargs='*', help='the epoch of decay lr')
    parser.add_argument('-ldr','--lr_decay_rate', default=0.1, type=float, help='the decay rate of lr')
    parser.add_argument('-e',  '--epoch', default=100, type=int, help='the number of epoch')
    parser.add_argument('-di', '--disp_interval', default=100, type=int, help='the interval of display result')
    parser.add_argument('-en', '--exp_name', default='sample', help='the model name')
    parser.add_argument('-ut', '--use_tensorboard', default=False, type=bool, help='whether using tensorboard or not')
    args = parser.parse_args()
    main(args)
