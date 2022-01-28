# -------------------------------------#
#       Model training
# -------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.model import YoloBody

from training.utils import weights_init, load_pretrained_weight
from training.loss import YOLOLoss

from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes


if __name__ == "__main__":
    # -------------------------------#
    #   Cuda status
    # -------------------------------#
    Cuda = True
    # --------------------------------------------------------#
    #   class path
    # --------------------------------------------------------#
    classes_path = 'model_data/smd_class_name.txt'
    # ---------------------------------------------------------------------#
    #   anchor path & mask
    # ---------------------------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   99% use pre-trained weights
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = 'model_data/yolov4.pth'  # 'data/ep017-loss10.131-val_loss10.271.pth' #'model_data/yolo4_weights.pth'
    # ------------------------------------------------------#
    #   multiple of 32
    # ------------------------------------------------------#
    input_shape = [416, 416]
    # ------------------------------------------------------#
    #   Yolov4 tricks
    #   mosaic augmentation : True or False
    #   Cosine_lr : True or False
    #   label_smoothing = 1
    # ------------------------------------------------------#
    mosaic = False
    Cosine_lr = True
    label_smoothing = 0

    # ----------------------------------------------------#
    #   training : freeze phase or unfreeze phase
    #   batch size : 8, 16, 32, ...
    # ----------------------------------------------------#
    # ----------------------------------------------------#
    #   frozen phase
    #   backbone is frozen, features remains unchanged
    #   takes up a small memory, only fine-tunes the network
    # ----------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 16
    Freeze_lr = 1e-3
    # ----------------------------------------------------#
    #   unfrozen phase
    #   backbone is changed
    #   a lot of gpu memory, all parameters are changed
    # ----------------------------------------------------#
    UnFreeze_Epoch = 200
    Unfreeze_batch_size = 8
    Unfreeze_lr = 1e-4
    # ------------------------------------------------------#
    #   frozen -> unfrozen
    # ------------------------------------------------------#
    Freeze_Train = False
    # ------------------------------------------------------#
    #   data load by multi thread
    #   read data faster, but more memory
    #   depends on RAM
    # ------------------------------------------------------#
    num_workers = 8
    # ----------------------------------------------------#
    #   image & label path
    # ----------------------------------------------------#
    train_annotation_path = 'smd_train.txt'
    val_annotation_path = 'smd_test.txt'

    # ----------------------------------------------------#
    #   get class and anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)


    resume = True
    resume_path = ''
    # ------------------------------------------------------#
    #   yolo model
    # ------------------------------------------------------#
    model = YoloBody(anchors_mask, num_classes)
    weights_init(model)

    model_train = model.train()
    """
    if resume == True:

        # Optimizer
        ckpt = torch.load(resume_path, map_location=device)  # load checkpoint
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict
    """
    if model_path != '':
        # ------------------------------------------------------#
        #   load pre-trained weight
        # ------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        load_pretrained_weight(model, model_path)


    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    loss_history = LossHistory("data/")

    # ---------------------------#
    #   load annotations
    # ---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # ------------------------------------------------------#
    #   Frozen phase training
    #   prevents weights from being destroyed at starting
    #   Init_Epoch : start epoch
    #   Freeze_Epoch : frozen training
    #   UnFreeze_Epoch : total training epoch
    # ------------------------------------------------------#
    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, mosaic=False, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Too small datasets to train model")

        # ------------------------------------#
        #   frozen backbone parameters
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        """
        for epoch in range(start_epoch, end_epoch):
            fit_test(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)


        exit()
        """

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, mosaic=False, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Too small datasets to train model")

        # ------------------------------------#
        #   unfrozen backbone parameters
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
