from __future__ import division

from models import *
from utils.utils import *
from utils.utils import save_img
from utils.datasets import *
from utils.parse_config import *

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30000, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=12, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/visdrone.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/visdrone.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path)
# model.load_weights(opt.weights_path)
model.apply(weights_init_normal)
metrics = [
    "grid_size",
    "loss",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "cls",
    "cls_acc",
    "recall50",
    "recall75",
    "precision",
    "conf_obj",
    "conf_noobj",
]
if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataset = ListDataset(train_path)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=dataset.collate_fn, pin_memory=True
)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate/2, amsgrad=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 150)

for epoch in range(opt.epochs):
    start_time = time.time()
    scheduler.step()
    for batch_i, (paths, imgs, targets) in enumerate(dataloader):
        batches_done = len(dataloader) * epoch + batch_i
        optimizer.zero_grad()

        loss, outputs = model(imgs.cuda(), targets.cuda())

        loss.backward()
        optimizer.step()

        log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

        metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

        # Log metrics at each YOLO layer
        for i, metric in enumerate(metrics):
            formats = {m: "%.6f" for m in metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            row_metrics = [formats[metric] % yolo.metrics[metric] for yolo in model.yolo_layers]
            metric_table += [[metric, *row_metrics]]

        log_str += AsciiTable(metric_table).table

        global_metrics = [("Total Loss", loss.item())]

        # Print mAP and other global metrics
        log_str += "\n" + ", ".join([f"{metric_name} {metric:f}" for metric_name, metric in global_metrics])

        # Determine approximate time left for epoch
        epoch_batches_left = len(dataloader) - (batch_i + 1)
        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
        log_str += f"\n---- ETA {time_left}"

        print(log_str)

        model.seen += imgs.size(0)


    if epoch % 5 == 0:
        predictions = non_max_suppression(outputs, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
        targets[:, 1:] = xywh2xyxy(targets[:, 1:])
        # Rescale to image dimension
        targets[:, 1:] *= opt.img_size
        # Get batch statistics used to compute metrics
        statistics = get_batch_statistics(predictions, targets, iou_threshold=0.5)
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*statistics))]
        # Compute metrics
        precision, recall, AP, f1, ap_class = ap_per_class(
        true_positives, pred_scores, pred_labels, list(range(int(data_config["classes"])))
        )
        global_metrics = [
            ("F1", f1.mean()),
            ("Recall", recall.mean()),
            ("Precision", precision.mean()),
            ("mAP", AP.mean()),
        ]
        print(global_metrics)
        for img_i, (img_path, detections) in enumerate(zip(paths, predictions)):
           save_img(img_path, detections, classes, "output", str(epoch)+"_"+str(img_i), opt)


    if epoch % opt.checkpoint_interval == 0  or epoch == opt.epochs-1:
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
