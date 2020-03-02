import os
import time
import torch
import json
import numpy as np
import time
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from net.loss import *
from net.network_sn_101 import ACSPNet
from config import Config
from dataloader.loader import *
from util.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate
from sys import exit

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = Config()
config.test_path = './data/citypersons'
config.size_test = (1280, 2560)
config.init_lr = 2e-4
config.offset = True
config.val = True
config.val_frequency = 1
config.teacher = True
config.print_conf()

# dataset
testtransform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
testdataset = CityPersons(path=config.train_path, type='val', config=config, transform=testtransform, preloaded=True)
testloader = DataLoader(testdataset, batch_size=1)

# net
print('Net...')
net = ACSPNet().cuda()


# position
center = cls_pos().cuda()
height = reg_pos().cuda()
offset = offset_pos().cuda()

teacher_dict = net.state_dict()


def val(r, name, log=None):
    
    net.eval()
    #load the model here!!!
    teacher_dict = torch.load(name)
    net.load_state_dict(teacher_dict)
     
    print(net)
    print('Perform validation...')
    res = []
    t3 = time.time()
    for i, data in enumerate(testloader, 0):
        inputs = data.cuda()
        with torch.no_grad():
            pos, height, offset = net(inputs)

        boxes = parse_det_offset(r, pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test, score=0.1, down=4, nms_thresh=0.5)
        if len(boxes) > 0:
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]

            for box in boxes:
                temp = dict()
                temp['image_id'] = i+1
                temp['category_id'] = 1
                temp['bbox'] = box[:4].tolist()
                temp['score'] = float(box[4])
                res.append(temp)

        print('\r%d/%d' % (i + 1, len(testloader))),
        sys.stdout.flush()
    print('')

    with open('./_temp_val.json', 'w') as f:
        json.dump(res, f)

    MRs = validate('./eval_city/val_gt.json', './_temp_val.json')
    t4 = time.time()
    print(name)
    print('Summarize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
          % (MRs[0]*100, MRs[1]*100, MRs[2]*100, MRs[3]*100))
    if log is not None:
        log.write("%.7f %.7f %.7f %.7f\n" % tuple(MRs))
    print('Validation time used: %.3f' % (t4 - t3))
    return MRs[0]



name_1 = './models/ACSP(Smooth L1).pth.tea'
name_2 = './models/ACSP(Vanilla L1).pth.tea'
#or Val your own model
#name = './ckpt/ACSP_XXX.pth.tea'
#val(0.36, name)

val(0.40, name_1)
val(0.36, name_2)
