#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
import colorsys
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from nets.yolo4 import YoloBody
from utils.utils import (DecodeBox, bbox_iou, letterbox_image,
                         non_max_suppression, yolo_correct_boxes)


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path"        : 'model_data/Epoch102-Total_Loss11.0130-Val_Loss8.8086.pth',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "classes_path"      : 'model_data/helmet_classes.txt',
        "model_image_size"  : (416, 416, 3),
        "confidence"        : 0.5,
        "iou"               : 0.3,
        "cuda"              : False,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1,:,:]

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   建立yolov4模型
        #---------------------------------------------------#
        self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()

        #---------------------------------------------------#
        #   载入yolov4模型的权重
        #---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        print('Finished!')
        
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        #---------------------------------------------------#
        #   建立三个特征层解码用的工具
        #---------------------------------------------------#
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.anchors[i], len(self.class_names),  (self.model_image_size[1], self.model_image_size[0])))


        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img,dtype = np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()

            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            
        return output_list


def show_CAM(image_path, feature_maps, class_id, all_ids=10, show_one_layer=True):
            """
            feature_maps: this is a list [tensor,tensor,tensor], tensor shape is [1, 3, N, N, all_ids]
            """
            SHOW_NAME = ["score", "class", "class_score"]
            img_ori = cv2.imread(image_path)
            layers0 = feature_maps[0].reshape([-1, all_ids])
            layers1 = feature_maps[1].reshape([-1, all_ids])
            layers2 = feature_maps[2].reshape([-1, all_ids])
            layers = torch.cat([layers0, layers1, layers2], 0)
            score_max_v = layers[:, 4].max()  # compute max of score from all anchor
            score_min_v = layers[:, 4].min()  # compute min of score from all anchor
            class_max_v = layers[:, 5 + class_id].max()  # compute max of class from all anchor
            class_min_v = layers[:, 5 + class_id].min()  # compute min of class from all anchor
            all_ret = [[],[],[]]
            for j in range(3):  # layers
                layer_one = feature_maps[j]
                # compute max of score from three anchor of the layer
                anchors_score_max = layer_one[0, ..., 4].max(0)[0]
                # compute max of class from three anchor of the layer
                anchors_class_max = layer_one[0, ..., 5 + class_id].max(0)[0]
 
                scores = ((anchors_score_max - score_min_v) / (
                        score_max_v - score_min_v))
 
                classes = ((anchors_class_max - class_min_v) / (
                        class_max_v - class_min_v))
 
                layer_one_list = []
                layer_one_list.append(scores)
                layer_one_list.append(classes)
                layer_one_list.append(scores*classes)
                for idx, one in enumerate(layer_one_list):
                    layer_one = one.cpu().numpy()
                    ret = ((layer_one - layer_one.min()) / (layer_one.max() - layer_one.min())) * 255
                    ret = ret.astype(np.uint8)
                    gray = ret[:, :, None]
                    ret = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                    if not show_one_layer:
                        all_ret[j].append(cv2.resize(ret, (img_ori.shape[1], img_ori.shape[0])).copy())
                    else:
                        ret = cv2.resize(ret, (img_ori.shape[1], img_ori.shape[0]))
                        show = ret * 0.8 + img_ori * 0.2
                        show = show.astype(np.uint8)
                        cv2.imshow(f"one_{SHOW_NAME[idx]}", show)
                        cv2.imwrite('./cam_results/head'+str(j)+'layer'+str(idx)+SHOW_NAME[idx]+".jpg", show)
                        # cv2.imshow(f"map_{SHOW_NAME[idx]}", ret)
                if show_one_layer:
                    cv2.waitKey(0) 
            if not show_one_layer:
                for idx, one_type in enumerate(all_ret):
                    map_show = one_type[0] / 3 + one_type[1] / 3 + one_type[2] / 3
                    show = map_show * 0.8 + img_ori * 0.2
                    show = show.astype(np.uint8)
                    map_show = map_show.astype(np.uint8)
                    cv2.imshow(f"all_{SHOW_NAME[idx]}", show)
                    cv2.imwrite('./cam_results/head_cont'+str(idx)+SHOW_NAME[idx]+".jpg", show)
                    # cv2.imshow(f"map_{SHOW_NAME[idx]}", map_show)
                cv2.waitKey(0)


ret = []
stride = [13,26,52]
yolo = YOLO()
path = 'img/00148.jpg'
image = Image.open(path)
output_list = yolo.detect_image(image)
for i,f in enumerate(output_list):
    ret.append(f.reshape(1,3,stride[i],stride[i],10))

# features1 = torch.randn(1,3,13,13,10)
# features2 = torch.randn(1,3,26,26,10)
# features3 = torch.randn(1,3,52,52,10)

show_CAM(path, ret, 1)
