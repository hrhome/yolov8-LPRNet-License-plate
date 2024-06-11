import torch
from torch import nn
from model import LPRNet
import numpy as np
import cv2
import time
# from general import color_detect
import os
import cv2


class LPR(nn.Module):
    def __init__(self):
        pass
    def transform(self,img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img
    def preprocess(self,images, alignment=True):
        """
        数据预处理
        :param images: BGR images
        :param alignment: 车牌倾斜矫正
        :return:
        """
        # if not isinstance(images, list): images = [images]
        # if alignment: images = self.plates_correction(images)
        image_tensors = []
        for img in images:
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            image_tensor = self.transform(img)
            image_tensors.append(image_tensor)
            image_tensors = torch.Tensor(image_tensors)
        return image_tensors
    def map_class_name(self, pred_labels):
        """
        :param pred_labels:
        :return:
        """
        CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
                 '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
                 '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
                 '新',
                 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z', 'I', 'O', '-'
                 ]
        pred_names = []
        for preb_label in pred_labels:
            preb_label = [CHARS[int(l)] for l in preb_label]
            pred_names.append("".join(preb_label))
        return pred_names
    def plates_recognize(self,model, images):
        """车牌识别"""
        image_tensors = []
        CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
                 '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
                 '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
                 '新',
                 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z', 'I', 'O', '-'
                 ]
        images = cv2.resize(images,(94,24))
        image = images
        image_tensor = self.transform(image)
        image_tensors.append(image_tensor)
        image_tensors = torch.Tensor(image_tensors)
        outputs = model(image_tensors)  # classifier prediction
        outputs = outputs.cpu().detach().numpy()
        imagess =[]
        preb_labels = outputs.argmax(axis=1)
        preb_labels,b = np.unique(preb_labels,axis=0,return_index = True)
        for i in b:
            imagess.append(cv2.resize(images[i],(94,24)))
        name_num =[]
        for preb_label in preb_labels:
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(int(pre_c))
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            name_num.append(no_repeat_blank_label)
        name_num1=[]
        images2 =[]
        for i in range(len(name_num)):
            if len(name_num[i])==7:
                name_num1.append(name_num[i])
                images2.append(imagess[i])
        plates = self.map_class_name(np.array(name_num1))
        result = {label: image for label,image in zip(plates,imagess)}
        return result
    def cv2_show(name, image):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)
        # 展示窗口等待时间，0表示任意键关闭窗口，
        # 其他数值表示毫秒后关闭窗口，如 cv2.waitKey(1000)表示1000毫秒后关闭窗口
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def read_images(path_b,images):
        for filename in os.listdir(path_b):
            if filename.endswith('.jpg'):  # 代表结尾是bmp格式的
                print(filename)
                img_path = path_b + '/' + filename
                img = cv2.imread(img_path)
                images.append(img)
        return images
