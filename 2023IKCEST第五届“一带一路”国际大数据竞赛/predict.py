# -*- coding:utf-8 -*-

from functools import partial
import time
import os 
import copy
import json
import gc
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import imghdr
import sys
import cv2
import logging
#from easydict import EasyDict as edict
import paddle
from paddlenlp.datasets import load_dataset
import paddle.nn.functional as F
import paddle.nn as nn
import paddlenlp as ppnlp
from paddlenlp.transformers import LinearDecayWithWarmup
from paddle.vision import transforms as T
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.vision import models
from paddlenlp.transformers import ErnieMModel,ErnieMTokenizer
from paddle.metric import Metric
os.environ["FLAGS_eager_delete_tensor_gb"] = "0.0"
sys.path.insert(0, "PaddleOCR-2.6.0")
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image
seed = 2021
paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)


#读取数据

logger = get_logger()
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
class Point_Attribute(dict):
    
    #给字典中加入点属性
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        #设置属性
        self[key] = value

    def __getattr__(self, key):  #当出现iris.datas时，调用getattr方法
        #获取属性
        self[key]
        return self[key]  

args = Point_Attribute(
use_gpu=False,
use_xpu=False, 
ir_optim=True, 
use_tensorrt=False, 
min_subgraph_size=15, 
shape_info_filename=None, 
precision='fp32', 
gpu_mem= 500, 
image_dir="/home/aistudio/queries_dataset_merge/val/img/1044.jpg",
det_algorithm='DB',
det_model_dir="PaddleOCR-2.6.0/inference/ch_PP-OCRv3_det_infer/", 
det_limit_side_len=960, 
det_limit_type='max', 
det_db_thresh=0.3, 
det_db_box_thresh=0.6, 
det_db_unclip_ratio=1.5, 
max_batch_size=10, 
use_dilation=False, 
det_db_score_mode='fast', 
det_east_score_thresh= 0.8, 
det_east_cover_thresh= 0.1,
det_east_nms_thresh=0.2, 
det_sast_score_thresh= 0.5,
det_sast_nms_thresh=0.2, 
det_sast_polygon=False,
det_pse_thresh= 0, 
det_pse_box_thresh=.85, 
det_pse_min_area=16,
det_pse_box_type='quad', 
det_pse_scale=1, 
scales=[8, 16, 32], 
alpha= 1.0, 
beta= 1.0, 
fourier_degree= 5, 
det_fce_box_type='poly', 
rec_algorithm= 'SVTR_LCNet', 
rec_model_dir="PaddleOCR-2.6.0/inference/ch_PP-OCRv3_rec_infer/", 
rec_image_shape='3, 48, 320',
rec_batch_num= 6, 
max_text_length= 25, 
rec_char_dict_path= 'PaddleOCR-2.6.0/ppocr/utils/ppocr_keys_v1.txt', 
use_space_char= True, 
vis_font_path= './doc/fonts/simfang.ttf', 
drop_score=0.5, e2e_algorithm='PGNet', 
e2e_model_dir= None, e2e_limit_side_len= 768, 
e2e_limit_type= 'max', e2e_pgnet_score_thresh= 0.5, 
e2e_char_dict_path= 'PaddleOCR-2.6.0/ppocr/utils/ic15_dict.txt',
e2e_pgnet_valid_set='totaltext', e2e_pgnet_mode= 'fast',
use_angle_cls= False, cls_model_dir= None, cls_image_shape= '3, 48, 192',
label_list= ['0', '180'], cls_batch_num= 6, cls_thresh= 0.9, enable_mkldnn= False,
cpu_threads= 10, use_pdserving=False, warmup= False, sr_model_dir= None, sr_image_shape= '3, 32, 128',
sr_batch_num=1, draw_img_save_dir= 'PaddleOCR-2.6.0/ch_PP-OCRv2_results', save_crop_res=False, crop_res_save_dir= './output',
use_mp= False, total_process_num= 1, process_id= 0, benchmark= False, save_log_path= './log_output/', show_log=True, use_onnx= False)

class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'csl': 0, 'all': 0}
        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse
        # logger.debug("dt_boxes num : {}, elapse : {}".format(
        #     len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        # logger.debug("rec_res num  : {}, elapse : {}".format(
        #     len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, 0, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def orc_image(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []

    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()
    count = 0
    for idx, image_file in enumerate(image_file_list):

        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            #logger.debug("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res, time_dict = text_sys(img)
        elapse = time.time() - starttime
        total_time += elapse

        # logger.debug(
        #     str(idx) + "  Predict time of %s: %.3fs" % (image_file, elapse))
        # for text, score in rec_res:
        #     logger.debug("{}, {:.3f}".format(text, score))

        res = [{
            "transcription": rec_res[idx][0],
            "points": np.array(dt_boxes[idx]).astype(np.int32).tolist(),
        } for idx in range(len(dt_boxes))]
        save_pred = os.path.basename(image_file) + "\t" + json.dumps(
            res, ensure_ascii=False) + "\n"
        save_results.append(save_pred)

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]
    if args.benchmark:
        text_sys.text_detector.autolog.report()
        text_sys.text_recognizer.autolog.report()
    text_all = []
    if len(save_results) != 0:
        for i in eval(save_results[0].split("\t")[1]):
            text_all.append(i['transcription'])
    return ",".join(text_all)

def process_string(input_str):
    input_str = input_str.replace('&#39;', ' ')
    input_str = input_str.replace('<b>','')
    input_str = input_str.replace('</b>','')
    #input_str = unidecode(input_str)  
    return input_str
    
class NewsContextDatasetEmbs(Dataset):
    def __init__(self, context_data_items_dict, queries_root_dir, split):
        self.context_data_items_dict = context_data_items_dict
        self.queries_root_dir = queries_root_dir
        self.idx_to_keys = list(context_data_items_dict.keys())
        self.transform =T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        self.split=split
    def __len__(self):
        return len(self.context_data_items_dict)   


    def load_img_pil(self,image_path):
        if imghdr.what(image_path) == 'gif': 
            try:
                with open(image_path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('RGB')
            except:
                return None 
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def load_imgs_direct_search(self,item_folder_path,direct_dict):   
        list_imgs_tensors = []
        count = 0   
        keys_to_check = ['images_with_captions','images_with_no_captions','images_with_caption_matched_tags']
        for key1 in keys_to_check:
            if key1 in direct_dict.keys():
                for page in direct_dict[key1]:
                    image_path = os.path.join(item_folder_path,page['image_path'].split('/')[-1])
                    try:
                        pil_img = self.load_img_pil(image_path)
                    except Exception as e:
                        print(e)
                        print(image_path)
                    if pil_img == None: continue
                    transform_img = self.transform(pil_img)
                    count = count + 1 
                    list_imgs_tensors.append(transform_img)
        stacked_tensors = paddle.stack(list_imgs_tensors, axis=0)
        return stacked_tensors

    def load_captions(self,inv_dict):
        captions = ['']
        pages_with_captions_keys = ['all_fully_matched_captions','all_partially_matched_captions']
        for key1 in pages_with_captions_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        item = page['title']
                        item = process_string(item)
                        captions.append(item)
                    
                    if 'caption' in page.keys():
                        sub_captions_list = []
                        unfiltered_captions = []
                        for key2 in page['caption']:
                            sub_caption = page['caption'][key2]
                            sub_caption_filter = process_string(sub_caption)
                            if sub_caption in unfiltered_captions: continue 
                            sub_captions_list.append(sub_caption_filter) 
                            unfiltered_captions.append(sub_caption) 
                        captions = captions + sub_captions_list 
                    
        pages_with_title_only_keys = ['partially_matched_no_text','fully_matched_no_text']
        for key1 in pages_with_title_only_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        title = process_string(page['title'])
                        captions.append(title)
        return captions

    def load_captions_weibo(self,direct_dict):
        captions = ['']
        keys = ['images_with_captions','images_with_no_captions','images_with_caption_matched_tags']
        for key1 in keys:
            if key1 in direct_dict.keys():
                for page in direct_dict[key1]:
                    if 'page_title' in page.keys():
                        item = page['page_title']
                        item = process_string(item)
                        captions.append(item)
                    if 'caption' in page.keys():
                        sub_captions_list = []
                        unfiltered_captions = []
                        for key2 in page['caption']:
                            sub_caption = page['caption'][key2]
                            sub_caption_filter = process_string(sub_caption)
                            if sub_caption in unfiltered_captions: continue 
                            sub_captions_list.append(sub_caption_filter) 
                            unfiltered_captions.append(sub_caption) 
                        captions = captions + sub_captions_list 
        #print(captions)
        return captions
        #加载img文件夹
    def load_queries(self,key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.queries_root_dir,self.context_data_items_dict[key]['image_path'])
        pil_img = self.load_img_pil(image_path)
        # # ======================
        # with open(image_path+".txt") as f:
        #     data =f.readlines()
        # text_all = []
        # if len(data) !=0:
        #     for i in eval(data[0].split("\t")[1]):
        #         text_all.append(i['transcription'])
        #     ocr_image_text = [",".join(text_all)]
        # else:
        #     ocr_image_text = [""]
        # # ======================
        args.image_dir = image_path
        ocr_image_text = [orc_image(args)]
        transform_img = self.transform(pil_img)
        return transform_img, caption,ocr_image_text
    def __getitem__(self, idx):
        #print(idx)
        #print(self.context_data_items_dict)      
        #idx = idx.tolist()               
        key = self.idx_to_keys[idx]
        #print(key)
        item=self.context_data_items_dict.get(str(key))
        #print(item)
        # 如果为test没有label属性
        #print(self.split)
        if self.split=='train' or self.split=='val':
            label = paddle.to_tensor(int(item['label']))
            direct_path_item = os.path.join(self.queries_root_dir,item['direct_path'])
            inverse_path_item = os.path.join(self.queries_root_dir,item['inv_path'])
            inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json')))
            direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json')))
            captions= self.load_captions(inv_ann_dict)
            captions += self.load_captions_weibo(direct_dict)
            imgs = self.load_imgs_direct_search(direct_path_item,direct_dict)  
            #imgs_text = self.load_imgs_direct_search_text(direct_path_item,direct_dict)  
            qImg,qCap,imgs_text =  self.load_queries(key)
            sample = {'label': label, 'caption': captions,'imgs': imgs,  'qImg': qImg, 'qCap': qCap,"imgs_text":imgs_text}
        else:
            direct_path_item = os.path.join(self.queries_root_dir,item['direct_path'])
            inverse_path_item = os.path.join(self.queries_root_dir,item['inv_path'])
            inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json')))
            direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json')))
            captions= self.load_captions(inv_ann_dict)
            captions += self.load_captions_weibo(direct_dict)
            imgs = self.load_imgs_direct_search(direct_path_item,direct_dict)    
            #imgs_text = self.load_imgs_direct_search_text(direct_path_item,direct_dict) 
            qImg,qCap,imgs_text =  self.load_queries(key)
            sample = {'caption': captions,'imgs': imgs,  'qImg': qImg, 'qCap': qCap,"imgs_text":imgs_text}
        return sample,  len(captions), imgs.shape[0]
def collate_context_bert_test(batch):
    samples = [item[0] for item in batch]
    max_captions_len = max([item[1] for item in batch])
    max_images_len = max([item[2] for item in batch])
    qCap_batch = []
    qImg_batch = []
    img_batch = []
    img_text_batch = []
    cap_batch = []
    for j in range(0,len(samples)):  
        sample = samples[j]    
        captions = sample['caption']
        cap_len = len(captions)
        for i in range(0,max_captions_len-cap_len):
            captions.append("")
        if len(sample['imgs'].shape) > 2:
            padding_size = (max_images_len-sample['imgs'].shape[0],sample['imgs'].shape[1],sample['imgs'].shape[2],sample['imgs'].shape[3])
        else:
            padding_size = (max_images_len-sample['imgs'].shape[0],sample['imgs'].shape[1])
        padded_mem_img = paddle.concat((sample['imgs'], paddle.zeros(padding_size)),axis=0)
        img_batch.append(padded_mem_img)
        cap_batch.append(captions)
        img_text_batch.append(sample['imgs_text'])
        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])        
    img_batch = paddle.stack(img_batch, axis=0)
    qImg_batch = paddle.stack(qImg_batch, axis=0)
    return cap_batch, img_batch, qCap_batch, qImg_batch,img_text_batch

class EncoderCNN(nn.Layer):
    def __init__(self, resnet_arch = 'resnext101_64x4d'):
        super(EncoderCNN, self).__init__()
        if resnet_arch == 'resnet101':
            resnet = models.resnet101(pretrained=False)
        elif resnet_arch == 'resnext101_64x4d':
            resnet = models.resnext101_64x4d(pretrained=False)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2D((1, 1))
    def forward(self, images, features='pool'):
        out = self.resnet(images)
        if features == 'pool':
            out = self.adaptive_pool(out)
            out = paddle.reshape(out, (out.shape[0],out.shape[1]))
        return out


class NetWork(nn.Layer):
    def __init__(self, mode):
        super(NetWork, self).__init__()
        self.mode = mode           
        # self.ernie = ErnieMModel.from_pretrained('ernie-m-base')
        # self.tokenizer = ErnieMTokenizer.from_pretrained('ernie-m-base')
        
        self.ernie = ErnieMModel.from_pretrained('./ernie-m-base')
        self.tokenizer = ErnieMTokenizer.from_pretrained('./ernie-m-base')

        # self.ernie = ErnieMModel.from_pretrained(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ernie-m-base'))
        # self.tokenizer = ErnieMTokenizer.from_pretrained(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ernie/'))
        
        self.resnet = EncoderCNN()
        # 只能存在一个
        self.ocr = False
        self.fusion_type = True
        if self.ocr:
            self.classifier1 = nn.Linear(1*(768+2048)+768*4,1024) 
        self.classifier2 = nn.Linear(1024,3)
        # self.attention_text1 = nn.Linear(128*768,2048*5) 
        # self.attention_text2 = nn.Linear(2048*5,2048)
        self.attention_text = nn.MultiHeadAttention(768,16)
        self.attention_image = nn.MultiHeadAttention(2048,16)
        self.attention_image_text = nn.MultiHeadAttention(768,4)
        self.mid_attention_image_text = nn.Linear(2048,128*768) 
        # 创建可学习的系数参数
        self.weight1 = paddle.create_parameter(shape=[1], dtype='float32', name="learnable_coeff1")
        self.weight2 = paddle.create_parameter(shape=[1], dtype='float32', name="learnable_coeff2")
        self.weight3 = paddle.create_parameter(shape=[1], dtype='float32', name="learnable_coeff3")
        # 设置初始权重为可学习的系数
        self.weight1.set_value(paddle.to_tensor([1.], dtype='float32'))
        self.weight2.set_value(paddle.to_tensor([1.], dtype='float32'))
        self.weight3.set_value(paddle.to_tensor([1.], dtype='float32'))
        self.mlp1 = paddle.nn.Sequential(
                paddle.nn.Linear(2048*2+768*3, 1024),
                paddle.nn.ReLU(),
                paddle.nn.Dropout(0.2),
                paddle.nn.Linear(1024, 3)
            )

        self.mlp2 = paddle.nn.Sequential(
                paddle.nn.Linear(2048*1+768*4, 1024),
                paddle.nn.ReLU(),
                paddle.nn.Dropout(0.2),
                paddle.nn.Linear(1024, 3)
            )

        self.mlp3 = paddle.nn.Sequential(
                paddle.nn.Linear(768, 256),
                paddle.nn.ReLU(),
                paddle.nn.Dropout(0.2),
                paddle.nn.Linear(256, 3)
            )

        self.mlp4 = paddle.nn.Sequential(
                paddle.nn.Linear(2048, 1024),
                paddle.nn.ReLU(),
                paddle.nn.Dropout(0.2),
                paddle.nn.Linear(1024, 3)
            )

        self.fusion_mlp = paddle.nn.Sequential(
                        paddle.nn.Linear(2048, 768),
                        paddle.nn.ReLU(),
                        paddle.nn.Dropout(0.2),
                    )
        self.fusion_mlp2 = paddle.nn.Sequential(
                paddle.nn.Linear(768, 256),
                paddle.nn.ReLU(),
                paddle.nn.Dropout(0.2),
                paddle.nn.Linear(256, 3)
            )
        self.fusion_attention1 = nn.MultiHeadAttention(768,16)
        self.fusion_attention2 = nn.MultiHeadAttention(768,16)
        self.fusion_attention3 = nn.MultiHeadAttention(768,16)
        self.fusion_attention4 = nn.MultiHeadAttention(768,16)
        self.dropout = nn.Dropout(p=0.5)
        if self.mode == 'text':
            self.classifier = nn.Linear(768,3)
        self.resnet.eval()

    def forward(self,qCap,qImg,caps,imgs,img_text):
        self.resnet.eval()
        encode_dict_qcap = self.tokenizer(text = qCap ,max_length = 128 ,truncation=True, padding='max_length')
        input_ids_qcap = encode_dict_qcap['input_ids']
        input_ids_qcap = paddle.to_tensor(input_ids_qcap)
        qcap_feature, pooled_output= self.ernie(input_ids_qcap) #(b,length,dim)
        if self.mode == 'text':
            logits = self.classifier(qcap_feature[:,0,:].squeeze(1))
            return logits
        caps_feature = []
        for i,caption in enumerate (caps):
            encode_dict_cap = self.tokenizer(text = caption ,max_length = 128 ,truncation=True, padding='max_length')
            input_ids_caps = encode_dict_cap['input_ids']
            input_ids_caps = paddle.to_tensor(input_ids_caps)
            cap_feature, pooled_output= self.ernie(input_ids_caps) #(b,length,dim)
            caps_feature.append(cap_feature)
        caps_feature = paddle.stack(caps_feature,axis=0) #(b,num,length,dim)
        caps_feature = caps_feature.mean(axis=1)#(b,length,dim)
        caps_feature_fusion = self.attention_text(qcap_feature,caps_feature,caps_feature) #(b,length,dim)
        # =======================================================================
        if True:#self.ocr:
            img_text_feature = []
            for i,img_caption in enumerate (img_text):
                encode_dict_cap_img = self.tokenizer(text = img_caption ,max_length = 128 ,truncation=True, padding='max_length')
                input_ids_caps_img = encode_dict_cap_img['input_ids']
                input_ids_caps_img = paddle.to_tensor(input_ids_caps_img)
                cap_img_feature, pooled_img_output= self.ernie(input_ids_caps_img) #(b,length,dim)
                img_text_feature.append(cap_img_feature)
            img_text_feature = paddle.stack(img_text_feature,axis=0) #(b,num,length,dim)
            img_text_feature = img_text_feature.mean(axis=1)#(b,length,dim)
            img_text_feature_qcap = self.attention_text(qcap_feature,img_text_feature,img_text_feature) #(b,length,dim)
            img_text_feature_caps = self.attention_text(caps_feature,img_text_feature,img_text_feature) #(b,length,dim)
        # =======================================================================
        imgs_features = []
        for img in imgs:
            imgs_feature = self.resnet(img) #(length,dim)
            imgs_features.append(imgs_feature)
        imgs_features = paddle.stack(imgs_features,axis=0) #(b,length,dim)
        qImg_features = []
        for qImage in qImg:
            qImg_feature = self.resnet(qImage.unsqueeze(axis=0)) #(1,dim)
            qImg_features.append(qImg_feature)
        qImg_feature = paddle.stack(qImg_features,axis=0) #(b,1,dim)
        imgs_features = self.attention_image(qImg_feature,imgs_features,imgs_features) #(b,1,dim)
        # [1, 128, 768] [1, 128, 768] [1, 1, 2048] [1, 1, 2048] origin
        # print(qcap_feature.shape,caps_feature.shape,qImg_feature.shape,imgs_features.shape)
        # print((qcap_feature[:,0,:].shape,caps_feature[:,0,:].shape,qImg_feature.squeeze(1).shape,imgs_features.squeeze(1).shape))
        # ([1,768], [1 , 768], [1, 2048], [1,  2048])
        #imgs_text_features = self.attention_image_text(imgs_features,imgs_features,self.attention_text2(self.attention_text1(qcap_feature.reshape([qcap_feature.shape[0],1,-1]))))
        # 计算用的是融合后的图像信息，而不是imgs_feature,可能会有问题？
        # 图像和文本证据融合
        imgs_features_qcap = (self.mid_attention_image_text(imgs_features)).reshape([qcap_feature.shape[0],qcap_feature.shape[1],qcap_feature.shape[2]])
        imgs_text_features_fusion_qcap = self.attention_image_text(imgs_features_qcap,imgs_features_qcap,qcap_feature)
        # 图像和文本融合
        imgs_features_caps = (self.mid_attention_image_text(imgs_features)).reshape([caps_feature.shape[0],caps_feature.shape[1],caps_feature.shape[2]])
        imgs_text_features_fusion_caps = self.attention_image_text(imgs_features_caps,imgs_features_caps,caps_feature)
        if self.ocr:
            pass
            # feature = paddle.concat(x=[caps_feature_fusion[:,0,:], imgs_text_features_fusion[:,0,:],imgs_features.squeeze(1),img_text_feature_qcap[:,0,:],img_text_feature_caps[:,0,:]], axis=-1)
        #feature = paddle.concat(x=[qcap_feature[:,0,:], caps_feature[:,0,:], img_text_feature[:,0,:],qImg_feature.squeeze(1), imgs_features.squeeze(1),imgs_text_features[:,0,:]], axis=-1) 
        # logits = self.classifier1(feature)
        # logits = self.classifier2(logits)
        def fusion():
            # 早期融合
            feature1 = paddle.concat(x=[qcap_feature[:,0,:], caps_feature[:,0,:],imgs_features[:,0,:],qImg_feature[:,0,:],img_text_feature[:,0,:]], axis=-1) 
            logits1 = self.mlp1(feature1)
            # 交叉融合
            fusion1,fusion2,fusion3,fusion4,fusion5,fusion6 = caps_feature_fusion,imgs_text_features_fusion_qcap,imgs_features,imgs_text_features_fusion_caps,img_text_feature_qcap,img_text_feature_caps
            if True:
                fusion3 = self.fusion_mlp(fusion3)
                fusion3 = paddle.tile(fusion3 , repeat_times=[128, 1])
                
                x1 = self.fusion_attention1(fusion1,fusion2,fusion2) + fusion1
                x2 = self.fusion_attention1(fusion2,fusion3,fusion3) + fusion2
                x3 = self.fusion_attention1(fusion3,fusion4,fusion4) + fusion3
                x4 = self.fusion_attention1(fusion4,fusion5,fusion5) + fusion4
                x5 = self.fusion_attention1(fusion5,fusion6,fusion6) + fusion6
                x2_1 = self.fusion_attention2(x1,x2,x2) + x1
                x2_2 = self.fusion_attention2(x2,x3,x3) + x2
                x2_3 = self.fusion_attention2(x3,x4,x4) + x3
                x2_4 = self.fusion_attention2(x4,x5,x5) + x4
                x3_1 = self.fusion_attention3(x2_1,x2_2,x2_2) + x2_1
                x3_2 = self.fusion_attention3(x2_2,x2_3,x2_3) + x2_2
                x3_3 = self.fusion_attention3(x2_3,x2_4,x2_4) + x2_3
                x4_1 = self.fusion_attention4(x3_1,x3_2,x3_2) + x3_1
                x4_2 = self.fusion_attention4(x3_2,x3_3,x3_3) + x3_2
                x5_1 = self.fusion_attention4(x4_1,x4_2,x4_2) + x4_1
                logits2 = self.fusion_mlp2(x5_1[:,0,:])
            else:
                feature2 = paddle.concat(x=[fusion1,fusion2,fusion3.squeeze(1),fusion4,fusion5,fusion6], axis=-1) 
                logits2 = self.mlp2(feature2)
            # 5个模态的特征，经过全连接层，进行晚期融合
            muti1 = self.mlp3(qcap_feature[:,0,:])
            muti2 = self.mlp3(caps_feature[:,0,:])
            muti3 = self.mlp4(imgs_features[:,0,:])
            muti4 = self.mlp4(qImg_feature[:,0,:])
            muti5 = self.mlp3(img_text_feature[:,0,:])
            logits3 =  muti1 + muti2 + muti3 + muti4 + muti5
            # 三种方式进行融合，还不知系数，因此先都为1

            #logits = logits2+logits1 + logits3
            logits = self.weight2*logits2 + self.weight1*logits1 + self.weight3*logits3
            return logits

        
        if self.fusion_type:
            logits = fusion()
        else:
            # logits = self.classifier1(feature)
            # logits = self.classifier2(logits)
            logits = F.relu(self.classifier1(feature))
            logits = self.dropout(logits)
            #logits = self.classifier1(feature)
            #output = self.bn_1(output)
            logits = self.classifier2(logits)
        return logits

def evaluate(model, result_csv):
    results = []
    #切换model模型为评估模式，关闭dropout等随机因素
    model.eval()
    count=0
    for batch in test_dataloader:
        print(count)
        gc.collect()
        count+=1
        cap_batch, img_batch, qCap_batch, qImg_batch,img_text_batch= batch
        logits = model(qCap=qCap_batch,qImg=qImg_batch,caps=cap_batch,imgs=img_batch,img_text=img_text_batch)
        gc.collect()
        # 预测分类
        probs = F.softmax(logits, axis=-1)
        label = paddle.argmax(probs, axis=1).numpy()
        results += label.tolist()
    print(results[:5])
    # 输出结果
    #id/label
    #字典中的key值即为csv中的列名
    id_list=range(len(results))
    print(id_list)
    for i in range(len(results)):
        if results[i] == 0:
            results[i] = "non-rumor"
        elif results[i] == 1:
            results[i] = "rumor"
        else:
            results[i] = "unverified"
    frame = pd.DataFrame({'id':id_list,'label':results})
    frame.to_csv(result_csv,index=False,sep=',')

if __name__ == '__main__':
    test_csv = sys.argv[1]  # 测试集路径
    result_csv = sys.argv[2]  # 结果文件路径
    # test_csv = '/home/aistudio/queries_dataset_merge/dataset_items_test-Copy1.json'  # 测试集路径
    # result_csv = 'result.csv'  # 结果文件路径
    data_items_test = json.load(open(test_csv))# 处理测试集数据
  
    test_dataset = NewsContextDatasetEmbs(data_items_test,os.path.dirname(os.path.realpath(__file__)),'test')
    #test_dataset = NewsContextDatasetEmbs(data_items_test,'/home/aistudio/queries_dataset_merge','test')
    # load DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn = collate_context_bert_test, return_list=True)
    # 声明模型
    model = NetWork("image")

    # # 根据实际运行情况，更换加载的参数路径
    params_path = 'model/model_state.pdparams'
    #params_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model/model_state.pdparams')
    if params_path and os.path.isfile(params_path):
        # 加载模型参数
        state_dict = paddle.load(params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % params_path)
        del state_dict

    # 加载模型
    evaluate(model, result_csv=result_csv)  # 预测测试集
