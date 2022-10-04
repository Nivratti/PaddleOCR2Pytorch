import os
import sys
import importlib

import cv2
import logging
import numpy as np
from pathlib import Path
from argparse import Namespace
from loguru import logger
import copy
import time

__dir__ = os.path.dirname(__file__)

sys.path.append(os.path.join(__dir__, ''))

# from predict_det import TextDetector
# from predict_rec import TextRecognizer
# from predict_cls import TextClassifier
# from infer import (
#     get_default_args, 
#     sorted_boxes,
#     get_rotate_crop_image
# )
import cv2
import copy
import numpy as np
import time
from PIL import Image
# import tools.infer.pytorchocr_utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from pytorchocr.utils.utility import get_image_file_list, check_and_read_gif
from tools.infer.pytorchocr_utility import draw_ocr_box_txt

from tools.infer.predict_system import TextSystem
from tools.infer.pytorchocr_utility import parse_args

def get_default_args():
    from argparse import Namespace

    default_dict = {
        'use_gpu': True,
        'gpu_mem': 500,
        'image_dir': None,
        'det_algorithm': 'DB',
        'det_model_path': None,
        'det_limit_side_len': 960,
        'det_limit_type': 'max',
        'det_db_thresh': 0.3,
        'det_db_box_thresh': 0.6,
        'det_db_unclip_ratio': 1.5,
        'max_batch_size': 10,
        'use_dilation': False,
        'det_db_score_mode': 'fast',
        'det_east_score_thresh': 0.8,
        'det_east_cover_thresh': 0.1,
        'det_east_nms_thresh': 0.2,
        'det_sast_score_thresh': 0.5,
        'det_sast_nms_thresh': 0.2,
        'det_sast_polygon': False,
        'det_pse_thresh': 0,
        'det_pse_box_thresh': 0.85,
        'det_pse_min_area': 16,
        'det_pse_box_type': 'box',
        'det_pse_scale': 1,
        'scales': [8, 16, 32],
        'alpha': 1.0,
        'beta': 1.0,
        'fourier_degree': 5,
        'det_fce_box_type': 'poly',
        'rec_algorithm': 'CRNN',
        'rec_model_path': None,
        'rec_image_shape': '3, 32, 320',
        'rec_char_type': 'ch',
        'rec_batch_num': 6,
        'max_text_length': 25,
        'use_space_char': True,
        'drop_score': 0.5,
        'limited_max_width': 1280,
        'limited_min_width': 16,
        'vis_font_path': './doc/fonts/simfang.ttf',
        'rec_char_dict_path': './pytorchocr/utils/ppocr_keys_v1.txt',
        'use_angle_cls': False,
        'cls_model_path': None,
        'cls_image_shape': '3, 48, 192',
        'label_list': ['0', '180'],
        'cls_batch_num': 6,
        'cls_thresh': 0.9,
        'enable_mkldnn': False,
        'use_pdserving': False,
        'e2e_algorithm': 'PGNet',
        'e2e_model_path': None,
        'e2e_limit_side_len': 768,
        'e2e_limit_type': 'max',
        'e2e_pgnet_score_thresh': 0.5,
        'e2e_char_dict_path': './pytorchocr/utils/ic15_dict.txt',
        'e2e_pgnet_valid_set': 'totaltext',
        'e2e_pgnet_polygon': True,
        'e2e_pgnet_mode': 'fast',
        'det_yaml_path': None,
        'rec_yaml_path': None,
        'cls_yaml_path': None,
        'e2e_yaml_path': None,
        'use_mp': False,
        'total_process_num': 1,
        'process_id': 0,
        'benchmark': False,
        'save_log_path': './log_output/',
        'show_log': True
    }
    args = Namespace(**default_dict)
    return args


class PaddlePytorchOCR(TextSystem):
    def __init__(self, **kwargs):
        params = {**kwargs}

        # args = parse_args() # 
        args = get_default_args()
        
        # import ipdb;ipdb.set_trace()
        ## update args if params available
        ## Prefer params  value over args
        args.__dict__.update(params)
        
        self.args = args
        
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.crop_image_res_index = 0

        # debug print
        print(args.__dict__)

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

    # def __call__(self, img, cls=True):
    #     # import ipdb; ipdb.set_trace()
    #     time_dict = {'det': 0, 'rec': 0, 'csl': 0, 'all': 0}
    #     start = time.time()
    #     ori_im = img.copy()
    #     dt_boxes, elapse = self.text_detector(img)
    #     time_dict['det'] = elapse
    #     logger.debug("dt_boxes num : {}, elapse : {}".format(
    #         len(dt_boxes), elapse))
    #     if dt_boxes is None:
    #         return None, None
    #     img_crop_list = []

    #     dt_boxes = sorted_boxes(dt_boxes)

    #     for bno in range(len(dt_boxes)):
    #         tmp_box = copy.deepcopy(dt_boxes[bno])
    #         img_crop = get_rotate_crop_image(ori_im, tmp_box)
    #         img_crop_list.append(img_crop)
    #     if self.use_angle_cls and cls:
    #         img_crop_list, angle_list, elapse = self.text_classifier(
    #             img_crop_list)
    #         time_dict['cls'] = elapse
    #         logger.debug("cls num  : {}, elapse : {}".format(
    #             len(img_crop_list), elapse))

    #     rec_res, elapse = self.text_recognizer(img_crop_list)
    #     time_dict['rec'] = elapse
    #     logger.debug("rec_res num  : {}, elapse : {}".format(
    #         len(rec_res), elapse))
    #     if self.args.save_crop_res:
    #         self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
    #                                rec_res)
    #     filter_boxes, filter_rec_res = [], []
    #     for box, rec_result in zip(dt_boxes, rec_res):
    #         text, score = rec_result
    #         if score >= self.drop_score:
    #             filter_boxes.append(box)
    #             filter_rec_res.append(rec_result)
    #     end = time.time()
    #     time_dict['all'] = end - start
    #     return filter_boxes, filter_rec_res, time_dict

    def ocr(self, img, det=True, rec=True, cls=True):
        """
        ocr
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not. If false, only rec will be exec. Default is True
            rec: use text recognition or not. If false, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If true, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
        """

        assert isinstance(img, (np.ndarray, list, str))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                'Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process'
            )

        if isinstance(img, str):
            image_file = img
            with open(image_file, 'rb') as f:
                np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None

        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if det and rec:
            # dt_boxes, rec_res, _ = self.__call__(img, cls)
            dt_boxes, rec_res = self.__call__(img)
            return [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        elif det and not rec:
            dt_boxes, elapse = self.text_detector(img)
            if dt_boxes is None:
                return None
            return [box.tolist() for box in dt_boxes]
        else:
            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls and cls:
                img, cls_res, elapse = self.text_classifier(img)
                if not rec:
                    return cls_res
            rec_res, elapse = self.text_recognizer(img)
            return rec_res

    
def main():
    from pathlib import Path

    __dir__ = os.path.dirname(__file__)

    ## Get home dir
    home = str(Path.home())

    ## model dir
    pdtorch_model_dir = os.path.join(
        home, ".ocr_models", "PaddleOCR2Pytorch-models"
    )

    ocr_engine = PaddlePytorchOCR(
        det_model_path=os.path.join(pdtorch_model_dir, "en_ptocr_v3_det_infer.pth"),
        det_yaml_path=os.path.join(__dir__, "configs/det/det_ppocr_v3.yml"),
        
        use_angle_cls=True,
        cls_model_path=os.path.join(pdtorch_model_dir, "ch_ptocr_mobile_v2.0_cls_infer.pth"),
        cls_yaml_path=os.path.join(__dir__, "configs/cls/cls_mv3.yml"),

        rec_model_path=os.path.join(pdtorch_model_dir, "en_ptocr_v3_rec_infer.pth"),
        rec_yaml_path=os.path.join(__dir__, "configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml"),
        rec_char_dict_path=os.path.join(__dir__, "pytorchocr/utils/en_dict.txt"),
    )
    rec_res = ocr_engine.ocr(os.path.join(__dir__, "../images/pan-card-500x500.jpg"))
    print(f"rec_res: ", rec_res)

    print(f"Wordlevel-------")

    ## custom
    paddleocr_engine_wordlevel = PaddlePytorchOCR(
        det_model_path=os.path.join(pdtorch_model_dir, "en_ptocr_v3_det_infer.pth"),
        det_yaml_path=os.path.join(__dir__, "configs/det/det_ppocr_v3.yml"),
        
        use_angle_cls=True,
        cls_model_path=os.path.join(pdtorch_model_dir, "ch_ptocr_mobile_v2.0_cls_infer.pth"),
        cls_yaml_path=os.path.join(__dir__, "configs/cls/cls_mv3.yml"),

        rec_model_path=os.path.join(pdtorch_model_dir, "model_rec_custom_fs_v7.pth"),
        rec_yaml_path=os.path.join(__dir__, "configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml"),
        rec_char_dict_path=os.path.join(__dir__, "pytorchocr/utils/ppocr_keys_v1.txt"),
    )
    rec_res = paddleocr_engine_wordlevel.ocr(os.path.join(__dir__, "../images/pan-card-500x500.jpg"))
    print(f"rec_res: ", rec_res)

    # import pdb;pdb.set_trace()
    pass

if __name__ == "__main__":
    main()
