import json
import os
from collections import OrderedDict
from os.path import join as ospjoin
from typing import Tuple, Dict, List

import cv2
import numpy as np
import sklearn.preprocessing as preprocess
import torch
from numpy.typing import NDArray
from skimage import transform as trans

from recog.backbones import get_model


class Recognition:
    def __init__(self, args, device=0):
        self.args = args
        if os.path.exists(ospjoin('api', args.feature_path + 'backup.npz')):
            backup = np.load(ospjoin('api', args.feature_path + 'backup.npz'))
            self.cls_feature = backup.get("feature")
            self.unique_label = backup.get("unique_templates")
            self.template2id = backup.get("template2id")
            if 'labels' in list(backup.keys()):
                self.labeling = 1
                self.label = backup.get("labels")
            else:
                self.labeling = 0
        self.weight = torch.load(args.rec_checkpoint_path, map_location=lambda storage, _: storage.cuda(device))
        resnet = get_model(args.network, dropout=0, fp16=False)
        resnet.load_state_dict(self.weight)
        self.model = torch.nn.DataParallel(resnet, device_ids=[device])
        self.model.eval()

    def api_recognition(
            self, file: Dict[str, list], crop_img: Dict[str, NDArray], bbox: List[List], img: NDArray, test=0
    ) -> Tuple[List, List[Dict[int, list]], NDArray, List, Dict]:
        class_id, result, scores = list(), list(), list()
        img_feats, _ = self.api_image_feature(crop_img, file)
        img_input_feats = img_feats[:, :img_feats.shape[1] // 2]
        testfeat = preprocess.normalize(img_input_feats)
        assert len(testfeat) == len(bbox)
        for i, v in enumerate(img_feats):
            num_feat = dict()
            score = np.sum(self.cls_feature * testfeat[i], -1)
            scores.append(np.round(max(score) * 100, 1)[0])
            predict = self.unique_label[score.argmax()]
            if self.labeling:
                class_id.append(self.label[int(predict)])
            else:
                class_id.append(int(predict))
            num_feat['label'] = class_id[-1]
            num_feat['bbox'] = bbox[i]
            result.append(num_feat)
            if test == 'show_test':
                draw_result_and_mask_glass(file, img, bbox, class_id, i)

        return class_id, result, img, scores, crop_img

    def newclass_feature(self, file: List[dict], crop_imgs: list, labels: List[int]):
        img_feats, faceness_scores = self.api_image_feature(crop_imgs, file)
        img_input_feats = img_feats[:, :img_feats.shape[1] // 2] + img_feats[:, img_feats.shape[1] // 2:]
        img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
        template_norm_feats, unique_templates = image2template_feature(img_input_feats, labels)
        # sort label-featuremap
        template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
        for count_template, uqt in enumerate(unique_templates):
            template2id[uqt] = count_template
        feature = template_norm_feats[template2id[unique_templates]]
        return feature, unique_templates, template2id

    def load_model(self, batch_size):
        embedding = Embedding(self.model, (3, 112, 112), batch_size)
        return embedding

    def api_image_feature(self, img: Dict, files: Dict) -> Tuple[NDArray, NDArray]:
        test_num, face_scores = 0, list()
        if len(files) < 2:
            try:
                files = files['landmk_info']
                test_num = 1
            except:
                pass

        batch_size = (len(files) if len(files) else 1)
        data_shape = (3, 112, 112)
        img_feats = np.empty((batch_size, 1024), dtype=np.float32)
        batch_data = np.empty((2 * batch_size, 3, 112, 112))

        embedding = Embedding(self.model, data_shape, batch_size)
        for img_index, each_line in enumerate(files):
            name_lmk_score = each_line.strip().split(' ') if test_num else each_line['landmk_info'].strip().split(' ')
            lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
            lmk = lmk.reshape((5, 2))
            input_blob = embedding.get(img[img_index], lmk)
            batch_data[2 * img_index][:] = input_blob[0]
            batch_data[2 * img_index + 1][:] = input_blob[1]
            face_scores.append(name_lmk_score[-1])
        img_feats[0:batch_size][:] = embedding.forward_db(batch_data)
        face_scores = np.array(face_scores).astype(np.float32)
        return img_feats, face_scores

    def write_json(self, result, name, test=None):
        if not test:
            test = name
        save_path = ospjoin('api', 'test' + str(test) + '.json')
        print('write json file : ', save_path)
        imgpath_list, annotation_list = list(), list()
        if os.path.isfile(save_path):
            with open(save_path, 'r') as f:
                old_file = json.load(f)
                imgpath_list = old_file['image']
                annotation_list = old_file['annotation']
        if not isinstance(name, list):
            imgpath_list.append(f'{name}.jpg')
            annotation_list.append(result)
        save_results = OrderedDict()
        save_results['annotation'] = annotation_list
        save_results['image'] = imgpath_list
        with open(save_path, 'w', encoding="utf-8") as make_file:
            json.dump(save_results, make_file, ensure_ascii=False, indent="\t")


def image2template_feature(img_feats: NDArray = None, templates: List[int] = None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    print('total ', len(unique_templates))
    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        template_feats[count_template] = np.sum(np.array(face_norm_feats), axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))

    template_norm_feats = preprocess.normalize(template_feats)
    return template_norm_feats, unique_templates


def draw_result_and_mask_glass(file, img, bbox, class_id, i):
    land = file['landmk_info'][i].split(' ')
    # show result
    boxx = bbox[i][0]
    boxy = bbox[i][1]
    cv2.rectangle(img, (boxx, boxy), (bbox[i][2], bbox[i][3]), (0, 0, 255), 2)

    txtsize = float((bbox[i][2] - bbox[i][0]) / 200)
    cv2.putText(img, str(class_id[i]), (boxx, boxy + 12), cv2.FONT_HERSHEY_DUPLEX, txtsize, (255, 255, 255))
    cv2.circle(img, (boxx + int(land[1]), boxy + int(land[2])), 1, (0, 255, 255), 4)
    cv2.circle(img, (boxx + int(land[3]), boxy + int(land[4])), 1, (0, 255, 255), 4)
    cv2.circle(img, (boxx + int(land[5]), boxy + int(land[6])), 1, (0, 255, 255), 4)
    cv2.circle(img, (boxx + int(land[7]), boxy + int(land[8])), 1, (0, 255, 255), 4)
    cv2.circle(img, (boxx + int(land[9]), boxy + int(land[10])), 1, (0, 255, 255), 4)


def for_same_name(path: str, name: str, data_type='jpg') -> str:
    save_path, uni = ospjoin(f'{path}', f'{name}.{data_type}'), 1
    while os.path.exists(save_path):
        save_path = ospjoin(f'{path}', f'{name}_({uni}).{data_type}')
        uni += 1
    return save_path


class Embedding(object):
    # This is Embedding for verification (recognition)
    def __init__(self, model, data_shape, batch_size=1):
        image_size = (112, 112)
        self.image_size = image_size
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.model = model
        self.batch_size = batch_size
        self.data_shape = data_shape

    def get(self, rimg: NDArray, landmark: NDArray) -> NDArray:
        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]

        img = cv2.warpAffine(rimg, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data: NDArray) -> NDArray:
        device = torch.cuda.current_device()
        imgs = torch.Tensor(batch_data).to(device)
        imgs.div_(255).sub_(0.5).div_(0.5)
        # needto explain which train or eval
        feat = self.model(imgs)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()
