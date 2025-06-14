#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : det_utils.py
# @Time : 2025/6/14 14:59

import cv2
import math
import pyclipper
import numpy as np
from shapely.geometry import Polygon

def resize_image(img, short_size=736):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = short_size
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

def normalize_img(img, short_size = 736):
    img = resize_image(img, short_size = short_size)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self, params):
        self.thresh = params['thresh']
        self.box_thresh = params['box_thresh']
        self.max_candidates = params['max_candidates']
        self.is_poly = params['is_poly']
        self.unclip_ratio = params['unclip_ratio']
        self.min_size = params['min_size']

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        pred = pred
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))

            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            #             points = np.array([min(points[:,0]),min(points[:,1]),
            #                        max(points[:,0]),min(points[:,1]),
            #                        max(points[:,0]),max(points[:,1]),
            #                        min(points[:,0]),max(points[:,1])]).reshape(4,2)

            #             w,h = points[1][0] - points[0][0],points[2][1] - points[1][1]

            #             if w/h>8:
            #                 unclip_ratio = self.unclip_ratio + 0.8
            #             elif w/h>5:
            #                 unclip_ratio = self.unclip_ratio + 0.5
            #             else:
            #                 unclip_ratio = self.unclip_ratio

            #             box = self.unclip(points,unclip_ratio).reshape(-1, 1, 2)

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=2):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, pred, ratio_list):
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        score_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = pred.shape[-2:]
            if (self.is_poly):
                tmp_boxes, tmp_scores = self.polygons_from_bitmap(
                    pred[batch_index], segmentation[batch_index], width, height)

                boxes = []
                score = []
                for k in range(len(tmp_boxes)):
                    if tmp_scores[k] > self.box_thresh:
                        boxes.append(tmp_boxes[k])
                        score.append(tmp_scores[k])
                if len(boxes) > 0:
                    ratio_w, ratio_h = ratio_list[batch_index]
                    for i in range(len(boxes)):
                        boxes[i] = np.array(boxes[i])
                        boxes[i][:, 0] = boxes[i][:, 0] * ratio_w
                        boxes[i][:, 1] = boxes[i][:, 1] * ratio_h

                boxes_batch.append(boxes)
                score_batch.append(score)
            else:
                tmp_boxes, tmp_scores = self.boxes_from_bitmap(
                    pred[batch_index], segmentation[batch_index], width, height)

                boxes = []
                score = []
                for k in range(len(tmp_boxes)):
                    if tmp_scores[k] > self.box_thresh:
                        boxes.append(tmp_boxes[k])
                        score.append(tmp_scores[k])
                if len(boxes) > 0:
                    boxes = np.array(boxes)

                    ratio_w, ratio_h = ratio_list[batch_index]
                    boxes[:, :, 0] = boxes[:, :, 0] * ratio_w
                    boxes[:, :, 1] = boxes[:, :, 1] * ratio_h

                boxes_batch.append(boxes)
                score_batch.append(score)
        return boxes_batch, score_batch