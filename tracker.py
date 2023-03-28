from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import math
import numpy as np

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(
    cfg.DEEPSORT.REID_CKPT,
    max_dist=cfg.DEEPSORT.MAX_DIST,
    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg.DEEPSORT.MAX_AGE,
    n_init=cfg.DEEPSORT.N_INIT,
    nn_budget=cfg.DEEPSORT.NN_BUDGET,
    use_cuda=True,
)


def plot_bboxes_original(image, bboxes, line_thickness=None):

    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    )  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ["person"]:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        # cls_id can be only person or suitcase
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # facenet: Please add facenet id to this text label.
        cv2.putText(
            image,
            "{}".format(cls_id),
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    return image


def plot_bboxes(
    image,
    bboxes,
    line_thickness=None,
    target_detector=None,
    track_id=None,
    trackitem_id=None,
):

    tl = (
        line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    )  # line/font thickness

    def check_lost(image, person, suitcase, target_detector):
        # definition_of_lost = either vertical or horizontal distance between the detected person and suitcase > a constant
        # image has person, suitcase, and 
        # target_detector.personAndSuitcaseLostCounter > 30
        if person is None or suitcase is None:
            target_detector.personAndSuitcaseLostCounter = 0
            return image
        (xp, yp) = person
        (xs, ys) = suitcase
        distance = 220
        # distance be adjusted acd to video parameters
        x_sqr = abs(xp - xs) * abs(xp - xs)
        y_sqr = abs(yp - ys) * abs(yp - ys)
        print(f"distance {math.sqrt(x_sqr + y_sqr)}")
        if x_sqr + y_sqr >= distance * distance:
            target_detector.personAndSuitcaseLostCounter += 1
        else:
            target_detector.personAndSuitcaseLostCounter = 0
        lost = False
        if target_detector.personAndSuitcaseLostCounter > 20:
            print('LOST!!!')
            lost = True
            color= (0, 0, 255)
            x_lost_1 = int(xs - 20)
            y_lost_1 = int(ys + 20)
            x_lost_2 = int(xs + 20)
            y_lost_2 = int(ys - 20)
            # 20 be the number adjuested acd to model
            c1, c2 = (x_lost_1, y_lost_1), (x_lost_2, y_lost_2)
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            cv2.putText(
                image,
                "LOST",
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tl,
                lineType=cv2.LINE_AA,
            )
        if lost is True:
            import pytest
            target_detector.isLost=True
            # 在这里保存图片 并退出?
            # pytest.set_trace()
        return image

    # without face recognition, can only handle scenarios where there is only
    # 1 person and 1 suit case,
    # mark suite case as lost when distance is larger than 0.5m.

    def render_person(image, bbox, target=False):
        (x1, y1, x2, y2, cls_id, pos_id) = bbox
        if target:
            color = (0, 255, 0)
        else:
            color = (220,220,220)
        text = "{}-{}".format(cls_id, pos_id)
        if target:
            text = text + "TARGET"
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            text,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
        x_gravity_center = (x1 + x2) / 2
        y_gravity_center = (y1 + y2) / 2
        return image, (x_gravity_center, y_gravity_center)

    def render_suitcase(image, bbox, target=False):
        (x1, y1, x2, y2, cls_id, pos_id) = bbox
        if target:
            color = (0, 0, 255)
        else:
            color = (255,255,0)
        text = "{}-{}".format(cls_id, pos_id)
        if target:
            text = text + "ITEM"
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            text,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
        x_gravity_center = (x1 + x2) / 2
        y_gravity_center = (y1 + y2) / 2
        return image, (x_gravity_center, y_gravity_center)

    person_gravity_center = None
    suitcase_gravity_center = None
    for bbox in bboxes:
        # cls_id can be only person or suitcase
        (x1, y1, x2, y2, cls_id, pos_id) = bbox
        if cls_id == "person" and pos_id == track_id:
            image, person_gravity_center = render_person(image, bbox, target=True)
        elif cls_id == "person" and pos_id != track_id:
            image, _ = render_person(image, bbox, target=False)
        elif cls_id == "backpack":
     #   elif cls_id == "backpack" and pos_id == trackitem_id:
     #       image, suitcase_gravity_center = render_suitcase(image, bbox, target=True)
     #   elif cls_id == "backpack" and pos_id != trackitem_id:
            image, suitcase_gravity_center = render_suitcase(image, bbox, target=False)
    image = check_lost(image, person_gravity_center, suitcase_gravity_center, target_detector)
    # facenet: Please add facenet id to this text label.

    return image


def update_tracker(target_detector, image, draw=True, target_track_id=None):
    if draw:
        new_faces = []
        _, bboxes = target_detector.detect(image)
        # / print("detect result", bboxes)
        bbox_xywh = []
        confs = []
        clss = []

        for x1, y1, x2, y2, cls_id, conf in bboxes:
            obj = [int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1]
            bbox_xywh.append(obj)
            confs.append(conf)
            clss.append(cls_id)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, clss, image)
        bboxes2draw = []
        face_bboxes = []
        current_ids = []
        for value in list(outputs):
            x1, y1, x2, y2, cls_, track_id = value
            bboxes2draw.append((x1, y1, x2, y2, cls_, track_id))
            current_ids.append(track_id)
            if cls_ == "face":
                if not track_id in target_detector.faceTracker:
                    target_detector.faceTracker[track_id] = 0
                    face = image[y1:y2, x1:x2]
                    new_faces.append((face, track_id))
                face_bboxes.append((x1, y1, x2, y2))

        ids2delete = []
        for history_id in target_detector.faceTracker:
            if not history_id in current_ids:
                target_detector.faceTracker[history_id] -= 1
            if target_detector.faceTracker[history_id] < -5:
                ids2delete.append(history_id)
        for ids in ids2delete:
            target_detector.faceTracker.pop(ids)
            print("-[INFO] Delete track id:", ids)

        image = plot_bboxes(
            image, bboxes2draw, line_thickness=None, target_detector=target_detector, track_id=target_track_id
        )

        return image, current_ids, face_bboxes
    else:
        # _, bboxes = target_detector.detect(image)
        pass


def list_txt(path, list=None):
    '''
    :param path: 储存list的位置
    :param list: list数据
    :return: None/re将txt读取为list
             当path参数和list都有输入时为保存模式将list保存为txt
    '''
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

def _pdist(a, b, single_embedding):
    new = np.asarray(a)
    known = np.asarray(b)
    if len(new) == 0 or len(known) == 0:
        return np.zeros((len(new), len(known)))
    new2, known2 = np.square(new).sum(axis=1), np.square(known).sum(axis=1)
    r2 = -2. * np.dot(new, known.T) + new2[:, None] + known2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

def _nn_euclidean_distance(x, y, single_embedding):
    distances = _pdist(x, y, single_embedding)
    return np.maximum(0.0, distances.min(axis=0))
