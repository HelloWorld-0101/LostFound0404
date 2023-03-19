from AIDetector_pytorch import Detector
import imutils
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 保存模式
# list_txt(path='savelist.txt', list=List1)
# 读取模式
# List_rd = list_txt(path='savelist.txt')
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

def main():
    name = 'demo'
    # change suitcase / phone / others
    item_to_detect = ['person', 'suitcase']
    det = Detector(item_to_detect)
    name_list = []
    known_embedding = []
    name_list, known_embedding = det.loadIDFeats()
    #print(name_list, known_embedding)
    list_txt(path='name_list.txt', list=name_list)

    fw = open('known_embedding.txt', 'w')
    for line in known_embedding:
        for a in line:
            fw.write(str(a))
            fw.write('\t')
        fw.write('\n')
    fw.close()

    #cap = cv2.VideoCapture('videos/test1.mp4')
    #cap = cv2.VideoCapture('onlylost.mp4')
    cap = cv2.VideoCapture('IMG_1752.mp4')
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(100 / fps)
    framecounter = 0
    # 初始化帧数计数器
    frame_count = 0
    conf_index = 0
    trackingcounter = 0
    videoWriter = None
    targetLocked = False
    minIndex = None
    trackId = None
    while True:

        success, im = cap.read()
        if im is None:
            break
        # 如果帧数计数器能被3整除，则处理该帧
        if frame_count % 3 == 0:
            DetFeatures = []
            DetFeatures, img_input, box_input = det.loadDetFeats(im)
            if len(DetFeatures) > 0 and not targetLocked:
                dist_matrix = _nn_euclidean_distance(known_embedding, DetFeatures, known_embedding[0])
                minimum = np.min(dist_matrix)
                minIndex = dist_matrix.argmin()
                print('minimum:', minimum, 'dist_matrix:', dist_matrix)
                if minimum > 0.12:
                    minIndex = -1

            if (minIndex == conf_index) & (minIndex != -1):
                trackingcounter = trackingcounter + 1
                print('conf_index:',conf_index, 'minIndex:',minIndex, 'trackingcounter', trackingcounter)
            else:
                conf_index = minIndex
                trackingcounter = 0
            if trackingcounter == 5:
                trackingcounter = 0
                print('certain person：', conf_index)
                if trackId is None:
                    trackId = conf_index + 1
                det.targetTrackId = trackId
                targetLocked = True

            result = det.feedCap(im)
            result = result['frame']
            result = imutils.resize(result, height=500)
            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videoWriter = cv2.VideoWriter(
                    'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

            videoWriter.write(result)
            if det.isLost is True:
                framecounter = framecounter + 1
                print('lost')
            # 每间隔15帧输出一个失主截图
            if framecounter == 3:
                cv2.imwrite(f'./test-{det.frameCounter / fps}-second.png', result)
                #framecounter = 0

        frame_count = frame_count+1
                # todo: quit after write.
        cv2.imshow(name, result)
        cv2.waitKey(t)

        # if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
        #     # 点x退出
        #     break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()