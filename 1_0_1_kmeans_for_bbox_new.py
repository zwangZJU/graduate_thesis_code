# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/9/12
@Desc  : 
'''
import cv2
class KMeans():
    def __init__(self, n_clusters, boxes, data_size, max_epoch=1000, cls='', file_name='result'):
        '''

        :param n_clusters: 聚类中心的个数
        :param label_file: 标签文件的存储路径，txt文件，里面存储着所有label文件的路径
        :param max_epoch:  k-means算法的最大迭代次数
        :param cls: 类别，str格式，不为空时分类别聚类，'0'代表person,'1'代表face；''代表两类都加载，一起聚类，默认为空
        :param file_name: 存储的文件名
        '''
        self.n_clusters = n_clusters
        self.file_name = file_name
        self.cls = cls
        # if cls:
        #     self.boxes, self.data_size = self.dataloader_by_class_new(label_file, cls=cls)
        # else:
        #     self.boxes, self.data_size= self.dataloader_new(label_file)
        self.boxes = boxes
        self.data_size = data_size
        #self.boxes, self.data_size = self.boxes[:4], 4

        self.random_init_centroids(5)
        self.max_epoch = max_epoch
        self.epsilon = 1e-8

    def dataloader(self, path, size=416):
        '''
        将label的宽高值读入，path中存储着所有label的路径，每个label中的每一行是calss x, y, w, h
        为了减少内存的使用，只载入w和h，并将比例换算成实际长度
        :param path: 保存所有label路径的txt文件
        :return: List[List]: 包含所有[w,h]的list
        '''
        print("[*]Start loading data!")
        all_wh = []
        data_size = 0
        with open(path,'r') as flist:
            for line in flist.readlines():
                with open(line.strip(),'r') as f:
                    for info in f.readlines():
                        label = 0

                        wh = list(map(lambda x: int(float(x)*size), info.split()[-2:]))

                        all_wh.append([label, wh])
                        data_size += 1
        print("[*]Load data successfully!")
        print("There are {} boxes".format(data_size))
        return all_wh, data_size

    def dataloader_new(self, path, size=416):
        '''
        将label的宽高值读入，path中存储着所有label的路径，每个label中的每一行是calss x, y, w, h
        为了减少内存的使用，只载入w和h，并将比例换算成实际长度
        :param path: 保存所有label路径的txt文件
        :return: List[List]: 包含所有[w,h]的list
        '''
        print("[*]Start loading data!")
        all_wh = []
        data_size = 0
        with open(path,'r') as flist:
            for line in flist.readlines():
                img_path = line.replace('labels', 'images').replace('.txt', '.jpg').replace('_l', '').strip()
                img = cv2.imread(img_path)
                w0, h0, c = img.shape

                with open(line.strip(), 'r') as f:
                    for info in f.readlines():

                        label = 0
                        w, h = info.split()[-2:]
                        if w0 > h0:
                            wh = [int(float(w) * 416), int(float(h) * h0 * 416 / w0)]
                        else:
                            wh = [int(float(w) * w0 * 416 / h0), int(float(h) * 416)]

                        all_wh.append([label, wh])
                        data_size += 1
        print("[*]Load data successfully!")
        print("There are {} boxes".format(data_size))
        return all_wh, data_size


    def dataloader_by_class_new(self, path, size=416, cls='0'):
        '''
        将label的宽高值读入，path中存储着所有label的路径，每个label中的每一行是calss x, y, w, h
        为了减少内存的使用，只载入w和h，并将比例换算成实际长度
        :param path: 保存所有label路径的txt文件
        :return: List[List]: 包含所有[w,h]的list
        '''
        print("[*]Start loading data!")
        all_wh = []
        data_size = 0
        with open(path, 'r') as flist:
            for line in flist.readlines():
                img_path = line.replace('labels', 'images').replace('.txt', '.jpg').replace('_l', '').strip()

                img = cv2.imread(img_path)

                w0, h0, c = img.shape

                with open(line.strip(), 'r') as f:
                    for info in f.readlines():
                        if info.split()[0] != cls:
                            continue
                        label = 0
                        w, h = info.split()[-2:]
                        if w0 > h0:
                            wh = [int(float(w) * 416), int(float(h) * h0 * 416 / w0)]
                        else:
                            wh = [int(float(w) * w0 * 416 / h0), int(float(h) * 416)]
                        all_wh.append([label, wh])
                        data_size += 1
        print("[*]Load data successfully!")
        print("There are {} boxes".format(data_size))
        return all_wh, data_size
    def dataloader_by_class(self, path, size=416, cls='0'):
        '''
        将label的宽高值读入，path中存储着所有label的路径，每个label中的每一行是calss x, y, w, h
        为了减少内存的使用，只载入w和h，并将比例换算成实际长度
        :param path: 保存所有label路径的txt文件
        :return: List[List]: 包含所有[w,h]的list
        '''
        print("[*]Start loading data!")
        all_wh = []
        data_size = 0
        with open(path, 'r') as flist:
            for line in flist.readlines():
                with open(line.strip(), 'r') as f:
                    for info in f.readlines():
                        if info.split()[0] != cls:
                            continue
                        label = 0

                        wh = list(map(lambda x: int(float(x) * size), info.split()[-2:]))

                        all_wh.append([label, wh])
                        data_size += 1
        print("[*]Load data successfully!")
        print("There are {} boxes".format(data_size))
        return all_wh, data_size

    def iou(self, bbox1, bbox2):
        '''
        输入两个bounding box的值，计算iou，以iou作为距离度量
        :param bbox1: list: [width, height]
        :param bbox2: list: [width, height]
        :return: iou: 比例
        '''
        w1, h1 = bbox1[0], bbox1[1]
        w2, h2 = bbox2[0], bbox2[1]
        overlap = min(w1, w2) * min(h1, h2)
        union = w1 * h1 + w2 * h2 - overlap

        return overlap/(union+self.epsilon)

    def kmeans(self, init_center=None, square=False):
        if square:
            centroids = init_center if init_center else self.random_init_square_centroids(self.n_clusters)
        else:
            centroids = init_center if init_center else self.random_init_centroids(self.n_clusters)

        epoch = 0
        is_changed = True
        while is_changed and epoch<self.max_epoch:
            epoch += 1
            is_changed = False
            # 划分类别
            for i in range(self.data_size):
                label, wh = self.boxes[i]
                # max_iou = 0
                # 计算到k个中心框的iou
                ious = [self.iou(c, wh) for c in centroids]

                max_iou = max(ious)
                # 得到所属类别，即iou最大的那一类
                index = ious.index(max_iou)
                # 当类别划分较上一轮有所改变时，更新类别标签
                if label != index:
                    is_changed = True
                    self.boxes[i] = [index, wh]

            # 重新计算中心点
            centroids = self.computer_centroids(self.boxes)
            print(epoch, centroids)
        #当类别不再改变的时候
        total_distance = self.computer_total_distance(centroids, self.boxes)
        with open('results/'+ self.file_name +self.cls+'.txt','a') as f:
            f.write(str(self.n_clusters)+ ' '+ ''.join([str(int(w))+','+str(int(h))+','+' ' for w, h in centroids])+ ' ' + str(total_distance)+'\n')
        return centroids

    def computer_centroids(self, boxes):
        '''
        根据带标签的boxes,找到k个类的新的中心点
        :param boxes: 所有的box，格式为[label,[w,h]]
        :return: k个类的中心点
        '''
        counters = [0] * self.n_clusters
        centroids = [[0,0] for _ in range(self.n_clusters)]

        for box in boxes:
            label, wh = box
            counters[label] += 1
            centroids[label][0] += wh[0]
            centroids[label][1] += wh[1]
        for i in range(self.n_clusters):
            centroids[i][0] //= counters[i] + self.epsilon
            centroids[i][1] //= counters[i] + self.epsilon
        return centroids

    def computer_total_distance(self, centroids, bboxes):
        '''
        计算所有bbox到达各自中心的总距离
        :param centroids: k个聚类中心
        :param bboxes: 所有的边框，格式为[类别，[宽， 高]]
        :return: 总距离
        '''
        total_distance = 0
        for label, wh in bboxes:
            total_distance += 1 - self.iou(centroids[label], wh)
        return total_distance

    def random_init_centroids(self, k):
        '''
        初始化k个聚类中心，分别找出w和h的最大值和最小值，在值域范围内均匀取得k个值，组合成w,h对
        :param k: 聚类中心的个数
        :return: k个中心
        '''
        min_w, max_w, min_h, max_h =  float('INF'), 0, float('INF'), 0
        for l, wh in self.boxes:
            if wh[0] < min_w:
                min_w = wh[0]
            elif wh[0] > max_w:
                max_w = wh[0]

            if wh[1] < min_h:
                min_h = wh[1]
            elif wh[1] > max_h:
                max_h = wh[1]
        dw = int(max_w - min_w) // k
        dh = int(max_h - min_h) // k
        w = list(range(min_w+dw//2,max_w, dw))
        h = list(range(min_h+dh//2,max_h, dh))
        centroids = [[w[i], h[i]] for i in range(k)]

        return centroids

    def random_init_square_centroids(self, k):
        '''
        初始化k个聚类中心，找出w的最大值和最小值，在值域范围内均匀取得k个值，作为正方形的bbox
        :param k: 聚类中心的个数
        :return: k个中心
        '''
        min_w, max_w, min_h, max_h =  float('INF'), 0, float('INF'), 0
        for l, wh in self.boxes:
            if wh[0] < min_w:
                min_w = wh[0]
            elif wh[0] > max_w:
                max_w = wh[0]

        dw = int(max_w - min_w) // k

        w = list(range(min_w+dw//2,max_w, dw))

        centroids = [[w[i], w[i]] for i in range(k)]
        print(centroids)

        return centroids


    def get_silhouette_coefficient(self, centroids, boxes):
        n_in, n_out = 0, 0
        n = len(centroids)
        boxes_in_class = [None]*n
        for label, wh in boxes:
            boxes_in_class[label].append(wh)
        avg_s = s = 0
        for i in range(n):
            for w, h in boxes_in_class[i]:
                d1 = d2 = 0
                a = b = 1
                # 求类内距离
                for w1, h1 in boxes_in_class[i]:
                    d1 += 1 - self.iou([w, h], [w1, h1])
                a = d1 / len(boxes_in_class[i])
                # 求最近的其他中心
                nearest_d, nearest_j = float('INF'), 0
                for j in range(n):
                    if j != i:
                        if nearest_d >( 1 - self.iou([w, h], centroids[j])):
                            nearest_j = j
                # 求到最近其他类的距离
                for w1, h1 in boxes_in_class[nearest_j]:
                    d2 += 1 - self.iou([w, h], [w1, h1])
                b = d2 / boxes_in_class[nearest_j]
                s += (b-a)/max(b, a)
        avg_s = s / len(boxes)
            # 计算最近的其他中心







def test_stop():
    k = 2
    label_txt = 'E:/Datasets/CrowdHuman/train/labels_train.txt'
    init_centroids = [[23, 27], [100, 400]]
    kmeans = KMeans(k, label_txt)
    #result = kmeans.kmeans()
    a = 1 - kmeans.iou([0, 0], [0, 1])
    b = 1 - kmeans.iou([23, 45], [24, 41])
    c = 1 - kmeans.iou([56, 296], [62, 296])
    d = 1 - kmeans.iou([69, 296], [62, 296])
    print(a + b + c + d)

def dataloader_new(path, size=416):
    '''
    将label的宽高值读入，path中存储着所有label的路径，每个label中的每一行是calss x, y, w, h
    为了减少内存的使用，只载入w和h，并将比例换算成实际长度
    :param path: 保存所有label路径的txt文件
    :return: List[List]: 包含所有[w,h]的list
    '''
    print("[*]Start loading data!")
    all_wh = []
    data_size = 0
    with open(path,'r') as flist:
        for line in flist.readlines():
            img_path = line.replace('labels', 'images').replace('.txt', '.jpg').replace('_l', '').strip()
            img = cv2.imread(img_path)
            h0, w0, c = img.shape

            with open(line.strip(), 'r') as f:
                for info in f.readlines():

                    label = 0
                    w, h = info.split()[-2:]
                    if w0 > h0:
                        wh = [int(float(w) * 416), int(float(h) * h0 * 416 / w0)]
                    else:
                        wh = [int(float(w) * w0 * 416 / h0), int(float(h) * 416)]

                    all_wh.append([label, wh])
                    data_size += 1
    print("[*]Load data successfully!")
    print("There are {} boxes".format(data_size))
    return all_wh, data_size


def dataloader_by_class_new( path, size=416, cls='0'):
    '''
    将label的宽高值读入，path中存储着所有label的路径，每个label中的每一行是calss x, y, w, h
    为了减少内存的使用，只载入w和h，并将比例换算成实际长度
    :param path: 保存所有label路径的txt文件
    :return: List[List]: 包含所有[w,h]的list
    '''
    print("[*]Start loading data!")
    all_wh = []
    data_size = 0
    with open(path, 'r') as flist:
        for line in flist.readlines():
            img_path = line.replace('labels', 'images').replace('.txt', '.jpg').replace('_l', '').strip()

            img = cv2.imread(img_path)

            h0, w0, c = img.shape

            with open(line.strip(), 'r') as f:
                for info in f.readlines():
                    if info.split()[0] != cls:
                        continue
                    label = 0
                    w, h = info.split()[-2:]
                    if w0 > h0:
                        wh = [int(float(w) * 416), int(float(h) * h0 * 416 / w0)]
                    else:
                        wh = [int(float(w) * w0 * 416 / h0), int(float(h) * 416)]
                    all_wh.append([label, wh])
                    data_size += 1
    print("[*]Load data successfully!")
    print("There are {} boxes".format(data_size))
    return all_wh, data_size

if __name__ == '__main__':
    # image = cv2.imread('E:/Datasets/CrowdHuman/train/images/284193,faa9000f2678b5e.jpg')
    # print(1, image)
    label_txt = 'E:/Datasets/CrowdHuman/train_l/labels_train_new.txt'
    #init_centroids = [[23,27], [46,54], [55, 260], [69, 81], [92, 130], [135, 350]]
    # for k in range(3, 40, 1):
    #
    #     kmeans = KMeans(k, label_txt, cls='0')
    #     break
        #result = kmeans.kmeans()

    # for k in range(7, 13, 1):
    #
    #     kmeans = KMeans(k, label_txt, cls='', file_name='large_box_by_k_means')
    #
    #     result = kmeans.kmeans(square=True)

    box, data_size = dataloader_by_class_new(label_txt,cls='1')
    # large face
    for k in range(1, 17, 1):
        kmeans = KMeans(k, box, data_size, cls='1', file_name='large_box_by_k_means_face_new')

        result = kmeans.kmeans(square=True)
    box, data_size = dataloader_by_class_new(label_txt, cls='0')
    # large person
    for k in range(1, 17, 1):
        kmeans = KMeans(k, box, data_size, cls='0', file_name='large_box_by_k_means_person_new')
        result = kmeans.kmeans(square=True)

    box, data_size = dataloader_new(label_txt)
    # both
    for k in range(1, 17, 1):
        kmeans = KMeans(k, box, data_size,  cls='', file_name='large_box_by_k_means_both_new')
        result = kmeans.kmeans(square=True)
    # for k in range(1, 6, 1):
    #
    #     kmeans = KMeans(k, label_txt, cls='0')
    #
    #     result = kmeans.kmeans(square=True)
    # for k in range(1, 6, 1):
    #
    #     kmeans = KMeans(k, label_txt, cls='1')
    #
    #     result = kmeans.kmeans(square=True)

    # kmeans = KMeans(17, label_txt)
    #
    # result = kmeans.kmeans(square=True)
    #test_stop()