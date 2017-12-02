import os
import sys
import random
import numpy as np
import tensorflow as tf

srcPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(srcPath)
from helper.ImageHelper import ImageHelper
from PIL import Image


class ModelBase():
    _modelName = 'base'
    _saver = None

    def __init__(self, modelDir=None, sess=None):
        self._modelDir = modelDir + "/model"
        self._modelFile = self._modelDir + "/model"
        self._modelMeta = self._modelDir + "/model.meta"
        self._summaryDir = modelDir + "/summary"
        self._inputDir = modelDir + "/input"
        self._inputPath = self._inputDir + "/record"
        self._width = 80
        self._height = 80
        self._in_channels = 3
        self._batch_size = 100
        self._sess = sess
        if os.path.exists(modelDir) == False:
            os.mkdir(modelDir)
        if os.path.exists(self._modelDir) == False:
            os.mkdir(self._modelDir)
        if os.path.exists(self._summaryDir) == False:
            os.mkdir(self._summaryDir)
        if os.path.exists(self._inputDir) == False:
            os.mkdir(self._inputDir)
        if sess == None:
            self._sess = tf.Session()

    def BuildData(self, markPath, sourcePath):
        markImg = Image.open(markPath)
        [markWidth, markHeight] = markImg.size
        outPath = self._inputDir + "/example"
        if os.path.exists(outPath) == False:
            os.mkdir(outPath)
        tfWriter = tf.python_io.TFRecordWriter(self._inputPath)
        dWidth = int(self._width / 4)
        dHeight = int(self._height / 4)
        count = 0
        for imgName in os.listdir(sourcePath):
            imgPath = sourcePath + "/" + imgName
            img = Image.open(imgPath)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            [iWidth, iHeight] = img.size
            iHeight = int(400 * iHeight / iWidth)
            iWidth = 400
            img = img.resize((400, iHeight), Image.BICUBIC)
            wNum = int(round((iWidth - self._width) / dWidth))
            hNum = int(round((iHeight - self._height) / dHeight))
            for x in range(wNum):
                for y in range(hNum):
                    regin = (x * dWidth, y * dHeight, x * dWidth +
                             self._width, y * dHeight + self._height)
                    tmpImg = img.crop(regin)
                    tmpImgW = tmpImg.crop((0, 0, self._width, self._height))
                    tmpImgW = self._addWater(tmpImgW, markImg)
                    count = count + 1

                    if count % 99 == 1:
                        tmpImg.save(outPath + "/" + str(count) + ".png")
                        tmpImgW.save(outPath + "/" + str(count) + "_w.png")

                    if self._in_channels == 1:
                        tmpImg = tmpImg.convert("L")
                        tmpImgW = tmpImgW.convert("L")

                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[-1])),
                        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmpImg.tobytes()]))
                    }))
                    tfWriter.write(example.SerializeToString())

                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmpImgW.tobytes()]))
                    }))
                    tfWriter.write(example.SerializeToString())

        print("Create %d images" % count)
        tfWriter.close()

    def saveModel(self):
        if self._saver == None:
            self._saver = tf.train.Saver()
        self._saver.save(self._sess, self._modelFile)

    def restoreModel(self):
        print("Recover model from %s" % self._modelDir)
        self._saver = tf.train.import_meta_graph(self._modelMeta)
        self._saver.restore(self._sess, self._modelFile)

    def _conv_layer(self, x, width, height,  in_channel, out_channel, name):
        with tf.name_scope(name):
            w, b = self._get_conv_var(
                width, height, in_channel, out_channel, name)
            conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, b)
            relu = tf.nn.relu(bias)
            return relu

    def _get_conv_var(self, width, height, in_channel, out_channel, name):
        initial = tf.truncated_normal(
            [width, height, in_channel, out_channel], 0.0, 0.01)
        w = tf.Variable(initial, name=name + "_w")

        initial = tf.truncated_normal([out_channel], 0.0, 0.01)
        b = tf.Variable(initial, name=name + "_b")
        return w, b

    def _max_pool(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _fc_layer(self, x, in_size, out_size, name):
        with tf.name_scope(name):
            w, b = self._get_fc_var(in_size, out_size, name)
            x = tf.reshape(x, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, w), b)
            return fc

    def _get_fc_var(self, in_size, out_size, name):
        initial = tf.truncated_normal([in_size, out_size], 0.0, 0.01)
        w = tf.Variable(initial, name + "_w")

        initial = tf.truncated_normal([out_size], 0.0, 0.01)
        b = tf.Variable(initial, name + "_b")
        return w, b

    def _init_data_reader(self):
        queue = tf.train.string_input_producer([self._inputPath])

        reader = tf.TFRecordReader()
        _, serialize = reader.read(queue)

        features = tf.parse_single_example(serialize, features={
            'label': tf.FixedLenFeature([1], tf.int64),
            'img': tf.FixedLenFeature([], tf.string),
        })

        img = tf.decode_raw(features['img'], tf.uint8)
        img = tf.reshape(img, [self._width, self._height, self._in_channels])
        img = tf.cast(img, tf.float32)

        label = features['label']
        label = tf.cast(label, tf.float32)
        self._img, self._label = tf.train.shuffle_batch(
            [img, label], batch_size=self._batch_size, capacity=2000, min_after_dequeue=500)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self._sess, coord=coord)

    def _next_batch(self):
        img, label = self._sess.run([self._img, self._label])
        return img, label

    def _addWater(self, tmpImg, markImg):
        percent = random.randint(90, 110)

        [markWidth, markHeight] = markImg.size
        width = int(markWidth * percent / 100)
        height = int(markHeight * percent / 100)

        x1 = random.randint(0, self._width - width - 1)
        y1 = random.randint(0, self._height - height - 1)

        x2 = x1 + width
        y2 = y1 + height

        return ImageHelper.AddWaterWithImg(tmpImg, markImg, x1, y1, x2, y2)
