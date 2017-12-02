import os
import sys
import random
import numpy as np
import tensorflow as tf

srcPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(srcPath)
from model.ModelBase import ModelBase
from helper.ImageHelper import ImageHelper
from PIL import Image

class AutoEncoder(ModelBase):
    _modelName = 'auto_encoder'

    def BuildModel(self):
        """Build identify water mark model with vgg
        """
        print("Build new model")
        x = tf.placeholder(tf.float32, [None, self._width, self._height, self._in_channels])
        y_ = tf.placeholder(tf.float32, [None, self._width, self._height, self._in_channels])
        tf.add_to_collection('x', x)
        tf.add_to_collection('y_', y_)

        conv1 = self._conv_layer(x, 5, 5, 3, 16, 'conv1')
        conv2 = self._conv_layer(conv1, 5, 5, 16, 32, 'conv2')
        conv3 = self._conv_layer(conv2, 5, 5, 32, 3, 'conv3')

        error = (conv3 - y_) ** 2
        loss = tf.reduce_mean(error, name='loss')
        train = tf.train.AdamOptimizer(1e-3).minimize(loss)
        tf.add_to_collection('error', error)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('train', train)

        # summary data
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', loss)
            # tf.summary.scalar('error', error)
            tf.summary.image('x', x, 5)

        self._summary_merge = tf.summary.merge_all()
        self._summary_writer = tf.summary.FileWriter(self._summaryDir, self._sess.graph)

        return x, y_, train, error, loss

    def Train(self, loop_count):
        if self._inputDir is None:
            raise Exception("Input path can't be empty under train model")

        # if self._modelDir is None or os.path.exists(self._modelDir) == False:
        #     x, y_, train, error, loss = self.BuildModel()
        #     self._sess.run(tf.global_variables_initializer())
        # else:
        #     self.restoreModel()
        #     train = tf.get_collection('train')[0]
        #     loss = tf.get_collection('loss')[0]
        #     x = tf.get_collection('x')[0]
        #     y_ = tf.get_collection('y_')[0]
        #     error = tf.get_collection('error')[0]

        x, y_, train, error, loss = self.BuildModel()
        self._sess.run(tf.global_variables_initializer())

        self._init_data_reader()
        for i in range(loop_count):
            img, _y = self._next_batch()

            if i % 50 == 0:
                current_loss, summary = self._sess.run(
                    [loss, self._summary_merge], feed_dict={x: img, y_: img})
                self.saveModel()
                self._summary_writer.add_summary(summary, i)
                print('step %d, training loss %g' % (i, current_loss))
            else:
                self._sess.run([train], feed_dict={x: img, y_: img})

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
                    tmpImg = self._addWater(tmpImg, markImg)
                    count = count + 1

                    if count % 99 == 1:
                        tmpImg.save(outPath + "/" + str(count) + ".png")

                    if self._in_channels == 1:
                        tmpImg = tmpImg.convert("L")

                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmpImg.tobytes()]))
                    }))
                    tfWriter.write(example.SerializeToString())

        print("Create %d images" % count)
        tfWriter.close()
