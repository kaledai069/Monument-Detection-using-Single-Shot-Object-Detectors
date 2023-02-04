import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf
from absl import app

flags = app.flags
flags.DEFINE_string('image_dir', '', 'Path to image directory.')
flags.DEFINE_string('annotations_dir', '', 'Path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def dict_to_tf_example(data, image_dir, label_map_dict):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding
    box coordinates provided by the raw data.

    Arguments:
        data: dict holding XML fields for a single image (obtained by
          running dataset_util.recursive_parse_xml_to_dict)
        image_dir: Path to image directory.
        label_map_dict: A map from string label names to integers ids.

    Returns:
        example: The converted tf.Example.
    """
    full_path = os.path.join(image_dir, data['filename'])
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    try:
        for obj in data['object']:
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
    except KeyError:
        print(data['filename'] + ' without objects!')

    difficult_obj = [0]*len(classes)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
        'image/object/difficult': int64_list_feature(difficult_obj)
    }))
    return example

def recursive_parse_xml_to_dict(xml):
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def main(_):

    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = {
        'basantapur tower': 1,
        'bhimeleshvara': 2,
        'gaddi baithak' : 3,
        'garud' : 4, 
        'kasthamandap' : 5,
        'lalitpur tower': 6,
        'trailokya mohan':7
    }

    image_dir = FLAGS.image_dir
    annotations_dir = FLAGS.annotations_dir
    logging.info('Reading from dataset: ' + annotations_dir)
    examples_list = os.listdir(annotations_dir)

    for idx, example in enumerate(examples_list):
        if example.endswith('.xml'):
            if idx % 50 == 0:
                print('On image %d of %d' % (idx, len(examples_list)))

            path = os.path.join(annotations_dir, example)
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data, image_dir, label_map_dict)
            writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    app.run(main)