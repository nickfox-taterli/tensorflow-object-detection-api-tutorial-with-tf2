import os
import pathlib
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)

plt.interactive(False)  # must set it to view in pycharm.ac
matplotlib.use('TkAgg')

PATH_TO_LABELS = '/home/tater/PycharmProjects/tensorflow-object-detection-api-tutorial-with-tf2/workspace/training_demo/annotations/label_map.pbtxt'
PATH_TO_MODEL_DIR = '/home/tater/PycharmProjects/tensorflow-object-detection-api-tutorial-with-tf2/workspace/training_demo/exported-models/my_model'
IMAGE_PATHS = [
    '/home/tater/PycharmProjects/tensorflow-object-detection-api-tutorial-with-tf2/workspace/training_demo/images-origin/cat_281.jpeg',
    '/home/tater/PycharmProjects/tensorflow-object-detection-api-tutorial-with-tf2/workspace/training_demo/images-origin/dog_14146.jpeg']

print('Loading model...', end='')
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR + "/saved_model")
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

for image_path in IMAGE_PATHS:
    print('Running inference for {}... '.format(image_path), end='')

    image_np = np.array(Image.open(image_path))

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
plt.show()

