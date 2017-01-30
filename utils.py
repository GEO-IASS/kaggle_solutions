
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from scipy.misc import imread, imresize
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import xml.etree.ElementTree as ET

def full_report(model, directory, generator=ImageDataGenerator(), batch_size=8, n=5):

    batches = generator.flow_from_directory(directory, model.input_shape[1:3],
                                            shuffle=False,
                                            batch_size=batch_size)
    filenames = batches.filenames
    print("Generating predictions...")
    our_predictions = model.predict_generator(batches, batches.nb_sample)
    our_labels = our_predictions.argmax(1)
    print()
    print("PRECISION / RECALL / F1-Score / SUPPORT")
    print(classification_report(y_pred.argmax(1), batches.classes))
    return
    for c in batches.class_indices:
        class_idx = batches.class_indices[c]
        class_predictions = our_predictions[:,class_idx]
        false_positives = np.where((our_labels == class_idx) & (true_labels != class_idx))[0]
        top_n_false_positives = np.argsort(class_predictions[false_positives])[::-1][:n]
        idx_to_plot = false_positives[top_n_false_positives]
        for img_idx in idx_to_plot:
            plt.imshow(imread(filenames[img_idx]))

def plot_bounding_box(img, coords):
    H = img.shape[0]
    W = img.shape[1]
    xmin = coords[0] * W
    ymin = coords[1] * H
    xmax = coords[2] * W
    ymax = coords[3] * H

    fig,ax = plt.subplots(1)
    ax.imshow(img)

    rec_width = xmax - xmin
    rec_height = ymax - ymin
    rect = Rectangle((xmin,ymin), rec_width, rec_height,
                             linewidth=1, edgecolor='r', fill=False)
    ax.add_patch(rect)
    plt.show()

def compare_bounding_box(img, bbox1, bbox2, image_size=(200,350)):
    H = img.shape[0]
    W = img.shape[1]

    fig,ax = plt.subplots(1)
    ax.imshow(img)

    xmin1 = bbox1[0] * W
    ymin1 = bbox1[1] * H
    xmax1 = bbox1[2] * W
    ymax1 = bbox1[3] * H
    rec_width1 = xmax1 - xmin1
    rec_height1 = ymax1 - ymin1
    
    rect1 = Rectangle((xmin1,ymin1), rec_width1, rec_height1,
                             linewidth=3, edgecolor='g', fill=False)
    ax.add_patch(rect1)
    
    xmin2 = bbox2[0] * W
    ymin2 = bbox2[1] * H
    xmax2 = bbox2[2] * W
    ymax2 = bbox2[3] * H
    rec_width2 = xmax2 - xmin2
    rec_height2 = ymax2 - ymin2
    rect2 = Rectangle((xmin2,ymin2), rec_width2, rec_height2,
                             linewidth=1, edgecolor='r', fill=False)
    ax.add_patch(rect2)

    plt.show()

def parse_pasval_voc_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    img_path = root.findtext("path")
    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))
    xmin = int(root.findtext("./object/bndbox/xmin")) / width
    ymin = int(root.findtext("./object/bndbox/ymin")) / height
    xmax = int(root.findtext("./object/bndbox/xmax")) / width
    ymax = int(root.findtext("./object/bndbox/ymax")) / height
    bbox = np.array([xmin, ymin, xmax, ymax])
    return img_path, bbox, width, height