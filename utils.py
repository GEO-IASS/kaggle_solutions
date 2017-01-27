
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from scipy.misc import imread
from matplotlib import pyplot as plt

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

    for c in batches.class_indices:
        class_idx = batches.class_indices[c]
        class_predictions = our_predictions[:,class_idx]
        false_positives = np.where((our_labels == class_idx) & (true_labels != class_idx))[0]
        top_n_false_positives = np.argsort(class_predictions[false_positives])[::-1][:n]
        idx_to_plot = false_positives[top_n_false_positives]
        for img_idx in idx_to_plot:
            plt.plot(imread(filenames[img_idx]))

def plot_bounding_box(image_path, coords, image_size=(300,300)):
    H = image_size[0]
    W = image_size[1]
    img = imresize(imread(path), (H, W))
    xmin = coords[0] * W
    ymin = coords[1] * H
    xmax = coords[2] * W
    ymax = coords[3] * H
    rec_width = xmax - xmin
    rec_height = ymax - ymin
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle((xmin,ymin), rec_width, rec_height,
                             linewidth=1, edgecolor='r', fill=False)
    ax.add_patch(rect)
    plt.show()

