
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report


def full_report(model, directory, generator=ImageDataGenerator(), batch_size=8):
    batches = generator.flow_from_directory(directory, model.input_shape[1:3],
                                            shuffle=False,
                                            batch_size=batch_size)
    print("Generating predictions...")
    y_pred = model.predict_generator(batches, batches.nb_sample)
    print()
    print("PRECISION / RECALL / SUPPORT")
    print(classification_report(y_pred.argmax(1), batches.classes))