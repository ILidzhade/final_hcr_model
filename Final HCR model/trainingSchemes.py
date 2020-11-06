from keras import datasets
from keras.metrics import Accuracy
from matplotlib.pyplot import draw
import numpy as np
import matplotlib.pylab as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from augmentations import AugmentImages


# todo add default/normal training scheme
# todo default should optionally use that augmentor function
# todo augs need to run within loops

def plot_loss_accuracy(H, epochs, label):
    plt.style.use('ggplot')
    plt.figure()
    # plt.plot(np.arange(0, epochs), H['loss'], label='Training loss')
    plt.plot(np.arange(0, epochs), H['accuracy'], label='Training accuracy')
    # plt.plot(np.arange(0, epochs), H['val_loss'], label='Validation loss')
    plt.plot(np.arange(0, epochs), H['val_accuracy'], label='Validation accuracy')
    title = 'Training and Evaluation accuracy ' + label
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def perform_augmentations(set, labels, augmentations, augment_images_obj):
    # print(set.shape)
    augmentation_formulas = [augment_images_obj.extrapolate, augment_images_obj.PatchShuffle, 
        augment_images_obj.invert_imgs, augment_images_obj.swirl_imgs, augment_images_obj.morph_imgs, 
        augment_images_obj.random_erase_imgs, augment_images_obj.flip_imgs]

    out_set, out_labels = set, labels

    for i in augmentations:
        out_set, out_labels = augmentation_formulas[i](set=out_set, labels=out_labels)
    # print(out_set.shape)  
    return out_set, out_labels


def GridSearchTests(X_train, y_train):
    def create_model(v_dropout=0.0, h_dropout=0.0):
        # *can use dropout_rate when creating the model
        #   *means i'll prolly have to refactor this code
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
        model.add(BatchNormalization())
        model.add(Dropout(v_dropout))
        
        model.add(Conv2D(64, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(h_dropout))
        
        model.add(Conv2D(96, 3, activation='relu', padding='same'))
        model.add(MaxPooling2D(2))
        model.add(BatchNormalization())
        model.add(Dropout(h_dropout))
        
        model.add(Conv2D(96, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(h_dropout))
        
        model.add(Conv2D(128, 3, activation='relu', padding='same'))
        model.add(MaxPooling2D(2))
        model.add(BatchNormalization())
        model.add(Dropout(h_dropout))
        
        model.add(Conv2D(128, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(h_dropout))
        
        model.add(Conv2D(192, 3, activation='relu', padding='same'))
        model.add(MaxPooling2D(2))
        model.add(BatchNormalization())
        model.add(Dropout(h_dropout))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(v_dropout))

        model.add(Dense(27, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
        
    gs_model = KerasClassifier(build_fn=create_model)
    # v_dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
    # h_dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
    lr = [0.2, 0.3, 0.4, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    epochs = [100]
    param_grid = dict(lr=lr, epochs=epochs)
    grid = GridSearchCV(estimator=gs_model, param_grid=param_grid, n_jobs=1, cv=3)
    
    X_batch, y_batch = X_train, y_train 
    X_batch = X_batch / 255.0
    y_batch = to_categorical(y_batch)
    grid_result = grid.fit(X_batch, y_batch)
    # *may need to add an evaluation set

    # summarise
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    print("Press any key to continue:")
    continueA = input()


def default_training_scheme(target_model, X_train, y_train, epochs=100, X_test=[], y_test=[], augmentations=[], size=28*28, shape=(28,28,1), dataset_descr='EMNIST', classes=27):
    X_train = X_train.reshape([X_train.shape[0], X_train.shape[1], X_train.shape[2], 1])
    X_test = X_test.reshape([X_test.shape[0], X_test.shape[1], X_test.shape[2], 1])
    X_validation = X_test[:2500] / 255.0
    y_validation = y_test[:2500]
    y_validation = to_categorical(y_validation, num_classes=classes)
    
    augment_images_obj = AugmentImages(X_train, y_train, size=size, shape=shape, dataset_descr=dataset_descr)

    results = []
    title = 'on the baseline model'
    H = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    X_batch, y_batch = perform_augmentations(X_train, y_train, augmentations, augment_images_obj) 
    X_batch = X_batch / 255.0
    y_batch = to_categorical(y_batch, num_classes=classes)
    
    h = target_model.fit(
        X_batch, y_batch,
        validation_data=(X_validation, y_validation), 
        shuffle=True, 
        epochs=epochs, 
        batch_size=16,
        workers=4,
        use_multiprocessing=True
        )
        
    H['loss'] = h.history['loss']
    H['accuracy'] = h.history['accuracy']
    H['val_loss'] = h.history['val_loss']
    H['val_accuracy'] = h.history['val_accuracy']

    plot_loss_accuracy(H, epochs, title)
    y_test = to_categorical(y_test[2500:], num_classes=classes)
    X_test = X_test[2500:] / 255.0
    score = target_model.evaluate(X_test, y_test)
    print('Loss: ', score[0])
    print('Accuracy: ', score[1])
    print()
    y_pred = target_model.predict_classes(X_test)
    y_pred = to_categorical(y_pred, num_classes=classes)
    print(metrics.classification_report(y_test, y_pred))
    return target_model


def SimplePairing(input_set, input_labels):
    
    output_set = []
    output_labels = []
    p = np.random.randint(0, 10)

    for i in range(len(input_set)):
        temp_img = input_set[i]
        temp_label = input_labels[i]
        
        for j in range(10): 
            sp1 = (temp_img + input_set[np.random.randint(0, len(input_set))]) / 2
            output_set.append(sp1)
            output_labels.append(temp_label)
    return np.asarray(output_set), np.asarray(output_labels)


def train_via_SimplePairing(target_model, X_train, y_train, X_test=[], y_test=[], augmentations=[], size=28*28, shape=(28,28,1), dataset_descr='EMNIST', classes=27):
    X_train = X_train.reshape([X_train.shape[0], X_train.shape[1], X_train.shape[2], 1])
    X_test = X_test.reshape([X_test.shape[0], X_test.shape[1], X_test.shape[2], 1])
    X_validation = X_test[:2500] / 255.0
    y_validation = y_test[:2500]
    y_validation = to_categorical(y_validation, num_classes=classes)

    H = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    print('shape')
    print(shape)
    augment_images_obj = AugmentImages(X_train, y_train, size=size, shape=shape, dataset_descr=dataset_descr)

    train_len = X_train.shape[0]
    for j in range(100):
        print("(Initial training phase)")
        print("epoch:", j + 1, "/ 100")
        
        X_batch, y_batch = perform_augmentations(X_train[:train_len//2], y_train[:train_len//2], augmentations, augment_images_obj) 
        X_batch = X_batch / 255.0
        y_batch = to_categorical(y_batch, num_classes=classes)

        h = target_model.fit(
            X_batch,
            y_batch,
            validation_data=(X_validation, y_validation),
            batch_size=64,
            shuffle=True,
            verbose = 2
       )
        H['loss'].append(h.history['loss'][0])
        H['accuracy'].append(h.history['accuracy'][0])
        H['val_loss'].append(h.history['val_loss'][0])
        H['val_accuracy'].append(h.history['val_accuracy'][0])

    x = 0
    for i in range(200):
        print("(Intermittently applying sample pairing)")
        print("epoch: ", i + 1, "/ 200")

        if x < 8:
            X_batch, y_batch = perform_augmentations(X_train[:train_len//2], y_train[:train_len//2], augmentations, augment_images_obj) 
            X_batch = X_batch / 255.0
            X_batch, y_batch = SimplePairing(X_batch, y_batch)
        else:
            X_batch, y_batch = perform_augmentations(X_train[:train_len//2], y_train[:train_len//2], augmentations, augment_images_obj) 
            X_batch = X_batch / 255.0
        y_batch = to_categorical(y_batch, num_classes=classes)

        x = (x + 1) % 10
        h = target_model.fit(
            X_batch,
            y_batch,
            validation_data=(X_validation, y_validation),
            batch_size=128,
            shuffle=True
        )
        H['loss'].append(h.history['loss'][0])
        H['accuracy'].append(h.history['accuracy'][0])
        H['val_loss'].append(h.history['val_loss'][0])
        H['val_accuracy'].append(h.history['val_accuracy'][0])

    for j in range(100):
        print("Finetuning:")
        print("epoch: ", j + 1, "/ 100")
        
        X_batch, y_batch = perform_augmentations(X_train[train_len//2:], y_train[train_len//2:], augmentations, augment_images_obj) 
        X_batch = X_batch / 255.0
        y_batch = to_categorical(y_batch, num_classes=classes)

        h = target_model.fit(
            X_batch,
            y_batch,
            validation_data=(X_validation, y_validation),
            batch_size=64,
            shuffle=True
        )
        H['loss'].append(h.history['loss'][0])
        H['accuracy'].append(h.history['accuracy'][0])
        H['val_loss'].append(h.history['val_loss'][0])
        H['val_accuracy'].append(h.history['val_accuracy'][0])
    
    plot_loss_accuracy(H, 400, " SimplePairing training scheme")
    X_test = X_test[2500:] / 255.0
    y_test = to_categorical(y_test[2500:], num_classes=classes)
    score = target_model.evaluate(X_test, y_test)
    print('Loss: ', score[0])
    print('Accuracy: ', score[1])
    print()
    y_pred = target_model.predict_classes(X_test)
    y_pred = to_categorical(y_pred, num_classes=classes)
    print(metrics.classification_report(y_test, y_pred))

    return target_model


def train_on_extracted_context_vectors(baseline_model, X_train, y_train, X_test=[], y_test=[], epochs=100, augmentations=[], layers=24, size=0, shape=(0, 0), classes=27, dataset_descr='', class_layer_size=1728):
    print("training the model...")
    baseline_model = default_training_scheme(baseline_model, X_train, y_train, epochs=epochs, X_test=X_test, y_test=y_test, augmentations=augmentations, size=size, shape=shape, dataset_descr=dataset_descr, classes=classes)
    
    feature_extractor = Sequential()

    for layer in baseline_model.layers[:layers]:
        feature_extractor.add(layer)
    feature_extractor.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    len = X_train.shape[0]
    
    X_train = X_train.reshape([len, shape[0], shape[1], 1])
    X_test = X_test.reshape([X_test.shape[0], shape[0], shape[1], 1])
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = to_categorical(y_train, num_classes=classes)
    y_test = to_categorical(y_test, num_classes=classes)
    X_validation, y_validation = X_test[:2500], y_test[:2500]
    
    # cv = context vectors
    X_train_cv = feature_extractor.predict(X_train)
    X_validation = feature_extractor.predict(X_validation)

    classifier = Sequential()
    for layer in baseline_model.layers[layers:]:
        classifier.add(layer)

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    results = []
    H = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    X_batch, y_batch = X_train_cv, y_train
    X_batch = X_batch.reshape([X_batch.shape[0], class_layer_size])
    X_validation = X_validation.reshape([2500, class_layer_size])  
    print('performing icing on the cake...')
    h = classifier.fit(X_batch, y_batch, validation_data=(X_validation, y_validation), shuffle=True, epochs=epochs, batch_size=16, workers=4, use_multiprocessing=True)

    H['loss'].append(h.history['loss'][0])
    H['accuracy'].append(h.history['accuracy'][0])
    H['val_loss'].append(h.history['val_loss'][0])
    H['val_accuracy'].append(h.history['val_accuracy'][0])

    # plot_loss_accuracy(H, 99, " using icing on the cake")
    # X_test = feature_extractor.predict(X_test[2500:])
    # y_test = y_test[2500:]
    # score = classifier.evaluate(X_test, y_test)
    # print('Loss: ', score[0])
    # print('Accuracy: ', score[1])
    # print()
    # y_pred = classifier.predict_classes(X_test)
    # y_pred = to_categorical(y_pred, num_classes=classes)
    # print(metrics.classification_report(y_test, y_pred))

    final_model = Sequential()
    for layer in feature_extractor.layers:
        final_model.add(layer)
    for layer in classifier.layers:
        final_model.add(layer)
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    y_test = y_test[2500:]
    score = final_model.evaluate(X_test[2500:], y_test)
    print('Loss: ', score[0])
    print('Accuracy: ', score[1])
    print()
    y_pred = final_model.predict_classes(X_test[2500:])
    y_pred = to_categorical(y_pred, num_classes=classes)
    print(metrics.classification_report(y_test, y_pred))
    return final_model