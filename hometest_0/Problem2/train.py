import tensorflow as tf
import numpy as np

from modules.create_data import (
    X_train,
    X_test,
    y_train,
    y_test
)

from modules.dataset import EEGDataGenerator
from modules.model import EEGModel


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Model selection
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC


class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, val_data, batch_size):
        self.model = model
        self.val_data = val_data
        self.batch_size = batch_size
    def on_epoch_end(self, epoch, logs=None):

        (x,x1), y_val = self.val_data.__getitem__(0)
        y_pred = self.model.predict((x,x1))
        
        y_pred = np.argmax(y_pred, axis=1)
        y_true = y_val
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        print(f'F1-score: {f1:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f}')


if __name__ == '__main__':
    # Train: 1080*0.8 = 864, fold = 864/4 = 216 test: 1080*0.2 = 216
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    test_generator = EEGDataGenerator(X_test, y_test, 128)

    fold = 1
    best_accuracy = 0.0
    accs = list()
    for train, test in kfold.split(X_train, y_train):
        model = EEGModel()
        model.summary(line_length=80)

        learning_rate = tf.constant(0.001)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.9, nesterov=True)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])


        epochs = 0
        train_generator = EEGDataGenerator(X_train[train], y_train[train], 128)
        val_generator = EEGDataGenerator(X_train[test], y_train[test], 128)

        model_checkpoint = ModelCheckpoint('./output/model.h5', monitor='val_accuracy', save_best_only=True)
        mt_callback = MetricsCallback(model, val_generator, batch_size=216)

        print(f'Training for fold {fold} ...')
        model.fit(train_generator,
                  epochs=100,
                  initial_epoch=epochs - 1,
                  validation_data=val_generator,
                  callbacks=[model_checkpoint, mt_callback])

        evaluation = model.evaluate(val_generator)
        val_accuracy = evaluation[1]
        print(f'Validation Accuracy: {val_accuracy}')

        accs.append(val_accuracy)

        fold += 1

    print("Avg accuracies: %.2f%% (+/- %.2f%%)" % (np.mean(accs), np.std(accs)))

    # Load and retrain final model with full train set, use hold out test set as a validation
    model = EEGModel()
    learning_rate = tf.constant(0.001)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])


    train_generator = EEGDataGenerator(X_train, y_train, 128)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint('./output/final_model.h5', monitor='val_accuracy', save_best_only=True)
    mt_callback = MetricsCallback(model, test_generator, batch_size=216)
    
    epochs = 1
    model.fit(train_generator,
              epochs=200,
              initial_epoch=epochs - 1,
              validation_data = test_generator,
              callbacks=[early_stopping, model_checkpoint, mt_callback])

    # Create an SVC classifier
    svc = SVC()
    svc.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]), y_train)
    # Define parameter grid for grid search
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['poly', 'rbf', 'sigmoid'],
    }

    # # Perform grid search using GridSearchCV
    grid_search = GridSearchCV(svc, param_grid, cv=5, verbose=2)
    grid_search.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]), y_train)

    # Print best results
    print("Best Score: {:.4f}".format(grid_search.best_score_))
    print("Best Parameters:", grid_search.best_params_)

    # Evaluate the best model on the test set
    best_svc = grid_search.best_estimator_
    test_accuracy = best_svc.score(X_test, y_test)
    print("Test Set Accuracy: {:.4f}".format(test_accuracy))