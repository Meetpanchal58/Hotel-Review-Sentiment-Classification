import os
from src.logger.logging import logging
from src.utils.utils import save_GRU
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, Dropout,SpatialDropout1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from src.exception.exception import CustomException


@dataclass
class ModelTrainerConfig:
    model_file_path=os.path.join('artifacts','GRU_Model.h5')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def train_model(self, X_balanced, y_balanced):  
        try:
            logging.info("Model training started")

            X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify = y_balanced)

            #tf.random.set_seed(1)
            #tf.keras.utils.set_random_seed(1)
            gru = Sequential()
            gru.add(Embedding(input_dim=44778, output_dim=100, input_length=1889))
            gru.add(SpatialDropout1D(0.2))
            gru.add(GRU(100, dropout=0.2)) 
            gru.add(Dropout(0.2))
            gru.add(Dense(5, activation='softmax'))
            gru.summary()

            optimizer = Adam(learning_rate=0.01)
            gru.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer = optimizer)

            early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)     
            gru.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_test, y_test), callbacks = [early_stopping])

            save_GRU(
                file_path=self.model_trainer_config.model_file_path,
                model = gru
            )

            logging.info("Model training completed")
            return X_test, y_test

        except Exception as e:
            logging.exception("An error occurred during model training")
            raise CustomException(e)
 
        