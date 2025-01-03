import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(script_dir, '../models') 
sys.path.insert(0, module_dir)
import numpy as np
import datetime
import time
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from barlow_twins import BarlowModel, WarmUpCosine, Conv_1D_block_2, bottleneck_block_2


class CnnLstmModel():
    def __init__(self, config, train_data, val_data, test_data, repeat):
        
        cnn_lstm_params = config["cnn_lstm_params"]
        self.n_steps = cnn_lstm_params["n_steps"]
        self.epochs = cnn_lstm_params["epochs"]
        self.batch_size = cnn_lstm_params["batch_size"]
        self.dropout_rate = cnn_lstm_params["dropout_rate"]
        self.optimizer_params = cnn_lstm_params["optimizer"]
        self.early_stopping_params = cnn_lstm_params["early_stopping_params"]
        self.model_checkpoint_params = cnn_lstm_params["model_checkpoint_params"]
        self.repeat = repeat
        self.total_repeats = config["repeat"]

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        num_samples_per_split = np.load(os.path.join(config["data_directory"], config["dataset_name"], 'num_samples_per_split.npy'))
        self.n_train_samples = num_samples_per_split[0]
        self.n_val_samples = num_samples_per_split[1]
        self.n_test_samples = num_samples_per_split[2]

        self.classifier_name = config["classifier_name"]
        self.dataset_name = config["dataset_name"]      
        self.saved_model_directory = config["saved_model_directory"]
        self.results_directory = config["results_directory"]

        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Training {self.classifier_name} ...')

        self.model_fit()

    def make_model(self):

        K.clear_session()      
        
        inputs = Input(shape=(self.n_steps, 1))
        x = Conv1D(32, 64, activation='tanh', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(4)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LSTM(32, activation="tanh", recurrent_activation="sigmoid", return_sequences=False, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x) 
        outputs = Dense(2, activation='linear')(x)

        return Model(inputs, outputs)

    def model_fit(self):

        model = self.make_model()
        
        optimizer = optimizers.Adam(**self.optimizer_params)

        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
        print(model.summary()) 
        
        cp = ModelCheckpoint(filepath=os.path.join(self.saved_model_directory, self.dataset_name, f'{self.classifier_name}_{self.dataset_name}_{self.repeat}.h5'), **self.model_checkpoint_params)

        es = EarlyStopping(**self.early_stopping_params)

        tb = TensorBoard(log_dir=os.path.join(self.saved_model_directory, self.dataset_name, f'tensorboard_{self.classifier_name}_{self.dataset_name}_{self.repeat}'))

        model.fit(self.train_data,
                  steps_per_epoch=self.n_train_samples // self.batch_size,
                  validation_data=self.val_data,
                  validation_steps=self.n_val_samples // self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  callbacks=[cp, es, tb])
        
        total_sbp_mae = []
        total_dbp_mae = []
        val_dataset_iter = iter(self.val_data)
        for _ in range(int(self.n_val_samples // self.batch_size)):
            ppg, bp = val_dataset_iter.next()
            prediction = model(ppg)
            total_sbp_mae.append(mean_absolute_error(prediction[:, 0].numpy().squeeze(), bp[0].numpy()))
            total_dbp_mae.append(mean_absolute_error(prediction[:, 1].numpy().squeeze(), bp[1].numpy()))
            
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: End of run {self.repeat + 1}/{self.total_repeats} with val SBP MAE {np.mean(total_sbp_mae)} \u00B1 {np.std(total_sbp_mae)} and DBP MAE {np.mean(total_dbp_mae)} \u00B1 {np.std(total_dbp_mae)}')
        
    
    def evaluation(self):

        model_path = os.path.join(self.saved_model_directory, self.dataset_name, f'{self.classifier_name}_{self.dataset_name}_{self.repeat}.h5')
        model = self.make_model()
        model.load_weights(model_path)

        total_sbp_mae = []
        total_dbp_mae = []
        test_dataset_iter = iter(self.test_data)
        for _ in range(int(self.n_test_samples // self.batch_size)):
            ppg, bp = test_dataset_iter.next()
            prediction = model(ppg)
            total_sbp_mae.append(mean_absolute_error(prediction[:, 0].numpy().squeeze(), bp[0].numpy()))
            total_dbp_mae.append(mean_absolute_error(prediction[:, 1].numpy().squeeze(), bp[1].numpy()))
            
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: End of run {self.repeat + 1}/{self.total_repeats} with test SBP MAE {np.mean(total_sbp_mae)} \u00B1 {np.std(total_sbp_mae)} and DBP MAE {np.mean(total_dbp_mae)} \u00B1 {np.std(total_dbp_mae)}')
        
        ppg, _ = next(iter(self.test_data))
        sample = ppg[0, :].numpy().reshape(1, self.n_steps, 1)

        # Calculate total parameters
        total_params = model.count_params()

        # Estimate memory usage (assuming 32-bit floating-point weights)
        memory_usage_mb = (total_params * 4) / (1024 * 1024) 

        # Measure inference latency
        start_time = time.time()
        for _ in range(100):  # Run inference multiple times for better accuracy
            _ = model(sample)

        end_time = time.time()
        inference_latency = (end_time - start_time) / 100

        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Estimated memory usage {memory_usage_mb:.4f} MB and inference latency {inference_latency:.5f} seconds')
        
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        
        return  [np.mean(total_sbp_mae), np.std(total_sbp_mae)], [np.mean(total_dbp_mae), np.std(total_dbp_mae)], memory_usage_mb, inference_latency        
            

class Encoder_MLP():
    def __init__(self, config, train_data, val_data, test_data, repeat):
        
        encoder_mlp_params = config["encoder_mlp_params"]
        self.n_steps = encoder_mlp_params["n_steps"]
        self.epochs = encoder_mlp_params["epochs"]
        self.batch_size = encoder_mlp_params["batch_size"]
        self.optimizer_params = encoder_mlp_params["optimizer"]
        self.early_stopping_params = encoder_mlp_params["early_stopping_params"]
        self.model_checkpoint_params = encoder_mlp_params["model_checkpoint_params"]
        self.repeat = repeat
        self.total_repeats = config["repeat"]

        valid_bp_ranges_params = config["valid_bp_ranges"]
        self.up_sbp = valid_bp_ranges_params["up_sbp"]
        self.low_sbp = valid_bp_ranges_params["low_sbp"]
        self.up_dbp = valid_bp_ranges_params["up_dbp"]
        self.low_dbp = valid_bp_ranges_params["low_dbp"]

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        num_samples_per_split = np.load(os.path.join(config["data_directory"], config["dataset_name"], 'num_samples_per_split.npy'))
        self.n_train_samples = num_samples_per_split[0]
        self.n_val_samples = num_samples_per_split[1]
        self.n_test_samples = num_samples_per_split[2]

        self.encoder_name = config["encoder_name"]
        self.classifier_name = config["classifier_name"]
        self.dataset_name = config["dataset_name"]      
        self.saved_model_directory = config["saved_model_directory"]
        self.results_directory = config["results_directory"]

        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Restoring {self.encoder_name} ...')

        encoder_path = os.path.join(self.saved_model_directory, self.dataset_name, f'{self.encoder_name}_{self.dataset_name}_0.h5') 
        encoder = BarlowModel(config).mobile_net
        encoder.load_weights(encoder_path)
        encoder.trainable = False
        self.encoder = encoder

        self.model_fit()

    def make_model(self):   
                
        K.clear_session()        
        
        inputs = Input(shape=(self.n_steps, 1))
        x = self.encoder(inputs)
        x_sbp = Dense(1, activation='linear', name='SBP')(x)
        x_dbp = Dense(1, activation='linear', name='DBP')(x)

        return Model(inputs=inputs, outputs=[x_sbp, x_dbp])
        
    def model_fit(self):

        model = self.make_model()

        steps_per_epoch = self.n_train_samples // self.batch_size

        warmup_epochs = int(self.epochs * 0.1)
        warmup_steps = int(warmup_epochs * steps_per_epoch)

        lr_decayed_fn = WarmUpCosine(
            learning_rate_base=self.optimizer_params["learning_rate"],
            total_steps=self.epochs * steps_per_epoch,
            warmup_learning_rate=self.optimizer_params["warmup_learning_rate"],
            warmup_steps=warmup_steps
        )
        
        optimizer = optimizers.SGD(learning_rate=lr_decayed_fn)

        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
        print(model.summary()) 
        
        cp = ModelCheckpoint(filepath=os.path.join(self.saved_model_directory, self.dataset_name, f'{self.encoder_name}+{self.classifier_name}_{self.dataset_name}_{self.repeat}.h5'), **self.model_checkpoint_params)

        es = EarlyStopping(**self.early_stopping_params)

        tb = TensorBoard(log_dir=os.path.join(self.saved_model_directory, self.dataset_name, f'tensorboard_{self.encoder_name}+{self.classifier_name}_{self.dataset_name}_{self.repeat}'))

        model.fit(self.train_data,
                  steps_per_epoch=self.n_train_samples // self.batch_size,
                  validation_data=self.val_data,
                  validation_steps=self.n_val_samples // self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  callbacks=[cp, es, tb])
        
        total_sbp_mae = []
        total_dbp_mae = []
        val_dataset_iter = iter(self.val_data)
        for _ in range(int(self.n_val_samples // self.batch_size)):
            ppg, bp = val_dataset_iter.next()
            prediction = model(ppg)
            total_sbp_mae.append(mean_absolute_error(prediction[0].numpy().squeeze(), bp[0].numpy()))
            total_dbp_mae.append(mean_absolute_error(prediction[1].numpy().squeeze(), bp[1].numpy()))
            
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: End of run {self.repeat + 1}/{self.total_repeats} with val SBP MAE {np.mean(total_sbp_mae)} \u00B1 {np.std(total_sbp_mae)} and DBP MAE {np.mean(total_dbp_mae)} \u00B1 {np.std(total_dbp_mae)}')

    
    def evaluation(self):

        model_path = os.path.join(self.saved_model_directory, self.dataset_name, f'{self.encoder_name}+{self.classifier_name}_{self.dataset_name}_{self.repeat}.h5')
        model = self.make_model()
        model.load_weights(model_path)

        total_sbp_mae = []
        total_dbp_mae = []
        test_dataset_iter = iter(self.test_data)
        for _ in range(int(self.n_test_samples // self.batch_size)):
            ppg, bp = test_dataset_iter.next()
            prediction = model(ppg)
            total_sbp_mae.append(mean_absolute_error(prediction[0].numpy().squeeze(), bp[0].numpy()))
            total_dbp_mae.append(mean_absolute_error(prediction[1].numpy().squeeze(), bp[1].numpy()))
            
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: End of run {self.repeat + 1}/{self.total_repeats} with test SBP MAE {np.mean(total_sbp_mae)} \u00B1 {np.std(total_sbp_mae)} and DBP MAE {np.mean(total_dbp_mae)} \u00B1 {np.std(total_dbp_mae)}')
        
        ppg, _ = next(iter(self.test_data))
        sample = ppg[0, :].numpy().reshape(1, self.n_steps, 1)

        # Calculate total parameters
        total_params = model.count_params()

        # Estimate memory usage (assuming 32-bit floating-point weights)
        memory_usage_mb = (total_params * 4) / (1024 * 1024) 

        # Measure inference latency
        start_time = time.time()
        for _ in range(100):  # Run inference multiple times for better accuracy
            _ = model(sample)

        end_time = time.time()
        inference_latency = (end_time - start_time) / 100

        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Estimated memory usage {memory_usage_mb:.4f} MB and inference latency {inference_latency:.5f} seconds')
        
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        
        return  [np.mean(total_sbp_mae), np.std(total_sbp_mae)], [np.mean(total_dbp_mae), np.std(total_dbp_mae)], memory_usage_mb, inference_latency      
    

class MN_MLP():
    def __init__(self, config, train_data, val_data, test_data, repeat):
        
        mn_mlp_params = config["mn_mlp_params"]
        self.n_steps = mn_mlp_params["n_steps"]
        self.epochs = mn_mlp_params["epochs"]
        self.batch_size = mn_mlp_params["batch_size"]
        self.optimizer_params = mn_mlp_params["optimizer"]
        self.early_stopping_params = mn_mlp_params["early_stopping_params"]
        self.model_checkpoint_params = mn_mlp_params["model_checkpoint_params"]
        self.alpha = mn_mlp_params["alpha"]
        self.repeat = repeat
        self.total_repeats = config["repeat"]

        valid_bp_ranges_params = config["valid_bp_ranges"]
        self.up_sbp = valid_bp_ranges_params["up_sbp"]
        self.low_sbp = valid_bp_ranges_params["low_sbp"]
        self.up_dbp = valid_bp_ranges_params["up_dbp"]
        self.low_dbp = valid_bp_ranges_params["low_dbp"]

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        num_samples_per_split = np.load(os.path.join(config["data_directory"], config["dataset_name"], 'num_samples_per_split.npy'))
        self.n_train_samples = num_samples_per_split[0]
        self.n_val_samples = num_samples_per_split[1]
        self.n_test_samples = num_samples_per_split[2]

        self.encoder_name = config["encoder_name"]
        self.classifier_name = config["classifier_name"]
        self.dataset_name = config["dataset_name"]      
        self.saved_model_directory = config["saved_model_directory"]
        self.results_directory = config["results_directory"]

        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Training {self.classifier_name} ...')
        self.model_fit()

    def make_model(self):   
                
        K.clear_session()        
    
        inputs = tf.keras.Input((self.n_steps, 1))

        x = Conv_1D_block_2(inputs, 16, 3, strides=2, nl='HS')
        x = bottleneck_block_2(x, 16, 3, e=16, s=2, squeeze=True, nl='RE', alpha=self.alpha)
        x = bottleneck_block_2(x, 24, 3, e=72, s=2, squeeze=False, nl='RE', alpha=self.alpha)
        x = bottleneck_block_2(x, 24, 3, e=88, s=1, squeeze=False, nl='RE', alpha=self.alpha)
        x = bottleneck_block_2(x, 40, 5, e=96, s=2, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 40, 5, e=240, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 40, 5, e=240, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 48, 5, e=120, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 48, 5, e=144, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 96, 5, e=288, s=2, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 96, 5, e=576, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 96, 5, e=576, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = Conv_1D_block_2(x, 576, 1, strides=1, nl='HS')
        x = x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
        x = tf.keras.layers.Conv1D(1280, 1, padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        
        x_sbp = Dense(1, activation='linear', name='SBP')(x)
        x_dbp = Dense(1, activation='linear', name='DBP')(x)

        return Model(inputs=inputs, outputs=[x_sbp, x_dbp])
        
    def model_fit(self):

        model = self.make_model()

        steps_per_epoch = self.n_train_samples // self.batch_size

        warmup_epochs = int(self.epochs * 0.1)
        warmup_steps = int(warmup_epochs * steps_per_epoch)

        lr_decayed_fn = WarmUpCosine(
            learning_rate_base=self.optimizer_params["learning_rate"],
            total_steps=self.epochs * steps_per_epoch,
            warmup_learning_rate=self.optimizer_params["warmup_learning_rate"],
            warmup_steps=warmup_steps
        )
        
        optimizer = optimizers.SGD(learning_rate=lr_decayed_fn)

        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
        print(model.summary()) 
        
        cp = ModelCheckpoint(filepath=os.path.join(self.saved_model_directory, self.dataset_name, f'{self.classifier_name}_{self.dataset_name}_{self.repeat}.h5'), **self.model_checkpoint_params)

        es = EarlyStopping(**self.early_stopping_params)

        tb = TensorBoard(log_dir=os.path.join(self.saved_model_directory, self.dataset_name, f'tensorboard_{self.classifier_name}_{self.dataset_name}_{self.repeat}'))

        model.fit(self.train_data,
                  steps_per_epoch=self.n_train_samples // self.batch_size,
                  validation_data=self.val_data,
                  validation_steps=self.n_val_samples // self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  callbacks=[cp, es, tb])
        
        total_sbp_mae = []
        total_dbp_mae = []
        val_dataset_iter = iter(self.val_data)
        for _ in range(int(self.n_val_samples // self.batch_size)):
            ppg, bp = val_dataset_iter.next()
            prediction = model(ppg)
            total_sbp_mae.append(mean_absolute_error(prediction[0].numpy().squeeze(), bp[0].numpy()))
            total_dbp_mae.append(mean_absolute_error(prediction[1].numpy().squeeze(), bp[1].numpy()))
            
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: End of run {self.repeat + 1}/{self.total_repeats} with val SBP MAE {np.mean(total_sbp_mae)} \u00B1 {np.std(total_sbp_mae)} and DBP MAE {np.mean(total_dbp_mae)} \u00B1 {np.std(total_dbp_mae)}')

    
    def evaluation(self):

        model_path = os.path.join(self.saved_model_directory, self.dataset_name, f'{self.classifier_name}_{self.dataset_name}_{self.repeat}.h5')
        model = self.make_model()
        model.load_weights(model_path)

        total_sbp_mae = []
        total_dbp_mae = []
        test_dataset_iter = iter(self.test_data)
        for _ in range(int(self.n_test_samples // self.batch_size)):
            ppg, bp = test_dataset_iter.next()
            prediction = model(ppg)
            total_sbp_mae.append(mean_absolute_error(prediction[0].numpy().squeeze(), bp[0].numpy()))
            total_dbp_mae.append(mean_absolute_error(prediction[1].numpy().squeeze(), bp[1].numpy()))
            
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: End of run {self.repeat + 1}/{self.total_repeats} with test SBP MAE {np.mean(total_sbp_mae)} \u00B1 {np.std(total_sbp_mae)} and DBP MAE {np.mean(total_dbp_mae)} \u00B1 {np.std(total_dbp_mae)}')
        
        ppg, _ = next(iter(self.test_data))
        sample = ppg[0, :].numpy().reshape(1, self.n_steps, 1)

        # Calculate total parameters
        total_params = model.count_params()

        # Estimate memory usage (assuming 32-bit floating-point weights)
        memory_usage_mb = (total_params * 4) / (1024 * 1024) 

        # Measure inference latency
        start_time = time.time()
        for _ in range(100):  # Run inference multiple times for better accuracy
            _ = model(sample)

        end_time = time.time()
        inference_latency = (end_time - start_time) / 100

        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Estimated memory usage {memory_usage_mb:.4f} MB and inference latency {inference_latency:.5f} seconds')
        
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        
        return  [np.mean(total_sbp_mae), np.std(total_sbp_mae)], [np.mean(total_dbp_mae), np.std(total_dbp_mae)], memory_usage_mb, inference_latency    