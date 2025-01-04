import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))

# List of directories to add to the Python path
directories_to_add = [
    './data_utilities',
    './models'
]

# Insert directories at the beginning of the path (for higher priority)
for directory in directories_to_add:
    module_dir = os.path.join(script_dir, directory) 
    sys.path.insert(0, module_dir) 
from time import time
from datetime import datetime
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import logging
from sklearn.metrics import mean_absolute_error
import joblib
from prepare_and_split_mimic_iii import prepare_mimic_iii_dataset 
from mobilenet_v3 import MobileNetV3_small

# For deterministic algorithms
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 


class TrainingLogger:
    def __init__(self, log_interval=10, total_epochs=150):
        self.training_logs = []
        self.validation_logs = []
        self.test_logs = []
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    def generic_log(self, msg):
        logging.info(f"{msg}")

    def begin(self, epoch=None, stage='training'):
        
        if stage == 'training':
            self.epoch_training_start_time = time()
            logging.info(f"Epoch {epoch + 1}/{self.total_epochs} training start.")
        
        elif stage == 'validation':
            self.epoch_validation_start_time = time()
            logging.info(f"Epoch {epoch + 1}/{self.total_epochs} validation start.")

        else:
            self.test_start_time = time()
            logging.info(f"Test stage start.")
    
    def end(self, epoch=None, stage='training', logs=None):
        
        if stage == 'training':
            elapsed_time = time() - self.epoch_training_start_time
            logging.info(f"Training during epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
            logs['epoch_time'] = elapsed_time
            self.training_logs.append(logs)
        
        elif stage == 'validation':
            elapsed_time = time() - self.epoch_validation_start_time
            logging.info(f"Validation during epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
            logs['epoch_time'] = elapsed_time
            self.validation_logs.append(logs)

        else:
            elapsed_time = time() - self.test_start_time
            logging.info(f"Test after training and validation finished in {elapsed_time:.2f} seconds.")
            logs['epoch_time'] = elapsed_time
            self.test_logs.append(logs)

    def on_batch_end(self, batch, stage='training', logs=None):
        if (batch + 1) % self.log_interval == 0:
            if stage == 'training':
                logging.info(f"Training batch {batch + 1}: loss = {logs['loss']:.4f}, SBP MAE = {logs['sbp_mae']:.4f} / DBP MAE = {logs['dbp_mae']:.4f}")
            elif stage == 'validation':
                logging.info(f"Validation batch {batch + 1}: loss = {logs['loss']:.4f}, SBP MAE = {logs['sbp_mae']:.4f} / DBP MAE = {logs['dbp_mae']:.4f}")
            else:
                logging.info(f"Test batch {batch + 1}: loss = {logs['loss']:.4f}, SBP MAE = {logs['sbp_mae']:.4f} / DBP MAE = {logs['dbp_mae']:.4f}")


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        #print(f"Best: {self.best_loss} Current: {val_loss} Counter: {self.counter}")
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            #print(f"Metric improving")
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            #print(f"Metric not improving")
            self.counter += 1
            # print(f"\nINFO: Early stopping counter {self.counter} of {self.patience}\n")
            if self.counter >= self.patience:
                print('\nINFO: Early stopping\n')
                self.early_stop = True



class SignalDataset(Dataset):
    def __init__(self, config, split):
        self.signals = np.load(os.path.join(config["data_directory"], config["dataset_name"], split, 'ppg.npy'))
        self.labels = np.load(os.path.join(config["data_directory"], config["dataset_name"], split, 'bp.npy'))

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx]).float()
        label = torch.from_numpy(self.labels[idx]).float() 
        return signal, (label[0], label[1])
    

def train_loop(model, train_loader, optimizer, criterion, device, logger):
    model.train()
    
    train_loss = 0.0
    train_sbp_mae = 0.0
    train_dbp_mae = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        
        optimizer.zero_grad()

        output = model(data.unsqueeze(1))
        
        sbp_loss = criterion(output[0].squeeze(), target[0].to(device))
        dbp_loss = criterion(output[1].squeeze(), target[1].to(device))
        loss = sbp_loss + dbp_loss
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        sbp_mae = mean_absolute_error(target[0].cpu().detach().numpy(), output[0].squeeze().cpu().detach().numpy())
        dbp_mae = mean_absolute_error(target[1].cpu().detach().numpy(), output[1].squeeze().cpu().detach().numpy())
        train_sbp_mae += sbp_mae
        train_dbp_mae += dbp_mae

        logger.on_batch_end(batch_idx, stage='training', logs={
                'loss': train_loss / (batch_idx + 1),
                'sbp_mae': train_sbp_mae / (batch_idx + 1),
                'dbp_mae': train_dbp_mae / (batch_idx + 1)
            })
    
    logs = {
        'loss': train_loss / len(train_loader),
        'sbp_mae': train_sbp_mae / len(train_loader),
        'dbp_mae': train_dbp_mae / len(train_loader)
    }

    logger.generic_log(f'Training summary: loss {logs["loss"]}, Total SBP MAE {logs["sbp_mae"]} / Total DBP MAE {logs["dbp_mae"]}')
    
    return logs


def validate_loop(model, val_loader, criterion, device, logger):
    model.eval()

    val_loss = 0.0
    val_sbp_mae = 0.0
    val_dbp_mae = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)

            output = model(data.unsqueeze(1))
            
            sbp_loss = criterion(output[0].squeeze(), target[0].to(device))
            dbp_loss = criterion(output[1].squeeze(), target[1].to(device))
            loss = sbp_loss + dbp_loss
            
            val_loss += loss.item()
            
            sbp_mae = mean_absolute_error(target[0].cpu().detach().numpy(), output[0].squeeze().cpu().detach().numpy())
            dbp_mae = mean_absolute_error(target[1].cpu().detach().numpy(), output[1].squeeze().cpu().detach().numpy())
            val_sbp_mae += sbp_mae
            val_dbp_mae += dbp_mae

            logger.on_batch_end(batch_idx, stage='validation', logs={
                    'loss': val_loss / (batch_idx + 1),
                    'sbp_mae': val_sbp_mae / (batch_idx + 1),
                    'dbp_mae': val_dbp_mae / (batch_idx + 1)
                })

    logs = {
        'loss': val_loss / len(val_loader),
        'sbp_mae': val_sbp_mae / len(val_loader),
        'dbp_mae': val_dbp_mae / len(val_loader)
    }

    logger.generic_log(f'Validation summary: loss {logs["loss"]}, Total SBP MAE {logs["sbp_mae"]} / Total DBP MAE {logs["dbp_mae"]}')

    return logs


def test_loop(model, test_loader, criterion, device, logger):
    model.eval()

    test_loss = 0.0
    test_sbp_mae = 0.0
    test_dbp_mae = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data.to(device)

            output = model(data.unsqueeze(1))

            sbp_loss = criterion(output[0].squeeze(), target[0].to(device))
            dbp_loss = criterion(output[1].squeeze(), target[1].to(device))
            loss = sbp_loss + dbp_loss
            
            test_loss += loss.item()
            
            sbp_mae = mean_absolute_error(target[0].cpu().detach().numpy(), output[0].squeeze().cpu().detach().numpy())
            dbp_mae = mean_absolute_error(target[1].cpu().detach().numpy(), output[1].squeeze().cpu().detach().numpy())
            test_sbp_mae += sbp_mae
            test_dbp_mae += dbp_mae

            logger.on_batch_end(batch_idx, stage='test', logs={
                    'loss': test_loss / (batch_idx + 1),
                    'sbp_mae': test_sbp_mae / (batch_idx + 1),
                    'dbp_mae': test_dbp_mae / (batch_idx + 1)
                })

    logs = {
        'loss': test_loss / len(test_loader),
        'sbp_mae': test_sbp_mae / len(test_loader),
        'dbp_mae': test_dbp_mae / len(test_loader)
    }

    logger.generic_log(f'Test summary: loss {logs["loss"]}, Total SBP MAE {logs["sbp_mae"]} / Total DBP MAE {logs["dbp_mae"]}')

    return logs


def main():

    # Load configuration
    with open('config_mimic_iii.json') as config_file:
        config = json.load(config_file)

    torch.manual_seed(config["seed"])

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Number of GPUs available: {num_gpus}')
        device = torch._C.device(f'cuda:{config["gpu_id"]}')

        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Using GPU {config["gpu_id"]} - {torch.cuda.get_device_name(config["gpu_id"])}')
    else:
        raise ValueError("Impossible to compute without GPU :(")

    os.makedirs(os.path.join(config["data_directory"], config["dataset_name"]), exist_ok=True)
    os.makedirs(os.path.join(config["results_directory"], config["dataset_name"]), exist_ok=True)
    os.makedirs(os.path.join(config["saved_models_directory"], config["dataset_name"]), exist_ok=True)

    # Preprocess and split into train/val/test
    prepare_mimic_iii_dataset(config)

    # Create datasets
    train_dataset = SignalDataset(config, 'train')
    val_dataset = SignalDataset(config, 'val')
    test_dataset = SignalDataset(config, 'test')

    num_subjects_per_split = np.load(os.path.join(config["data_directory"], config["dataset_name"], 'num_subjects_per_split.npy'))
    n_train_subjects = num_subjects_per_split[0]
    n_val_subjects = num_subjects_per_split[1]
    n_test_subjects = num_subjects_per_split[2]

    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Total # of subjects for training/validation/testing {n_train_subjects}/{n_val_subjects}/{n_test_subjects}')

    num_samples_per_split = np.load(os.path.join(config["data_directory"], config["dataset_name"], 'num_samples_per_split.npy'))
    n_train_samples = num_samples_per_split[0]
    n_val_samples = num_samples_per_split[1]
    n_test_samples = num_samples_per_split[2]
    
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Total # of samples for training/validation/testing {n_train_samples}/{n_val_samples}/{n_test_samples}')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["n_cores"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["n_cores"], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], num_workers=config["n_cores"], pin_memory=True)

    log_callback = TrainingLogger(log_interval=1000, total_epochs=config["epochs"])

    model_directory = os.path.join(config["saved_models_directory"], config["dataset_name"], config["model_name"])
    os.makedirs(model_directory, exist_ok=True)
    total_repeats = config["repeat"] 
    for repeat in range(total_repeats):
        
        # Initialize model, optimizer, and loss function
        model = MobileNetV3_small(in_channels=1)
        print(model)
        model.to(device)

        optimizer_params = config["optimizer"]
        optimizer = torch.optim.RMSprop(model.parameters(), lr=optimizer_params["lr"], weight_decay=optimizer_params["l2norm"])
        
        lr_scheduler_params = config["lr_scheduler"]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, 
            milestones=tuple([int(s) for s in lr_scheduler_params["lrsched_step"].split(sep=",")]), 
            gamma=lr_scheduler_params["lrsched_gamma"]
            )
        
        early_stopping_params = config["early_stopping_params"]
        es = EarlyStopping(patience=early_stopping_params['es_patience'], min_delta=early_stopping_params['es_min_delta'])
        
        criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=5).to(device)

        # Training loop
        num_epochs = config["epochs"]
        best_val_loss = float('inf')
        best_model_path = os.path.join(model_directory, f'{config["model_name"]}_best_repeat_{repeat}.pth')
        
        for epoch in range(num_epochs):

            log_callback.begin(epoch=epoch, stage='training')
            train_logs = train_loop(model, train_loader, optimizer, criterion, device, log_callback)
            log_callback.end(epoch=epoch, stage='training', logs=train_logs)

            log_callback.begin(epoch=epoch, stage='validation')
            val_logs = validate_loop(model, val_loader, criterion, device, log_callback)
            log_callback.end(epoch=epoch, stage='validation', logs=val_logs)

            lr_scheduler.step()

            if val_logs['loss'] < best_val_loss:
                best_val_loss = val_logs['loss']
                torch.save(model.state_dict(), best_model_path)
            
            es(val_logs["loss"])
            if es.early_stop:
                break

        # Load the best model
        model = MobileNetV3_small(in_channels=1)
        print(model)
        model.load_state_dict(torch.load(best_model_path))
        model.to('cpu')

        # Evaluate on the test set
        log_callback.begin(epoch=epoch, stage='test')
        test_logs = test_loop(model, test_loader, criterion, 'cpu', log_callback)
        log_callback.end(epoch=epoch, stage='test', logs=test_logs)

        # Save results
        joblib.dump(log_callback.training_logs, os.path.join(model_directory, 'training_summary.pkl'))
        joblib.dump(log_callback.validation_logs, os.path.join(model_directory, 'validation_summary.pkl'))
        joblib.dump(log_callback.test_logs, os.path.join(model_directory, 'test_summary.pkl'))

if __name__ == '__main__':
    main()