import os
import sys
import json
import logging
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Union


class Logger:
    def __init__(self, experiment_name: str = None, mode: str = 'train', run_name: str = None, base_dir: str = 'logs'):
        # Generate names if not provided
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mode = mode.lower()
        
        # Updated directory structure
        self.base_dir = os.path.join(base_dir, self.experiment_name, self.run_name)
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.setup_logger()
        self.metrics_file = os.path.join(self.base_dir, f"{self.mode}_metrics.json")
        self.metrics = []
        
        self.run_info = {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "start_time": datetime.now().isoformat(),
            "status": "started",
            "mode": self.mode,
            "base_dir": self.base_dir
        }
        self.save_run_info()
    
    def setup_logger(self):
        self.logger = logging.getLogger(self.run_name)
        self.logger.setLevel(logging.INFO)
        
        log_file = os.path.join(self.base_dir, "run.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_params(self, params: Dict):
        params_file = os.path.join(self.base_dir, "params.json")
        with open(params_file, "w") as f:
            json.dump(params, f, indent=4)
        self.logger.info("Parameters logged.")
    
    def log_metrics(self, epoch: int, train_loss: Union[float, List[float]] = None, 
                    val_loss: Union[float, List[float]] = None, 
                    test_loss: float = None):
        # Handle list or single value inputs
        if isinstance(train_loss, list):
            for i, loss in enumerate(train_loss):
                metric = {
                    "epoch": i,
                    "train_loss": loss,
                    "val_loss": val_loss[i] if isinstance(val_loss, list) else val_loss
                }
                self.metrics.append(metric)
        else:
            metric = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss
            }
            self.metrics.append(metric)
        
        # Save metrics after each epoch
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)
        
        self.logger.info(f"Metrics logged for epoch {epoch}.")
    
    def plot_losses(self, train_losses: List[float], val_losses: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss", marker="o")
        plt.plot(val_losses, label="Validation Loss", marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss - {self.experiment_name}")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.base_dir, "loss_plot.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Loss plot saved to {plot_path}.")
    
    def save_model(self, model: torch.nn.Module, epoch: int = None):
        model_filename = f"model_epoch_{epoch}.pth" if epoch is not None else "final_model.pth"
        model_path = os.path.join(self.models_dir, model_filename)
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}.")
    
    def complete_run(self):
        self.run_info["status"] = "completed"
        self.run_info["end_time"] = datetime.now().isoformat()
        self.save_run_info()
    
    def save_run_info(self):
        run_info_path = os.path.join(self.base_dir, "run_info.json")
        with open(run_info_path, "w") as f:
            json.dump(self.run_info, f, indent=4)

