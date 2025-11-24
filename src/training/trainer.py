"""
Script de entrenamiento para modelos de visión
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime


class Trainer:
    """Clase para manejar el entrenamiento de modelos"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=10,
        checkpoint_dir='models/checkpoints',
        log_dir='logs'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Crear directorios
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Historia de entrenamiento
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        self.start_time = None
    
    def train_epoch(self):
        """Entrenar una época"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Actualizar barra de progreso
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validar el modelo"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/total:.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Guardar checkpoint del modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        # Guardar checkpoint regular
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f'✓ Mejor modelo guardado con accuracy: {val_acc:.2f}%')
    
    def save_history(self):
        """Guardar historia de entrenamiento"""
        history_path = os.path.join(
            self.log_dir,
            f'history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def train(self):
        """Entrenar el modelo completo"""
        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Épocas: {self.num_epochs}")
        print(f"Batches de entrenamiento: {len(self.train_loader)}")
        print(f"Batches de validación: {len(self.val_loader)}")
        print(f"{'='*60}\n")
        
        self.start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nÉpoca {epoch}/{self.num_epochs}")
            print("-" * 60)
            
            # Entrenar
            train_loss, train_acc = self.train_epoch()
            
            # Validar
            val_loss, val_acc = self.validate()
            
            # Guardar historia
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Imprimir resumen
            print(f"\nResumen época {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Guardar checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, is_best)
        
        # Tiempo total
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Entrenamiento completado!")
        print(f"Tiempo total: {total_time/60:.2f} minutos")
        print(f"Mejor accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        # Guardar historia
        self.save_history()
        
        return self.history


def create_optimizer(model, lr=0.001, optimizer_type='adam'):
    """
    Crear optimizador
    
    Args:
        model: Modelo a optimizar
        lr: Learning rate
        optimizer_type: 'adam' o 'sgd'
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Optimizador no soportado: {optimizer_type}")


def create_scheduler(optimizer, scheduler_type='step', **kwargs):
    """
    Crear learning rate scheduler
    
    Args:
        optimizer: Optimizador
        scheduler_type: 'step', 'cosine', 'plateau'
    """
    if scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50)
        )
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 5)
        )
    else:
        raise ValueError(f"Scheduler no soportado: {scheduler_type}")
