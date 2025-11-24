"""
Data loader for images and videos
"""

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class VideoImageDataset(Dataset):
    """
    Dataset que maneja tanto imágenes como videos.
    Para videos, extrae frames automáticamente.
    """
    
    def __init__(self, root_dir, transform=None, frames_per_video=16):
        """
        Args:
            root_dir: Directorio raíz con estructura clase1/, clase2/, etc.
            transform: Transformaciones de torchvision
            frames_per_video: Número de frames a extraer de cada video
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Cargar todos los archivos y crear el mapping de clases"""
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Directorio {self.root_dir} no existe")
        
        # Obtener clases (subdirectorios)
        classes = sorted([d for d in os.listdir(self.root_dir)
                         if os.path.isdir(os.path.join(self.root_dir, d))])
        
        if len(classes) == 0:
            raise ValueError(f"No se encontraron clases en {self.root_dir}")
        
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Extensiones soportadas
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        video_exts = ('.mp4', '.avi', '.mov', '.mkv')
        
        # Cargar archivos
        for class_name in classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                
                if not os.path.isfile(filepath):
                    continue
                
                # Verificar si es imagen o video
                if filename.lower().endswith(image_exts):
                    self.samples.append({
                        'path': filepath,
                        'class': class_idx,
                        'type': 'image'
                    })
                elif filename.lower().endswith(video_exts):
                    self.samples.append({
                        'path': filepath,
                        'class': class_idx,
                        'type': 'video'
                    })
    
    def _load_image(self, path):
        """Cargar una imagen"""
        try:
            image = Image.open(path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error cargando imagen {path}: {e}")
            # Retornar imagen negra como fallback
            return Image.new('RGB', (224, 224), color='black')
    
    def _extract_video_frames(self, path):
        """Extraer frames de un video"""
        try:
            cap = cv2.VideoCapture(path)
            
            # Obtener información del video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                # Video vacío, retornar imagen negra
                cap.release()
                return Image.new('RGB', (224, 224), color='black')
            
            # Calcular índices de frames a extraer (distribuidos uniformemente)
            if total_frames <= self.frames_per_video:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames-1, self.frames_per_video, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convertir BGR a RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                # No se pudo extraer ningún frame
                return Image.new('RGB', (224, 224), color='black')
            
            # Por ahora, retornar el frame del medio como representante
            # TODO: En el futuro, podríamos retornar todos los frames
            middle_frame = frames[len(frames) // 2]
            return Image.fromarray(middle_frame)
            
        except Exception as e:
            print(f"Error procesando video {path}: {e}")
            return Image.new('RGB', (224, 224), color='black')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Obtener un item del dataset"""
        sample = self.samples[idx]
        
        # Cargar imagen o frame de video
        if sample['type'] == 'image':
            image = self._load_image(sample['path'])
        else:  # video
            image = self._extract_video_frames(sample['path'])
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        return image, sample['class']


def create_dataloaders(data_dir, batch_size=32, num_workers=4, image_size=224, val_split=None):
    """
    Crear DataLoaders para entrenamiento y validación
    
    Args:
        data_dir: Directorio raíz con subdirectorios train/ y val/
        batch_size: Tamaño del batch
        num_workers: Número de workers para cargar datos
        image_size: Tamaño de las imágenes
        val_split: Si val/ no existe, usa esta proporción de train (ej: 0.2)
    
    Returns:
        train_loader, val_loader
    """
    
    # Transformaciones con data augmentation para train
    train_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.143)),  # Resize un poco más grande
        transforms.RandomCrop(image_size),           # Crop aleatorio
        transforms.RandomHorizontalFlip(),           # Flip horizontal
        transforms.RandomRotation(15),               # Rotación aleatoria
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transformaciones para validación (sin augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.143)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Directorios
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Crear datasets
    train_dataset = VideoImageDataset(train_dir, transform=train_transform)
    
    # Validación
    if os.path.exists(val_dir):
        val_dataset = VideoImageDataset(val_dir, transform=val_transform)
    else:
        # Si no existe val/, usar split de train
        if val_split is None:
            val_split = 0.2
        
        dataset_size = len(train_dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
