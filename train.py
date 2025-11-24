#!/usr/bin/env python3
"""
Script principal de entrenamiento
Ejecutar con: python train.py
"""

import os
import torch
import torch.nn as nn
import argparse
from src.models.vision_model import VisionModel
from src.data.data_loader import create_dataloaders
from src.training.trainer import Trainer, create_optimizer


def count_parameters(model):
    """Contar parámetros del modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_num_classes(data_dir):
    """Obtener número de clases del dataset"""
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"⚠️  Advertencia: {train_dir} no existe")
        return 10  # Default
    
    classes = [d for d in os.listdir(train_dir) 
               if os.path.isdir(os.path.join(train_dir, d))]
    return len(classes)


def main(args):
    """Función principal de entrenamiento"""
    
    print("\n" + "="*60)
    print("🚀 AI Vision Assistant - Entrenamiento")
    print("="*60 + "\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Detectar número de clases
    num_classes = get_num_classes(args.data_dir)
    print(f"✓ Número de clases detectadas: {num_classes}")
    
    # Crear dataloaders
    print(f"✓ Cargando datos desde: {args.data_dir}")
    try:
        train_loader, val_loader = create_dataloaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size
        )
        print(f"  - Batches de entrenamiento: {len(train_loader)}")
        print(f"  - Batches de validación: {len(val_loader)}")
        print(f"  - Tamaño de batch: {args.batch_size}")
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        print("\n💡 Asegúrate de que tus datos estén organizados así:")
        print("data/")
        print("  train/")
        print("    clase1/")
        print("      imagen1.jpg")
        print("    clase2/")
        print("      imagen1.jpg")
        print("  val/")
        print("    clase1/")
        print("      imagen1.jpg")
        return
    
    # Crear modelo
    print(f"✓ Creando modelo (pretrained={args.pretrained})")
    model = VisionModel(num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"  - Parámetros entrenables: {num_params:,}")
    
    # Loss y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, lr=args.lr, optimizer_type=args.optimizer)
    
    print(f"✓ Optimizador: {args.optimizer.upper()}")
    print(f"✓ Learning rate: {args.lr}")
    print(f"✓ Épocas: {args.epochs}")
    
    # Crear trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Entrenar
    history = trainer.train()
    
    print("\n✅ Entrenamiento completado!")
    print(f"📁 Checkpoints guardados en: {args.checkpoint_dir}")
    print(f"📊 Logs guardados en: {args.log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar modelo de visión')
    
    # Datos
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directorio con los datos (default: data)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Tamaño de las imágenes (default: 224)')
    
    # Modelo
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Usar modelo pre-entrenado')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='No usar modelo pre-entrenado')
    
    # Entrenamiento
    parser.add_argument('--epochs', type=int, default=20,
                        help='Número de épocas (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Tamaño del batch (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizador (default: adam)')
    
    # Sistema
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Número de workers para cargar datos (default: 4)')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                        help='Directorio para checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directorio para logs')
    
    args = parser.parse_args()
    
    main(args)
