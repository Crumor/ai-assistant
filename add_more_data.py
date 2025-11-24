#!/usr/bin/env python3
"""
Script para agregar más datos al dataset existente
Mantiene la estructura y hace el split automáticamente
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict


def get_file_type(filename):
    """Determinar el tipo de archivo"""
    ext = filename.lower().split('.')[-1]
    
    image_exts = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
    video_exts = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm']
    
    if ext in image_exts:
        return 'image'
    elif ext in video_exts:
        return 'video'
    return None


def analyze_current_data():
    """Analizar datos actuales"""
    print("\n" + "="*60)
    print("📊 DATOS ACTUALES")
    print("="*60)
    
    data_dir = Path('data')
    stats = {}
    
    for split in ['train', 'val']:
        split_dir = data_dir / split
        stats[split] = {}
        
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    files = [f for f in class_dir.iterdir() if f.is_file()]
                    stats[split][class_dir.name] = len(files)
    
    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        for cls, count in stats.get(split, {}).items():
            print(f"  {cls}: {count} archivos")
    
    return stats


def add_data_to_existing(source_dir, target_class=None, train_split=0.8):
    """
    Agregar datos a una clase existente
    
    Args:
        source_dir: Directorio con nuevos archivos
        target_class: Clase objetivo (None = detectar automáticamente)
        train_split: Proporción para train (default 0.8)
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"❌ El directorio {source_dir} no existe")
        return
    
    print(f"\n🔍 Buscando archivos en {source_dir}...")
    
    # Recolectar archivos
    files = []
    for file in source_path.rglob('*'):
        if file.is_file() and get_file_type(file.name):
            files.append(file)
    
    if not files:
        print("❌ No se encontraron archivos válidos")
        return
    
    # Agrupar por tipo
    images = [f for f in files if get_file_type(f.name) == 'image']
    videos = [f for f in files if get_file_type(f.name) == 'video']
    
    print(f"\n✓ Encontrados:")
    print(f"  Imágenes: {len(images)}")
    print(f"  Videos: {len(videos)}")
    
    # Si no se especifica clase, detectar
    if target_class is None:
        print("\n📁 Clases disponibles:")
        train_dir = Path('data/train')
        classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
        
        for i, cls in enumerate(classes, 1):
            print(f"  {i}. {cls}")
        print(f"  {len(classes) + 1}. Crear nueva clase")
        
        try:
            choice = int(input("\nSelecciona clase destino (número): ").strip())
            
            if choice <= len(classes):
                target_class = classes[choice - 1]
            else:
                target_class = input("Nombre de la nueva clase: ").strip()
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Cancelado")
            return
    
    print(f"\n✓ Clase destino: {target_class}")
    
    # Crear directorios si no existen
    train_class_dir = Path('data/train') / target_class
    val_class_dir = Path('data/val') / target_class
    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)
    
    # Copiar archivos
    print(f"\n📦 Copiando archivos...")
    
    files_by_type = {
        'imagenes': images,
        'videos': videos
    }
    
    total_train = 0
    total_val = 0
    
    for type_name, file_list in files_by_type.items():
        if not file_list:
            continue
        
        num_files = len(file_list)
        num_train = int(num_files * train_split)
        
        for i, file in enumerate(file_list):
            # Evitar duplicados agregando timestamp
            dest_name = file.name
            
            if i < num_train:
                dest = train_class_dir / dest_name
                counter = 1
                while dest.exists():
                    name, ext = dest_name.rsplit('.', 1)
                    dest_name = f"{name}_{counter}.{ext}"
                    dest = train_class_dir / dest_name
                    counter += 1
                
                shutil.copy2(file, dest)
                total_train += 1
            else:
                dest = val_class_dir / dest_name
                counter = 1
                while dest.exists():
                    name, ext = dest_name.rsplit('.', 1)
                    dest_name = f"{name}_{counter}.{ext}"
                    dest = val_class_dir / dest_name
                    counter += 1
                
                shutil.copy2(file, dest)
                total_val += 1
    
    print(f"\n✅ Archivos agregados:")
    print(f"  Train: {total_train}")
    print(f"  Val: {total_val}")
    print(f"  Total: {total_train + total_val}")


def main():
    """Función principal"""
    print("\n" + "➕ "*20)
    print("   AGREGAR MÁS DATOS AL DATASET")
    print("➕ "*20)
    
    # Mostrar datos actuales
    analyze_current_data()
    
    print("\n" + "="*60)
    print("OPCIONES:")
    print("="*60)
    print("1. Agregar datos a clase existente")
    print("2. Ver estadísticas actuales")
    print("3. Salir")
    
    try:
        choice = input("\nSelecciona una opción (1-3): ").strip()
        
        if choice == '1':
            source = input("\nRuta del directorio con nuevos archivos: ").strip()
            
            if source:
                add_data_to_existing(source)
                
                # Mostrar nuevo estado
                print("\n" + "="*60)
                print("📊 NUEVO ESTADO")
                print("="*60)
                analyze_current_data()
                
                print("\n💡 Ahora puedes re-entrenar con:")
                print("   python train.py --epochs 30")
            else:
                print("❌ Ruta vacía")
        
        elif choice == '2':
            # Ya mostrado arriba
            pass
        
        elif choice == '3':
            print("\n👋 Hasta luego!")
            return
        
        else:
            print("\n❌ Opción inválida")
    
    except KeyboardInterrupt:
        print("\n\n👋 Cancelado")


if __name__ == "__main__":
    main()
