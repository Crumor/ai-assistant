#!/usr/bin/env python3
"""
Script para organizar automáticamente imágenes y videos en la estructura correcta
Uso: python organize_data.py <directorio_con_archivos>
"""

import os
import sys
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


def organize_from_directory(source_dir, output_dir='data', train_split=0.8):
    """
    Organizar archivos de un directorio a la estructura train/val
    
    Args:
        source_dir: Directorio con archivos (puede tener subdirectorios o no)
        output_dir: Directorio de salida (default: data)
        train_split: Porcentaje para entrenamiento (default: 0.8)
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"❌ El directorio {source_dir} no existe")
        return
    
    # Detectar estructura
    print(f"\n🔍 Analizando {source_dir}...")
    
    # Buscar todos los archivos
    files_by_class = defaultdict(list)
    
    # Caso 1: Ya tiene subdirectorios (clases)
    subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if subdirs:
        print(f"✓ Detectadas {len(subdirs)} clases (subdirectorios)")
        for subdir in subdirs:
            class_name = subdir.name
            files = []
            
            for file in subdir.rglob('*'):
                if file.is_file() and get_file_type(file.name):
                    files.append(file)
            
            if files:
                files_by_class[class_name] = files
                print(f"  - {class_name}: {len(files)} archivos")
    
    # Caso 2: Todos los archivos en el mismo nivel
    else:
        print("ℹ️  No se detectaron subdirectorios")
        all_files = [f for f in source_path.iterdir() 
                     if f.is_file() and get_file_type(f.name)]
        
        if all_files:
            print(f"✓ Encontrados {len(all_files)} archivos")
            
            # Agrupar por tipo
            images = [f for f in all_files if get_file_type(f.name) == 'image']
            videos = [f for f in all_files if get_file_type(f.name) == 'video']
            
            if images:
                files_by_class['imagenes'] = images
                print(f"  - imagenes: {len(images)} archivos")
            if videos:
                files_by_class['videos'] = videos
                print(f"  - videos: {len(videos)} archivos")
    
    if not files_by_class:
        print("\n❌ No se encontraron archivos válidos (imágenes o videos)")
        print("\nFormatos soportados:")
        print("  Imágenes: .jpg, .jpeg, .png, .gif, .bmp, .webp")
        print("  Videos: .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm")
        return
    
    # Crear estructura de salida
    print(f"\n📁 Creando estructura en {output_dir}/...")
    train_dir = Path(output_dir) / 'train'
    val_dir = Path(output_dir) / 'val'
    
    # Copiar/mover archivos
    total_train = 0
    total_val = 0
    
    for class_name, files in files_by_class.items():
        # Crear directorios de clase
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Calcular split
        num_files = len(files)
        num_train = int(num_files * train_split)
        
        # Copiar archivos
        for i, file in enumerate(files):
            if i < num_train:
                dest = train_class_dir / file.name
                shutil.copy2(file, dest)
                total_train += 1
            else:
                dest = val_class_dir / file.name
                shutil.copy2(file, dest)
                total_val += 1
        
        print(f"✓ {class_name}:")
        print(f"    Train: {num_train} archivos")
        print(f"    Val: {num_files - num_train} archivos")
    
    print(f"\n{'='*60}")
    print(f"✅ Organización completada!")
    print(f"{'='*60}")
    print(f"Total Train: {total_train} archivos")
    print(f"Total Val: {total_val} archivos")
    print(f"\nEstructura creada:")
    print(f"  {output_dir}/train/ ({len(files_by_class)} clases)")
    print(f"  {output_dir}/val/ ({len(files_by_class)} clases)")
    print(f"\n💡 Ahora puedes ejecutar: python train.py")


def scan_current_data():
    """Escanear y mostrar el estado actual de data/"""
    print("\n" + "="*60)
    print("📊 Estado actual de data/")
    print("="*60)
    
    data_dir = Path('data')
    
    for split in ['train', 'val']:
        split_dir = data_dir / split
        
        if not split_dir.exists():
            print(f"\n❌ {split}/ no existe")
            continue
        
        classes = [d for d in split_dir.iterdir() if d.is_dir()]
        
        if not classes:
            print(f"\n⚠️  {split}/ está vacío")
            continue
        
        print(f"\n✓ {split}/ ({len(classes)} clases):")
        
        total_files = 0
        for cls in classes:
            files = list(cls.glob('*'))
            files = [f for f in files if f.is_file() and get_file_type(f.name)]
            total_files += len(files)
            print(f"    {cls.name}: {len(files)} archivos")
        
        print(f"  Total: {total_files} archivos")


def main():
    """Función principal"""
    print("\n" + "🗂️  "*20)
    print("   ORGANIZADOR DE DATOS - AI VISION ASSISTANT")
    print("🗂️  "*20)
    
    # Primero mostrar estado actual
    if os.path.exists('data'):
        scan_current_data()
    
    print("\n" + "="*60)
    print("OPCIONES:")
    print("="*60)
    print("1. Organizar archivos desde un directorio")
    print("2. Crear estructura de ejemplo")
    print("3. Salir")
    
    try:
        choice = input("\nSelecciona una opción (1-3): ").strip()
        
        if choice == '1':
            print("\n💡 Indica el directorio donde están tus imágenes/videos")
            print("   Pueden estar todos juntos o ya organizados en carpetas")
            source = input("\nRuta del directorio: ").strip()
            
            if source:
                organize_from_directory(source)
            else:
                print("❌ Ruta vacía")
        
        elif choice == '2':
            print("\n📝 Creando estructura de ejemplo...")
            
            # Crear directorios de ejemplo
            for split in ['train', 'val']:
                for cls in ['gatos', 'perros', 'pajaros']:
                    path = Path(f'data/{split}/{cls}')
                    path.mkdir(parents=True, exist_ok=True)
            
            print("\n✓ Estructura creada:")
            print("data/")
            print("  train/")
            print("    gatos/")
            print("    perros/")
            print("    pajaros/")
            print("  val/")
            print("    gatos/")
            print("    perros/")
            print("    pajaros/")
            print("\n💡 Ahora copia tus archivos en estas carpetas")
        
        elif choice == '3':
            print("\n👋 Hasta luego!")
            return
        
        else:
            print("\n❌ Opción inválida")
    
    except KeyboardInterrupt:
        print("\n\n👋 Cancelado por el usuario")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Modo directo desde línea de comandos
        source_dir = sys.argv[1]
        organize_from_directory(source_dir)
    else:
        # Modo interactivo
        main()
