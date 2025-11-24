#!/usr/bin/env python3
"""
Script de diagnóstico completo del proyecto
"""

import os
import sys
from pathlib import Path


def print_section(title):
    """Imprimir sección"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def check_environment():
    """Verificar entorno Python"""
    print_section("🔧 ENTORNO")
    
    print(f"✓ Python: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA: {'Disponible' if torch.cuda.is_available() else 'No disponible (CPU)'}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Memoria: {mem_gb:.1f} GB")
    except ImportError:
        print("❌ PyTorch no instalado")
        return False
    
    try:
        import torchvision
        print(f"✓ Torchvision: {torchvision.__version__}")
    except ImportError:
        print("❌ Torchvision no instalado")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("⚠️  OpenCV no instalado")
    
    return True


def check_data_structure():
    """Verificar estructura de datos"""
    print_section("📁 ESTRUCTURA DE DATOS")
    
    data_dir = Path('data')
    
    if not data_dir.exists():
        print("❌ Directorio 'data/' no existe")
        return False
    
    issues = []
    total_files = 0
    
    for split in ['train', 'val']:
        split_dir = data_dir / split
        
        if not split_dir.exists():
            issues.append(f"❌ {split}/ no existe")
            continue
        
        classes = [d for d in split_dir.iterdir() if d.is_dir()]
        
        if not classes:
            issues.append(f"❌ {split}/ está vacío")
            continue
        
        print(f"\n✓ {split.upper()}/")
        print(f"  Clases: {len(classes)}")
        
        split_total = 0
        for cls in sorted(classes):
            files = [f for f in cls.iterdir() if f.is_file()]
            split_total += len(files)
            
            # Detectar tipos
            images = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']]
            videos = [f for f in files if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
            
            print(f"    {cls.name}:")
            if images:
                print(f"      - Imágenes: {len(images)}")
            if videos:
                print(f"      - Videos: {len(videos)}")
            print(f"      - Total: {len(files)}")
        
        print(f"  TOTAL {split.upper()}: {split_total} archivos")
        total_files += split_total
    
    print(f"\n📊 TOTAL GENERAL: {total_files} archivos")
    
    if issues:
        print("\n⚠️  PROBLEMAS DETECTADOS:")
        for issue in issues:
            print(f"  {issue}")
        return False
    
    if total_files == 0:
        print("\n❌ No hay archivos en los datasets")
        return False
    
    return True


def check_project_structure():
    """Verificar estructura del proyecto"""
    print_section("📂 ESTRUCTURA DEL PROYECTO")
    
    required_dirs = {
        'src': 'Código fuente',
        'src/models': 'Modelos',
        'src/data': 'Data loaders',
        'src/training': 'Scripts de entrenamiento',
        'models/checkpoints': 'Checkpoints de modelos',
        'logs': 'Logs de entrenamiento',
        'notebooks': 'Jupyter notebooks',
    }
    
    required_files = {
        'train.py': 'Script principal de entrenamiento',
        'quick_start.py': 'Script de inicio rápido',
        'organize_data.py': 'Organizador de datos',
        'requirements.txt': 'Dependencias',
        'setup.sh': 'Script de setup',
        'README.md': 'Documentación',
        'GUIA_RAPIDA.md': 'Guía rápida',
    }
    
    all_ok = True
    
    print("\nDirectorios:")
    for dir_path, description in required_dirs.items():
        exists = Path(dir_path).exists()
        status = "✓" if exists else "❌"
        print(f"  {status} {dir_path:30} ({description})")
        if not exists:
            all_ok = False
    
    print("\nArchivos principales:")
    for file_path, description in required_files.items():
        exists = Path(file_path).exists()
        status = "✓" if exists else "❌"
        print(f"  {status} {file_path:30} ({description})")
        if not exists:
            all_ok = False
    
    return all_ok


def check_model_files():
    """Verificar archivos de modelos"""
    print_section("🤖 MODELOS Y CÓDIGO")
    
    files_to_check = [
        ('src/models/vision_model.py', 'Modelo de visión'),
        ('src/data/data_loader.py', 'Data loader'),
        ('src/training/trainer.py', 'Trainer'),
        ('src/utils/visualization.py', 'Visualización'),
    ]
    
    all_ok = True
    
    for file_path, description in files_to_check:
        exists = Path(file_path).exists()
        status = "✓" if exists else "❌"
        print(f"  {status} {file_path:35} ({description})")
        if not exists:
            all_ok = False
    
    return all_ok


def suggest_next_steps(env_ok, data_ok, structure_ok, models_ok):
    """Sugerir próximos pasos"""
    print_section("🎯 PRÓXIMOS PASOS")
    
    if not env_ok:
        print("\n1. ❌ Instalar dependencias:")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
        return
    
    if not structure_ok or not models_ok:
        print("\n1. ⚠️  Hay archivos faltantes en el proyecto")
        print("   Verifica la estructura arriba")
        return
    
    if not data_ok:
        print("\n1. ❌ Organizar datos:")
        print("   python organize_data.py")
        print("   O manualmente en data/train/ y data/val/")
        return
    
    # Todo está OK
    print("\n✅ ¡Todo listo para entrenar!\n")
    print("Opciones:")
    print("\n1. Inicio rápido (recomendado):")
    print("   python quick_start.py")
    print("\n2. Entrenamiento manual:")
    print("   python train.py --epochs 20 --batch-size 32")
    print("\n3. Explorar en Jupyter:")
    print("   jupyter notebook notebooks/01_getting_started.ipynb")
    
    # Sugerencias basadas en datos
    print("\n💡 Recomendaciones:")
    
    data_dir = Path('data')
    train_files = sum(1 for _ in (data_dir / 'train').rglob('*') if _.is_file())
    
    if train_files < 100:
        print("  • Tienes pocos datos, usa --pretrained (transfer learning)")
        print("  • Considera data augmentation (ya está activado)")
    
    if Path('data/train/videos').exists():
        print("  • Detectados videos: el procesamiento será más lento")
        print("  • Reduce batch size si tienes problemas de memoria")


def main():
    """Función principal"""
    print("\n" + "🔍 "*20)
    print("   DIAGNÓSTICO DEL PROYECTO")
    print("🔍 "*20)
    
    os.chdir('/var/www/html/ai-assistant')
    
    # Ejecutar checks
    env_ok = check_environment()
    data_ok = check_data_structure()
    structure_ok = check_project_structure()
    models_ok = check_model_files()
    
    # Resumen final
    print_section("📋 RESUMEN")
    
    checks = [
        ("Entorno Python", env_ok),
        ("Estructura de datos", data_ok),
        ("Estructura del proyecto", structure_ok),
        ("Archivos de código", models_ok),
    ]
    
    for name, status in checks:
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")
    
    all_ok = all([env_ok, data_ok, structure_ok, models_ok])
    
    if all_ok:
        print("\n🎉 TODO ESTÁ PERFECTO!")
    else:
        print("\n⚠️  HAY ALGUNOS PROBLEMAS")
    
    # Sugerencias
    suggest_next_steps(env_ok, data_ok, structure_ok, models_ok)
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error en diagnóstico: {e}")
        import traceback
        traceback.print_exc()
