#!/usr/bin/env python3
"""
Script de inicio rápido - Detecta automáticamente tu configuración y entrena
Ejecutar con: python quick_start.py
"""

import os
import sys
import torch


def check_environment():
    """Verificar el entorno"""
    print("\n" + "="*60)
    print("🔍 Verificando entorno...")
    print("="*60 + "\n")
    
    # Python version
    print(f"✓ Python {sys.version.split()[0]}")
    
    # PyTorch
    print(f"✓ PyTorch {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA disponible")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  CUDA no disponible - se usará CPU (será más lento)")
    
    print()


def check_data_structure():
    """Verificar estructura de datos"""
    print("="*60)
    print("📁 Verificando datos...")
    print("="*60 + "\n")
    
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    issues = []
    
    # Verificar train
    if not os.path.exists(train_dir):
        issues.append(f"❌ Falta directorio: {train_dir}")
    else:
        classes = [d for d in os.listdir(train_dir) 
                  if os.path.isdir(os.path.join(train_dir, d))]
        
        if len(classes) == 0:
            issues.append(f"❌ No hay clases (carpetas) en {train_dir}")
        else:
            print(f"✓ Directorio de entrenamiento: {train_dir}")
            print(f"  Clases encontradas: {len(classes)}")
            
            # Contar imágenes por clase
            for cls in classes[:5]:  # Mostrar solo primeras 5
                cls_path = os.path.join(train_dir, cls)
                files = [f for f in os.listdir(cls_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi'))]
                print(f"    - {cls}: {len(files)} archivos")
            
            if len(classes) > 5:
                print(f"    ... y {len(classes) - 5} clases más")
    
    # Verificar val
    if not os.path.exists(val_dir):
        issues.append(f"⚠️  Recomendado crear: {val_dir}")
        print(f"\n⚠️  No se encontró {val_dir}")
        print("   Se usará 20% de train para validación automáticamente")
    else:
        val_classes = [d for d in os.listdir(val_dir) 
                      if os.path.isdir(os.path.join(val_dir, d))]
        print(f"\n✓ Directorio de validación: {val_dir}")
        print(f"  Clases: {len(val_classes)}")
    
    print()
    
    return len(issues) == 0, issues


def suggest_config():
    """Sugerir configuración basada en los recursos"""
    print("="*60)
    print("⚙️  Configuración sugerida")
    print("="*60 + "\n")
    
    if torch.cuda.is_available():
        # Con GPU
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_memory >= 16:
            batch_size = 64
            num_workers = 8
        elif gpu_memory >= 8:
            batch_size = 32
            num_workers = 4
        else:
            batch_size = 16
            num_workers = 2
        
        print(f"✓ Con GPU ({gpu_memory:.1f} GB):")
    else:
        # Sin GPU
        batch_size = 16
        num_workers = 2
        print(f"✓ Sin GPU (CPU):")
    
    print(f"  - Batch size: {batch_size}")
    print(f"  - Workers: {num_workers}")
    print(f"  - Épocas recomendadas: 20-50")
    print(f"  - Learning rate: 0.001")
    print()
    
    return batch_size, num_workers


def create_validation_split():
    """Crear split de validación si no existe"""
    train_dir = 'data/train'
    val_dir = 'data/val'
    
    if not os.path.exists(val_dir) and os.path.exists(train_dir):
        print("="*60)
        print("🔄 Creando split de validación...")
        print("="*60 + "\n")
        
        import shutil
        from random import sample
        
        os.makedirs(val_dir, exist_ok=True)
        
        classes = [d for d in os.listdir(train_dir) 
                  if os.path.isdir(os.path.join(train_dir, d))]
        
        for cls in classes:
            src_cls_dir = os.path.join(train_dir, cls)
            dst_cls_dir = os.path.join(val_dir, cls)
            os.makedirs(dst_cls_dir, exist_ok=True)
            
            # Obtener archivos
            files = [f for f in os.listdir(src_cls_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi'))]
            
            # Mover 20% a validación
            num_val = max(1, int(len(files) * 0.2))
            val_files = sample(files, num_val)
            
            for f in val_files:
                src = os.path.join(src_cls_dir, f)
                dst = os.path.join(dst_cls_dir, f)
                shutil.move(src, dst)
            
            print(f"✓ {cls}: {num_val} archivos movidos a validación")
        
        print(f"\n✓ Split de validación creado en {val_dir}\n")


def main():
    """Función principal"""
    print("\n" + "🤖 "*20)
    print("   AI VISION ASSISTANT - QUICK START")
    print("🤖 "*20 + "\n")
    
    # 1. Verificar entorno
    check_environment()
    
    # 2. Verificar datos
    data_ok, issues = check_data_structure()
    
    if not data_ok:
        print("="*60)
        print("❌ Problemas encontrados:")
        print("="*60 + "\n")
        for issue in issues:
            print(issue)
        
        print("\n💡 Organiza tus datos así:")
        print("\ndata/")
        print("  train/")
        print("    clase1/")
        print("      imagen1.jpg")
        print("      imagen2.jpg")
        print("      video1.mp4")
        print("    clase2/")
        print("      imagen1.jpg")
        print("  val/")
        print("    clase1/")
        print("      imagen1.jpg")
        print("    clase2/")
        print("      imagen1.jpg")
        print()
        return
    
    # 3. Crear split de validación si es necesario
    create_validation_split()
    
    # 4. Sugerir configuración
    batch_size, num_workers = suggest_config()
    
    # 5. Ejecutar entrenamiento
    print("="*60)
    print("🚀 Iniciando entrenamiento...")
    print("="*60 + "\n")
    
    cmd = f"python train.py --batch-size {batch_size} --num-workers {num_workers} --epochs 20"
    
    print(f"Ejecutando: {cmd}\n")
    print("Presiona Ctrl+C para detener el entrenamiento\n")
    
    import time
    time.sleep(2)
    
    os.system(cmd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
        sys.exit(0)
