#!/usr/bin/env python3
"""
Test básico para verificar que los módulos se importan correctamente
"""

import sys


def test_imports():
    """Verificar que todos los imports funcionan"""
    print("🧪 Verificando imports...\n")
    
    errors = []
    
    # Test 1: Data loader
    print("1. Probando src.data.data_loader...")
    try:
        from src.data.data_loader import VideoImageDataset, create_dataloaders
        print("   ✓ data_loader OK")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        errors.append(("data_loader", str(e)))
    
    # Test 2: Models
    print("2. Probando src.models.vision_model...")
    try:
        from src.models.vision_model import VisionModel, MultiModalModel
        print("   ✓ vision_model OK")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        errors.append(("vision_model", str(e)))
    
    # Test 3: Virtual Try-On
    print("3. Probando src.inference.virtual_tryon...")
    try:
        from src.inference.virtual_tryon import VirtualTryOn, StyleTransferModel
        print("   ✓ virtual_tryon OK")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        errors.append(("virtual_tryon", str(e)))
    
    # Test 4: Trainer
    print("4. Probando src.training.trainer...")
    try:
        from src.training.trainer import Trainer, create_optimizer
        print("   ✓ trainer OK")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        errors.append(("trainer", str(e)))
    
    print("\n" + "="*60)
    if len(errors) == 0:
        print("✅ TODOS LOS TESTS PASARON")
        print("="*60)
        return 0
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        print("="*60)
        print("\nErrores encontrados:")
        for module, error in errors:
            print(f"\n{module}:")
            print(f"  {error}")
        return 1


def test_structure():
    """Verificar estructura de archivos"""
    print("\n📁 Verificando estructura de archivos...\n")
    
    import os
    
    required_files = [
        'src/data/__init__.py',
        'src/data/data_loader.py',
        'src/models/vision_model.py',
        'src/inference/virtual_tryon.py',
        'src/training/trainer.py',
        'virtual_tryon.py',
        'VIRTUAL_TRYON.md',
        'README.md',
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (falta)")
            missing.append(file)
    
    print("\n" + "="*60)
    if len(missing) == 0:
        print("✅ ESTRUCTURA COMPLETA")
    else:
        print(f"⚠️  Faltan {len(missing)} archivos")
    print("="*60)
    
    return len(missing)


def main():
    """Función principal"""
    print("\n" + "🧪 "*20)
    print("   TEST DE VERIFICACIÓN - VIRTUAL TRY-ON")
    print("🧪 "*20 + "\n")
    
    # Test imports
    import_result = test_imports()
    
    # Test structure
    structure_result = test_structure()
    
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    
    if import_result == 0 and structure_result == 0:
        print("\n✅ Sistema listo para usar!")
        print("\n💡 Próximos pasos:")
        print("1. python quick_start.py    # Entrenar modelo base")
        print("2. python virtual_tryon.py   # Usar Virtual Try-On")
        print("3. python ejemplos_virtual_tryon.py  # Ver ejemplos")
        return 0
    else:
        print("\n⚠️  Hay problemas que resolver")
        if import_result != 0:
            print("\n💡 Para resolver errores de import:")
            print("   pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
