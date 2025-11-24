#!/usr/bin/env python3
"""
Test rápido: Verificar que el sistema aprende y aplica estilo
"""

import subprocess
from pathlib import Path
import sys

def test_style_transfer():
    print("\n" + "="*60)
    print("  🧪 TEST: Aprendizaje y Aplicación de Estilo")
    print("="*60 + "\n")
    
    # Verificar que existe el modelo
    model_path = Path('models/checkpoints/best_model.pt')
    if not model_path.exists():
        print("❌ Error: No se encuentra el modelo entrenado")
        print("   Ejecuta primero: python quick_start.py")
        return False
    
    # Verificar dataset
    data_dir = Path('data/train/imagenes')
    images = list(data_dir.glob('*.jpg'))
    if len(images) < 5:
        print(f"⚠️  Advertencia: Solo {len(images)} imágenes en dataset")
        print("   Recomendado: 50+ para mejores resultados")
    
    # Buscar imagen de prueba
    test_image = None
    if Path('Pasted image.png').exists():
        test_image = 'Pasted image.png'
    elif images:
        test_image = str(images[0])
    else:
        print("❌ No hay imágenes para probar")
        return False
    
    print(f"📸 Imagen de prueba: {test_image}")
    print(f"📚 Dataset: {len(images)} imágenes")
    print(f"🤖 Modelo: {model_path}")
    print("\n🚀 Iniciando transferencia de estilo...\n")
    
    # Ejecutar apply_style.py
    cmd = [
        sys.executable, 'apply_style.py',
        '--input', test_image,
        '--output', 'outputs/test_styled.jpg',
        '--iterations', '100',  # Rápido para test
        '--max-learn', '20'     # Pocas imágenes para test
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        print("\n" + "="*60)
        print("  ✅ TEST EXITOSO")
        print("="*60)
        print("\n📁 Resultado guardado en: outputs/test_styled.jpg")
        print("\n💡 Para mejor calidad, usa:")
        print("   python apply_style.py --input tu_imagen.jpg --iterations 300 --max-learn 50")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en la ejecución: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == '__main__':
    success = test_style_transfer()
    sys.exit(0 if success else 1)
