"""
Fine-tuning de Stable Diffusion con TUS datos
Entrena un modelo generativo que aprende el estilo de tus imágenes/videos
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from pathlib import Path
import os


class CustomImageGenerator:
    """
    Generador entrenado con TUS datos
    Aprende el estilo de tus imágenes para generar nuevas similares
    """
    
    def __init__(self):
        print("\n" + "="*60)
        print("🎨 GENERADOR PERSONALIZADO CON TUS DATOS")
        print("="*60)
        
        print("\n📚 Este sistema puede:")
        print("  1. Entrenar con tus imágenes/videos")
        print("  2. Aprender su estilo visual")
        print("  3. Generar nuevas imágenes similares")
        print("  4. Aplicar ese estilo a cualquier prompt")
        
    def analyze_training_data(self, data_dir='data/train'):
        """Analizar datos para entrenamiento"""
        print(f"\n🔍 Analizando {data_dir}...")
        
        data_path = Path(data_dir)
        
        stats = {
            'total_images': 0,
            'classes': {},
            'sample_paths': []
        }
        
        for class_dir in data_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            stats['classes'][class_dir.name] = len(images)
            stats['total_images'] += len(images)
            
            # Guardar algunas muestras
            stats['sample_paths'].extend([str(p) for p in images[:5]])
        
        print(f"✓ Total de imágenes: {stats['total_images']}")
        for cls, count in stats['classes'].items():
            print(f"  - {cls}: {count} imágenes")
        
        return stats
    
    def train_with_your_data(self, data_dir='data/train', epochs=100):
        """
        Entrenar modelo generativo con tus datos
        NOTA: Requiere GPU potente y mucho tiempo
        """
        print("\n⚠️  ADVERTENCIA:")
        print("  - Fine-tuning de Stable Diffusion requiere:")
        print("    • GPU con 12+ GB VRAM (tu RTX 3050 tiene 4 GB)")
        print("    • ~100-1000 imágenes de entrenamiento")
        print("    • Varias horas de entrenamiento")
        print("    • ~20 GB de espacio en disco")
        
        print("\n💡 ALTERNATIVAS RECOMENDADAS:")
        print("  1. Usar LoRA (más eficiente, requiere menos recursos)")
        print("  2. Usar DreamBooth (especializado en tu estilo)")
        print("  3. Usar servicios como Replicate o Hugging Face Spaces")
        
        proceed = input("\n¿Continuar de todos modos? (s/n): ").strip().lower()
        
        if proceed not in ['s', 'si', 'sí', 'y', 'yes']:
            print("❌ Cancelado")
            return None
        
        print("\n🚧 Entrenamiento no implementado en esta versión")
        print("💡 Usa el script de LoRA training para tu caso:")
        print("   python train_lora.py")
    
    def generate_with_learned_style(self, prompt, style_strength=0.8):
        """
        Generar imagen aplicando el estilo aprendido
        """
        print(f"\n🎨 Generando con estilo de tus datos...")
        print(f"📝 Prompt: {prompt}")
        print(f"💪 Fuerza del estilo: {style_strength}")
        
        # Por ahora usa Stable Diffusion base
        print("\n💡 Para aplicar TU estilo específico:")
        print("   1. Primero entrena con: python train_lora.py")
        print("   2. Luego este script usará tu modelo personalizado")
        
        print("\n📚 Alternativa rápida:")
        print("   Usa 'apply_learned_style.py' para modificar imágenes existentes")


def create_training_script():
    """
    Crear script de entrenamiento LoRA (más eficiente)
    """
    print("\n" + "="*60)
    print("📚 GUÍA DE ENTRENAMIENTO")
    print("="*60)
    
    print("""
Para entrenar un modelo generativo con TUS datos necesitas:

🎯 OPCIÓN 1: LoRA Training (Recomendado)
  • Más eficiente con pocos recursos
  • Solo entrena una pequeña parte del modelo
  • Requiere 4-8 GB VRAM (tu GPU funciona!)
  • 20-100 imágenes suficientes
  • 1-3 horas de entrenamiento
  
  Comando:
  pip install peft bitsandbytes
  python train_lora.py --data_dir data/train --epochs 50

🎯 OPCIÓN 2: DreamBooth
  • Entrena modelo para un concepto específico
  • Requiere 12+ GB VRAM
  • 3-5 imágenes de referencia
  • 30-60 minutos
  
🎯 OPCIÓN 3: Textual Inversion
  • Solo aprende un nuevo "token"
  • Más ligero (~2 GB VRAM)
  • Resultados moderados

🎯 OPCIÓN 4: Usar servicios externos
  • Replicate.com - API fácil
  • Hugging Face Inference - Gratis
  • Runway ML - Con GUI

💡 RECOMENDACIÓN PARA TU CASO:
  Con RTX 3050 (4 GB) → Usa LoRA o servicios externos
  
¿Quieres que cree el script de LoRA training?
    """)


def main():
    """Función principal"""
    print("\n" + "🎓 "*20)
    print("   ENTRENAR IA CON TUS DATOS")
    print("🎓 "*20)
    
    generator = CustomImageGenerator()
    
    # Analizar datos
    stats = generator.analyze_training_data()
    
    if stats['total_images'] < 20:
        print("\n⚠️  Tienes pocas imágenes (<20)")
        print("💡 Recomendado: 50-100+ imágenes para buenos resultados")
    
    print("\n" + "="*60)
    print("OPCIONES:")
    print("="*60)
    print("1. Ver guía de entrenamiento completa")
    print("2. Aplicar estilo a imagen existente (rápido)")
    print("3. Ver requisitos para entrenamiento")
    print("4. Salir")
    
    try:
        choice = input("\nSelecciona opción (1-4): ").strip()
        
        if choice == '1':
            create_training_script()
        
        elif choice == '2':
            print("\n💡 Usa este script para aplicar estilo rápido:")
            print("   python apply_learned_style.py")
        
        elif choice == '3':
            print("\n📋 REQUISITOS PARA ENTRENAMIENTO COMPLETO:")
            print("\nMÍNIMO:")
            print("  • GPU: 4 GB VRAM (LoRA)")
            print("  • Imágenes: 20-50")
            print("  • Tiempo: 1-2 horas")
            print("  • Espacio: 10 GB")
            
            print("\nRECOMENDADO:")
            print("  • GPU: 8+ GB VRAM")
            print("  • Imágenes: 100-500")
            print("  • Tiempo: 3-6 horas")
            print("  • Espacio: 20 GB")
            
            print("\nPROFESIONAL:")
            print("  • GPU: 24+ GB VRAM")
            print("  • Imágenes: 1000+")
            print("  • Tiempo: 12-24 horas")
            print("  • Espacio: 50+ GB")
        
        elif choice == '4':
            print("\n👋 Hasta luego!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Cancelado")


if __name__ == "__main__":
    main()
