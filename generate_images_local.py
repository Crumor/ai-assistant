"""
Generador de imágenes LOCAL usando tu GPU
Usa Stable Diffusion ejecutándose en tu máquina
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from datetime import datetime


def check_system():
    """Verificar sistema"""
    print("\n" + "="*60)
    print("🔍 VERIFICANDO SISTEMA")
    print("="*60)
    
    print(f"✓ PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA disponible")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Memoria: {mem_gb:.1f} GB")
        
        if mem_gb < 4:
            print("\n⚠️  Advertencia: GPU con poca memoria")
            print("   Usa resolución baja (512x512)")
        
        return True
    else:
        print("❌ No hay GPU disponible")
        print("💡 Se usará CPU (será MUY lento, 5-10 minutos por imagen)")
        
        choice = input("\n¿Continuar con CPU? (s/n): ").strip().lower()
        return choice in ['s', 'si', 'sí', 'y', 'yes']


class LocalImageGenerator:
    """Generador 100% local usando tu GPU"""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """
        Inicializar generador local
        
        Modelos disponibles:
        - "runwayml/stable-diffusion-v1-5" (más rápido, 4GB RAM)
        - "stabilityai/stable-diffusion-2-1" (mejor calidad, 6GB RAM)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n🎨 Cargando modelo de generación...")
        print(f"📦 Modelo: {model_id}")
        print(f"📱 Device: {self.device}")
        print(f"\n⏳ Primera vez: descargará ~4-5 GB (puede tardar)")
        print("   Las siguientes veces será instantáneo\n")
        
        # Cargar pipeline
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
            )
            
            self.pipe = self.pipe.to(self.device)
            
            # Optimizaciones para GPU
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                print("✓ Optimizaciones de GPU activadas")
            
            print("✓ Modelo cargado y listo!\n")
            
        except Exception as e:
            print(f"\n❌ Error al cargar modelo: {e}")
            print("\n💡 Soluciones:")
            print("   1. Instalar dependencias: pip install diffusers transformers accelerate")
            print("   2. Verificar conexión a internet (primera descarga)")
            print("   3. Liberar memoria GPU si está llena")
            raise
    
    def generate(
        self,
        prompt,
        negative_prompt="low quality, blurry, distorted, ugly",
        width=512,
        height=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=None
    ):
        """
        Generar imagen localmente en tu GPU
        
        Args:
            prompt: Descripción en inglés
            negative_prompt: Qué evitar
            width/height: Resolución (512 recomendado para 4GB GPU)
            num_inference_steps: Calidad (20-50, más = mejor)
            guidance_scale: Literalidad (7-15)
            seed: Para reproducibilidad
        """
        print(f"🎨 Generando imagen LOCALMENTE en {self.device.upper()}...")
        print(f"📝 Prompt: {prompt}")
        print(f"⚙️  Resolución: {width}x{height}")
        print(f"⚙️  Pasos: {num_inference_steps}")
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        try:
            # Generar
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
            
            print("✓ Generación completada!")
            return result.images
        
        except torch.cuda.OutOfMemoryError:
            print("\n❌ Error: GPU sin memoria suficiente")
            print("💡 Soluciones:")
            print("   1. Reduce resolución: --width 448 --height 448")
            print("   2. Reduce steps: --steps 20")
            print("   3. Cierra otras aplicaciones que usen GPU")
            return None
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return None
    
    def save_image(self, image, output_dir="outputs/generated"):
        """Guardar imagen"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"local_generated_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        image.save(filepath)
        print(f"✓ Guardada: {filepath}")
        
        return filepath


def main():
    """Función principal"""
    print("\n" + "🎨 "*20)
    print("   GENERADOR LOCAL DE IMÁGENES")
    print("   100% en tu GPU - Sin APIs externas")
    print("🎨 "*20)
    
    # Verificar sistema
    if not check_system():
        print("\n❌ Cancelado")
        return
    
    # Verificar dependencias
    try:
        import diffusers
        import transformers
    except ImportError:
        print("\n❌ Faltan dependencias")
        print("\n💡 Instalar con:")
        print("   pip install diffusers transformers accelerate")
        return
    
    # Inicializar generador
    try:
        generator = LocalImageGenerator()
    except Exception as e:
        print(f"\n❌ No se pudo inicializar el generador")
        return
    
    print("="*60)
    print("📝 TIPS:")
    print("="*60)
    print("• Escribe en INGLÉS para mejores resultados")
    print("• Sé específico y descriptivo")
    print("• Ejemplos:")
    print("  'a beautiful cat, digital art, highly detailed'")
    print("  'mountain landscape at sunset, photorealistic'")
    print("="*60 + "\n")
    
    while True:
        try:
            prompt = input("📝 Describe la imagen (o 'exit'): ").strip()
            
            if prompt.lower() in ['exit', 'salir', 'quit']:
                print("\n👋 ¡Hasta luego!")
                break
            
            if not prompt:
                print("❌ Prompt vacío\n")
                continue
            
            # Configuración rápida
            print("\n⚙️  ¿Usar valores por defecto? (s/n):", end=" ")
            use_defaults = input().strip().lower() in ['s', 'si', 'sí', 'y', 'yes', '']
            
            if use_defaults:
                width, height = 512, 512
                steps = 30
                guidance = 7.5
            else:
                try:
                    width = int(input("  Ancho [512]: ") or 512)
                    height = int(input("  Alto [512]: ") or 512)
                    steps = int(input("  Pasos [30]: ") or 30)
                    guidance = float(input("  Guidance [7.5]: ") or 7.5)
                except ValueError:
                    width, height, steps, guidance = 512, 512, 30, 7.5
            
            # Generar
            print()
            images = generator.generate(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance
            )
            
            if images:
                # Guardar
                print()
                filepath = generator.save_image(images[0])
                
                print(f"\n✅ ¡Imagen generada LOCALMENTE!")
                print(f"📁 Ubicación: {filepath}")
                
                # Abrir
                open_choice = input("\n¿Abrir imagen? (s/n): ").strip().lower()
                if open_choice in ['s', 'si', 'sí', 'y', 'yes']:
                    os.system(f"xdg-open {filepath} 2>/dev/null || open {filepath} 2>/dev/null")
            
            # Continuar
            print()
            if input("¿Otra imagen? (s/n): ").strip().lower() not in ['s', 'si', 'sí', 'y', 'yes']:
                break
            print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Cancelado")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
