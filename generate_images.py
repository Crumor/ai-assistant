"""
Generador de imágenes con IA usando Stable Diffusion
Requiere: pip install diffusers transformers accelerate torch pillow
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
from datetime import datetime


class ImageGenerator:
    """Generador de imágenes con IA"""
    
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1", device=None):
        """
        Inicializar generador
        
        Args:
            model_id: ID del modelo en Hugging Face
            device: 'cuda' o 'cpu' (None = auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🎨 Inicializando generador de imágenes...")
        print(f"📱 Device: {self.device}")
        
        # Cargar pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,  # Opcional: remover para contenido sensible
        )
        
        # Optimizar scheduler para mejor calidad y velocidad
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Mover a GPU
        self.pipe = self.pipe.to(self.device)
        
        # Optimizaciones si hay GPU
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            # self.pipe.enable_xformers_memory_efficient_attention()  # Descomentar si tienes xformers
        
        print("✓ Generador listo!\n")
    
    def generate(
        self,
        prompt,
        negative_prompt="low quality, blurry, distorted, ugly, bad anatomy",
        num_images=1,
        width=512,
        height=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=None
    ):
        """
        Generar imagen desde texto
        
        Args:
            prompt: Descripción de la imagen en inglés
            negative_prompt: Qué evitar en la imagen
            num_images: Cantidad de imágenes a generar
            width: Ancho (múltiplo de 8, recomendado: 512-768)
            height: Alto (múltiplo de 8, recomendado: 512-768)
            num_inference_steps: Pasos de generación (más = mejor calidad, más lento)
            guidance_scale: Qué tan literal seguir el prompt (7-15)
            seed: Semilla para reproducibilidad
        
        Returns:
            Lista de imágenes PIL
        """
        # Configurar semilla si se proporciona
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print(f"🎨 Generando {num_images} imagen(es)...")
        print(f"📝 Prompt: {prompt}")
        print(f"⚙️  Pasos: {num_inference_steps}, Guidance: {guidance_scale}")
        
        # Generar
        with torch.autocast(self.device):
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
        
        return result.images
    
    def save_images(self, images, output_dir="outputs/generated", prefix="ai_generated"):
        """Guardar imágenes generadas"""
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, image in enumerate(images):
            filename = f"{prefix}_{timestamp}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            saved_paths.append(filepath)
            print(f"✓ Guardada: {filepath}")
        
        return saved_paths


def main():
    """Función principal"""
    print("\n" + "🎨 "*20)
    print("   GENERADOR DE IMÁGENES CON IA")
    print("🎨 "*20 + "\n")
    
    # Verificar GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU detectada: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️  No hay GPU, se usará CPU (será MUY lento)")
        print("💡 Recomendado: Usar GPU o servicios en la nube\n")
    
    # Inicializar generador
    try:
        generator = ImageGenerator()
    except Exception as e:
        print(f"\n❌ Error al cargar modelo: {e}")
        print("\n💡 Instala dependencias:")
        print("   pip install diffusers transformers accelerate")
        return
    
    print("="*60)
    print("📝 INSTRUCCIONES:")
    print("="*60)
    print("• Escribe en INGLÉS para mejores resultados")
    print("• Sé específico y descriptivo")
    print("• Ejemplos:")
    print("  - 'a beautiful sunset over mountains, digital art'")
    print("  - 'portrait of a cat wearing sunglasses, photorealistic'")
    print("  - 'futuristic city with flying cars, cyberpunk style'")
    print("="*60 + "\n")
    
    while True:
        try:
            prompt = input("📝 Describe la imagen (o 'exit' para salir): ").strip()
            
            if prompt.lower() in ['exit', 'salir', 'quit']:
                print("\n👋 ¡Hasta luego!")
                break
            
            if not prompt:
                print("❌ Prompt vacío, intenta de nuevo\n")
                continue
            
            # Preguntar configuración
            print("\n⚙️  Configuración (presiona Enter para usar valores por defecto):")
            
            try:
                num_images = input("  Cantidad de imágenes [1]: ").strip()
                num_images = int(num_images) if num_images else 1
                
                steps = input("  Pasos de inferencia [30]: ").strip()
                steps = int(steps) if steps else 30
                
                guidance = input("  Guidance scale [7.5]: ").strip()
                guidance = float(guidance) if guidance else 7.5
            except ValueError:
                print("⚠️  Valor inválido, usando valores por defecto")
                num_images = 1
                steps = 30
                guidance = 7.5
            
            # Generar
            print()
            images = generator.generate(
                prompt=prompt,
                num_images=num_images,
                num_inference_steps=steps,
                guidance_scale=guidance
            )
            
            # Guardar
            print()
            saved_paths = generator.save_images(images)
            
            print(f"\n✅ ¡Listo! {len(images)} imagen(es) generada(s)")
            print(f"📁 Guardadas en: outputs/generated/\n")
            
            # Preguntar si continuar
            continue_choice = input("¿Generar otra imagen? (s/n): ").strip().lower()
            if continue_choice not in ['s', 'si', 'sí', 'y', 'yes']:
                print("\n👋 ¡Hasta luego!")
                break
            print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Cancelado por usuario")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
