"""
Script simplificado para generar imágenes usando una API gratuita
No requiere GPU - Usa servicios en la nube
"""

import requests
import json
from PIL import Image
from io import BytesIO
import os
from datetime import datetime


class SimpleImageGenerator:
    """Generador simple usando APIs gratuitas"""
    
    def __init__(self, api="pollinations"):
        """
        Inicializar generador
        
        APIs disponibles:
        - 'pollinations': Gratuita, sin API key
        - 'replicate': Requiere API key pero mejor calidad
        """
        self.api = api
        print(f"🎨 Usando API: {api}")
        print("✓ Listo para generar!\n")
    
    def generate_pollinations(self, prompt, width=512, height=512):
        """Generar con Pollinations.ai (gratis, sin registro)"""
        # URL de la API
        url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
        params = {
            "width": width,
            "height": height,
            "nologo": "true"
        }
        
        print(f"🎨 Generando imagen...")
        print(f"📝 Prompt: {prompt}")
        
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            
            # Cargar imagen
            image = Image.open(BytesIO(response.content))
            return [image]
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def generate(self, prompt, width=512, height=512):
        """Generar imagen con la API configurada"""
        if self.api == "pollinations":
            return self.generate_pollinations(prompt, width, height)
        else:
            print(f"❌ API '{self.api}' no soportada")
            return None
    
    def save_images(self, images, output_dir="outputs/generated", prefix="ai_generated"):
        """Guardar imágenes generadas"""
        if not images:
            return []
        
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


def translate_to_english(text):
    """Traducir texto a inglés (opcional, mejora resultados)"""
    # Implementación simple de traducción común español->inglés
    translations = {
        "gato": "cat",
        "perro": "dog",
        "casa": "house",
        "playa": "beach",
        "montaña": "mountain",
        "ciudad": "city",
        "bosque": "forest",
        "cielo": "sky",
        "noche": "night",
        "día": "day",
        "sol": "sun",
        "luna": "moon",
        "estrella": "star",
        "flor": "flower",
        "árbol": "tree",
        "mar": "sea",
        "río": "river",
        "pájaro": "bird",
        "mujer": "woman",
        "hombre": "man",
        "niño": "child",
        "coche": "car",
        "avión": "airplane",
    }
    
    words = text.lower().split()
    translated = [translations.get(word, word) for word in words]
    return " ".join(translated)


def main():
    """Función principal"""
    print("\n" + "🎨 "*20)
    print("   GENERADOR SIMPLE DE IMÁGENES CON IA")
    print("   (Gratis, sin GPU necesaria)")
    print("🎨 "*20 + "\n")
    
    print("="*60)
    print("ℹ️  INFORMACIÓN:")
    print("="*60)
    print("• Usa Pollinations.ai (API gratuita)")
    print("• No requiere GPU ni cuenta")
    print("• Genera imágenes en ~10-30 segundos")
    print("• Escribe en inglés o español")
    print("="*60 + "\n")
    
    # Inicializar generador
    generator = SimpleImageGenerator()
    
    print("📝 EJEMPLOS DE PROMPTS:")
    print("  • 'a cute cat wearing a hat'")
    print("  • 'beautiful landscape with mountains and sunset'")
    print("  • 'futuristic city with neon lights'")
    print("  • 'gato con sombrero' (se traduce automáticamente)")
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
            
            # Mejorar prompt si está en español
            if any(ord(c) > 127 for c in prompt):  # Detectar caracteres no ASCII
                original = prompt
                prompt = translate_to_english(prompt)
                print(f"💡 Traducido a: {prompt}")
            
            # Generar
            print()
            images = generator.generate(prompt)
            
            if images:
                # Guardar
                print()
                saved_paths = generator.save_images(images)
                
                print(f"\n✅ ¡Imagen generada exitosamente!")
                print(f"📁 Guardada en: {saved_paths[0]}\n")
                
                # Preguntar si abrir
                open_choice = input("¿Abrir imagen ahora? (s/n): ").strip().lower()
                if open_choice in ['s', 'si', 'sí', 'y', 'yes']:
                    os.system(f"xdg-open {saved_paths[0]} 2>/dev/null || open {saved_paths[0]} 2>/dev/null")
            
            # Preguntar si continuar
            print()
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
            print("💡 Intenta con otro prompt\n")


if __name__ == "__main__":
    main()
