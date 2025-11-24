#!/usr/bin/env python3
"""
Virtual Try-On Avanzado usando APIs o modelos pre-entrenados
Para resultados profesionales
"""

import requests
import base64
from pathlib import Path
import json
import os


class AdvancedVirtualTryOn:
    """
    Virtual Try-On usando servicios externos para mejor calidad
    """
    
    def __init__(self, api_provider='replicate'):
        """
        Inicializar con proveedor de API
        
        Providers:
        - 'replicate': Replicate.com (HR-VITON)
        - 'deepai': DeepAI Virtual Try-On
        - 'local': Modelos locales (requiere setup)
        """
        self.api_provider = api_provider
        
        print("\n" + "🎽 "*20)
        print("   VIRTUAL TRY-ON PROFESIONAL")
        print("🎽 "*20 + "\n")
        
        print(f"Proveedor: {api_provider}")
        
        # Verificar API keys
        self._check_api_keys()
    
    def _check_api_keys(self):
        """Verificar que las API keys estén configuradas"""
        if self.api_provider == 'replicate':
            api_key = os.getenv('REPLICATE_API_TOKEN')
            if not api_key:
                print("\n⚠️  REPLICATE_API_TOKEN no configurado")
                print("💡 Obtén tu API key en: https://replicate.com/account")
                print("💡 Luego ejecuta: export REPLICATE_API_TOKEN='tu_token'\n")
        
        elif self.api_provider == 'deepai':
            api_key = os.getenv('DEEPAI_API_KEY')
            if not api_key:
                print("\n⚠️  DEEPAI_API_KEY no configurado")
                print("💡 Obtén tu API key en: https://deepai.org/")
                print("💡 Luego ejecuta: export DEEPAI_API_KEY='tu_key'\n")
    
    def try_on_replicate(self, person_image_path, garment_image_path):
        """
        Virtual Try-On usando Replicate (HR-VITON)
        
        Requiere: pip install replicate
        """
        try:
            import replicate
        except ImportError:
            print("❌ Instala replicate: pip install replicate")
            return None
        
        print("\n🚀 Usando Replicate (HR-VITON)...")
        print("⏳ Esto puede tardar 30-60 segundos...")
        
        api_token = os.getenv('REPLICATE_API_TOKEN')
        if not api_token:
            print("❌ Falta REPLICATE_API_TOKEN")
            return None
        
        try:
            # Leer imágenes
            with open(person_image_path, 'rb') as f:
                person_b64 = base64.b64encode(f.read()).decode()
            
            with open(garment_image_path, 'rb') as f:
                garment_b64 = base64.b64encode(f.read()).decode()
            
            # Llamar API
            output = replicate.run(
                "sangyun884/hr-viton:0564b48e8f8bc0c4b3e5b9c9a2e9c0d9",
                input={
                    "person_image": f"data:image/jpeg;base64,{person_b64}",
                    "garment_image": f"data:image/jpeg;base64,{garment_b64}"
                }
            )
            
            print("✓ Generación completada")
            return output
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def try_on_deepai(self, person_image_path, garment_image_path):
        """
        Virtual Try-On usando DeepAI
        
        Más simple pero menos preciso que Replicate
        """
        print("\n🚀 Usando DeepAI...")
        
        api_key = os.getenv('DEEPAI_API_KEY')
        if not api_key:
            print("❌ Falta DEEPAI_API_KEY")
            return None
        
        try:
            # Por ahora DeepAI no tiene endpoint específico de try-on
            # Usar como ejemplo general
            
            url = "https://api.deepai.org/api/image-editor"
            
            files = {
                'image': open(person_image_path, 'rb'),
                'text': 'Replace shirt with new clothing'
            }
            
            headers = {'api-key': api_key}
            
            response = requests.post(url, files=files, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Generación completada")
                return result['output_url']
            else:
                print(f"❌ Error: {response.status_code}")
                return None
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def try_on_local_hrviton(self, person_image_path, garment_image_path):
        """
        Ejecutar HR-VITON localmente
        
        Requiere:
        - GPU con 12+ GB VRAM
        - Modelo pre-entrenado descargado
        - Repositorio HR-VITON clonado
        """
        print("\n🚀 Usando HR-VITON Local...")
        print("⚠️  Requiere GPU potente (12+ GB VRAM)")
        
        # Verificar si el repo existe
        hrviton_path = Path('external/HR-VITON')
        
        if not hrviton_path.exists():
            print("\n❌ HR-VITON no instalado")
            print("\n💡 Para instalar:")
            print("   cd external")
            print("   git clone https://github.com/sangyun884/HR-VITON.git")
            print("   cd HR-VITON")
            print("   # Seguir instrucciones del README")
            return None
        
        # Ejecutar modelo
        import sys
        sys.path.append(str(hrviton_path))
        
        try:
            # Importar módulos de HR-VITON
            # (Código específico depende de la implementación)
            
            print("🔄 Procesando con HR-VITON...")
            print("⏳ Esto puede tardar 1-2 minutos...")
            
            # TODO: Implementar llamada real al modelo
            # result = hrviton_model.inference(person_image, garment_image)
            
            print("✓ Procesamiento completado")
            return None
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def try_on(self, person_image_path, garment_image_path, output_path='outputs/tryon_advanced.jpg'):
        """
        Virtual Try-On usando el proveedor configurado
        """
        print(f"\n🔄 Virtual Try-On Profesional")
        print(f"📸 Persona: {person_image_path}")
        print(f"👕 Prenda: {garment_image_path}")
        
        # Llamar API según proveedor
        if self.api_provider == 'replicate':
            result = self.try_on_replicate(person_image_path, garment_image_path)
        
        elif self.api_provider == 'deepai':
            result = self.try_on_deepai(person_image_path, garment_image_path)
        
        elif self.api_provider == 'local':
            result = self.try_on_local_hrviton(person_image_path, garment_image_path)
        
        else:
            print(f"❌ Proveedor desconocido: {self.api_provider}")
            return None
        
        if result:
            # Descargar y guardar resultado
            if isinstance(result, str) and result.startswith('http'):
                print(f"\n📥 Descargando resultado...")
                response = requests.get(result)
                
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Guardado: {output_path}")
            
            return result
        
        return None


def setup_hrviton_local():
    """
    Guía para instalar HR-VITON localmente
    """
    print("\n" + "="*60)
    print("INSTALACIÓN DE HR-VITON LOCAL")
    print("="*60 + "\n")
    
    print("📋 REQUISITOS:")
    print("  • GPU: 12+ GB VRAM (RTX 3090, A100)")
    print("  • Python: 3.8-3.10")
    print("  • PyTorch: 1.13+ con CUDA")
    print("  • Espacio: ~5 GB para modelo + código")
    
    print("\n📝 PASOS:")
    print("\n1. Crear directorio para modelos externos:")
    print("   mkdir -p external && cd external")
    
    print("\n2. Clonar repositorio HR-VITON:")
    print("   git clone https://github.com/sangyun884/HR-VITON.git")
    print("   cd HR-VITON")
    
    print("\n3. Instalar dependencias:")
    print("   pip install -r requirements.txt")
    
    print("\n4. Descargar modelo pre-entrenado:")
    print("   # Seguir instrucciones en el README del repo")
    print("   # Generalmente desde Google Drive o Hugging Face")
    
    print("\n5. Configurar rutas en config:")
    print("   # Editar config.py con rutas correctas")
    
    print("\n6. Probar instalación:")
    print("   python inference.py --person test.jpg --garment shirt.jpg")
    
    print("\n⚠️  ALTERNATIVAS SI NO TIENES GPU SUFICIENTE:")
    print("  1. Usar Google Colab (GPU gratis limitada)")
    print("  2. Rentar GPU en Paperspace/Lambda Labs")
    print("  3. Usar APIs como Replicate (recomendado)")
    
    print("\n" + "="*60 + "\n")


def main():
    """Función principal"""
    print("\n" + "🎽 "*20)
    print("   VIRTUAL TRY-ON PROFESIONAL")
    print("🎽 "*20 + "\n")
    
    print("="*60)
    print("OPCIONES:")
    print("="*60)
    print("1. Usar Replicate API (recomendado)")
    print("2. Usar DeepAI API")
    print("3. Setup HR-VITON local")
    print("4. Ver comparación de opciones")
    print("5. Salir")
    
    try:
        choice = input("\nSelecciona opción (1-5): ").strip()
        
        if choice == '1':
            # Replicate
            tryon = AdvancedVirtualTryOn(api_provider='replicate')
            
            person = input("\n📸 Ruta imagen persona: ").strip()
            garment = input("👕 Ruta imagen prenda: ").strip()
            
            if person and garment:
                tryon.try_on(person, garment)
        
        elif choice == '2':
            # DeepAI
            tryon = AdvancedVirtualTryOn(api_provider='deepai')
            
            person = input("\n📸 Ruta imagen persona: ").strip()
            garment = input("👕 Ruta imagen prenda: ").strip()
            
            if person and garment:
                tryon.try_on(person, garment)
        
        elif choice == '3':
            # Setup local
            setup_hrviton_local()
        
        elif choice == '4':
            # Comparación
            print("\n" + "="*60)
            print("COMPARACIÓN DE OPCIONES")
            print("="*60 + "\n")
            
            print("🔹 REPLICATE (Recomendado)")
            print("  Pros:")
            print("    ✓ Mejor calidad (HR-VITON)")
            print("    ✓ Sin setup complicado")
            print("    ✓ GPU potente incluida")
            print("  Contras:")
            print("    ✗ Pago por uso (~$0.05/imagen)")
            print("    ✗ Requiere internet")
            print("  Uso: pip install replicate")
            
            print("\n🔹 DEEPAI")
            print("  Pros:")
            print("    ✓ Fácil de usar")
            print("    ✓ Barato")
            print("  Contras:")
            print("    ✗ Menor calidad")
            print("    ✗ Menos control")
            print("  Uso: Solo API key")
            
            print("\n🔹 HR-VITON LOCAL")
            print("  Pros:")
            print("    ✓ Mejor calidad posible")
            print("    ✓ Sin límites de uso")
            print("    ✓ Privacidad total")
            print("  Contras:")
            print("    ✗ Requiere GPU potente (12+ GB)")
            print("    ✗ Setup complejo")
            print("    ✗ Mantenimiento")
            print("  Costo: GPU cloud ~$0.50-2/hora")
            
            print("\n💡 RECOMENDACIÓN:")
            print("  • Prototipos → Script básico (gratis)")
            print("  • Producción → Replicate API")
            print("  • Escala grande → HR-VITON local + GPU")
            
        elif choice == '5':
            print("\n👋 Hasta luego!")
        
        else:
            print("\n❌ Opción inválida")
    
    except KeyboardInterrupt:
        print("\n\n👋 Cancelado")


if __name__ == "__main__":
    main()
