#!/usr/bin/env python3
"""
Ejemplo simple de uso del sistema Virtual Try-On
"""



def print_header():
    print("\n" + "="*60)
    print("EJEMPLO: Virtual Try-On Sistema de Aprendizaje")
    print("="*60 + "\n")


def example_basic_flow():
    """Ejemplo básico del flujo de trabajo"""
    print("📚 FLUJO DE TRABAJO BÁSICO\n")
    
    print("Paso 1: Organizar el catálogo")
    print("-" * 40)
    print("""
    catalog/
      camisas/
        camisa_roja.jpg
        camisa_azul.jpg
        camisa_rayas.jpg
    """)
    
    print("\nPaso 2: Entrenar modelo base (si no existe)")
    print("-" * 40)
    print("$ python quick_start.py")
    
    print("\nPaso 3: Usar Virtual Try-On")
    print("-" * 40)
    print("$ python virtual_tryon.py")
    
    print("\nPaso 4: Aprender del catálogo")
    print("-" * 40)
    print("Opción 1 en el menú interactivo")
    print("O: python virtual_tryon.py --learn catalog/camisas --category camisas")
    
    print("\nPaso 5: Aplicar a foto de modelo")
    print("-" * 40)
    print("Opción 2 en el menú interactivo")
    print("O: python virtual_tryon.py --apply modelo.jpg --category camisas --output resultado.jpg")
    
    print("\n✅ Resultado: modelo.jpg con estilo de las camisas del catálogo\n")


def example_code_usage():
    """Ejemplo de uso programático"""
    print("\n💻 USO PROGRAMÁTICO (PYTHON)\n")
    
    code = """
# Importar módulo
from src.inference.virtual_tryon import VirtualTryOn

# Inicializar sistema
tryon = VirtualTryOn(model_path='models/virtual_tryon.pt')

# Aprender de catálogo de camisas
tryon.learn_from_catalog(
    catalog_dir='catalog/camisas',
    category_name='camisas'
)

# Guardar estilos aprendidos
tryon.save_styles('models/my_styles.pt')

# Aplicar estilo a imagen
styled_image = tryon.apply_to_image(
    image_path='modelo.jpg',
    category_name='camisas',
    output_path='outputs/modelo_estilizado.jpg'
)

print("✅ Imagen procesada!")
    """
    
    print(code)


def example_multiple_categories():
    """Ejemplo con múltiples categorías"""
    print("\n👔 MÚLTIPLES CATEGORÍAS\n")
    
    print("Estructura del catálogo:")
    print("-" * 40)
    print("""
    catalog/
      camisas/
        camisa1.jpg, camisa2.jpg, ...
      pantalones/
        pantalon1.jpg, pantalon2.jpg, ...
      vestidos/
        vestido1.jpg, vestido2.jpg, ...
    """)
    
    print("\nAprender todas las categorías:")
    print("-" * 40)
    print("""
python virtual_tryon.py --learn catalog/camisas --category camisas
python virtual_tryon.py --learn catalog/pantalones --category pantalones
python virtual_tryon.py --learn catalog/vestidos --category vestidos
    """)
    
    print("\nAplicar diferentes estilos:")
    print("-" * 40)
    print("""
python virtual_tryon.py --apply modelo.jpg --category camisas --output modelo_camisa.jpg
python virtual_tryon.py --apply modelo.jpg --category pantalones --output modelo_pantalon.jpg
python virtual_tryon.py --apply modelo.jpg --category vestidos --output modelo_vestido.jpg
    """)


def example_batch_processing():
    """Ejemplo de procesamiento por lotes"""
    print("\n⚡ PROCESAMIENTO POR LOTES\n")
    
    code = """
from pathlib import Path
from src.inference.virtual_tryon import VirtualTryOn

# Inicializar
tryon = VirtualTryOn(model_path='models/virtual_tryon.pt')
tryon.load_styles('models/learned_styles.pt')

# Procesar todas las imágenes en un directorio
input_dir = Path('modelos_input/')
output_dir = Path('modelos_output/')
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob('*.jpg'):
    print(f"Procesando {img_path.name}...")
    
    output_path = output_dir / f"styled_{img_path.name}"
    tryon.apply_to_image(
        str(img_path),
        category_name='camisas',
        output_path=str(output_path)
    )

print("✅ Todas las imágenes procesadas!")
    """
    
    print(code)


def show_architecture():
    """Mostrar arquitectura del sistema"""
    print("\n🏗️ ARQUITECTURA DEL SISTEMA\n")
    
    print("""
    ┌─────────────────────────────────────────┐
    │     Modelo Base (ResNet50)              │
    │     Pre-entrenado en ImageNet           │
    └─────────────────┬───────────────────────┘
                      │
    ┌─────────────────▼───────────────────────┐
    │     Style Encoder                       │
    │     Extrae características del catálogo │
    └─────────────────┬───────────────────────┘
                      │
    ┌─────────────────▼───────────────────────┐
    │     Style Vector (512-dim)              │
    │     Representación del estilo           │
    └─────────────────┬───────────────────────┘
                      │
    ┌─────────────────▼───────────────────────┐
    │     Style Decoder                       │
    │     Aplica estilo a imagen objetivo     │
    └─────────────────┬───────────────────────┘
                      │
    ┌─────────────────▼───────────────────────┐
    │     Imagen Estilizada                   │
    │     Foto de modelo con nuevo estilo     │
    └─────────────────────────────────────────┘
    """)


def main():
    """Función principal"""
    print_header()
    
    print("Este script muestra ejemplos de uso del sistema Virtual Try-On.\n")
    
    while True:
        print("\n" + "="*60)
        print("EJEMPLOS DISPONIBLES:")
        print("="*60)
        print("\n1. 📚 Flujo de trabajo básico")
        print("2. 💻 Uso programático (Python)")
        print("3. 👔 Múltiples categorías")
        print("4. ⚡ Procesamiento por lotes")
        print("5. 🏗️ Arquitectura del sistema")
        print("6. 📖 Ver documentación completa")
        print("7. 🚪 Salir")
        
        try:
            choice = input("\nSelecciona ejemplo (1-7): ").strip()
            
            if choice == '1':
                example_basic_flow()
            elif choice == '2':
                example_code_usage()
            elif choice == '3':
                example_multiple_categories()
            elif choice == '4':
                example_batch_processing()
            elif choice == '5':
                show_architecture()
            elif choice == '6':
                print("\n📖 Abre VIRTUAL_TRYON.md para la documentación completa")
                print("O visita: https://github.com/Crumor/ai-assistant")
            elif choice == '7':
                print("\n👋 ¡Hasta luego!")
                break
            else:
                print("\n❌ Opción inválida")
            
            input("\nPresiona Enter para continuar...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Cancelado")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
