#!/usr/bin/env python3
"""
Script para usar el sistema de Virtual Try-On
Aprende de un catálogo de ropa y aplica los estilos a fotos de modelos
"""

import os
import sys
import argparse
from pathlib import Path


def print_banner():
    """Imprimir banner"""
    print("\n" + "👔 " * 20)
    print("   VIRTUAL TRY-ON - PROBADOR VIRTUAL DE ROPA")
    print("👔 " * 20 + "\n")


def check_model_exists():
    """Verificar si existe un modelo entrenado"""
    model_paths = [
        'models/checkpoints/best_model.pt',
        'models/virtual_tryon.pt'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            return path
    
    return None


def setup_virtual_tryon():
    """Configurar el sistema de virtual try-on"""
    print("🔧 Configurando Virtual Try-On...")
    
    # Verificar modelo
    model_path = check_model_exists()
    
    if model_path is None:
        print("\n❌ No se encontró un modelo entrenado")
        print("\n💡 Opciones:")
        print("1. Entrenar un modelo primero:")
        print("   python train.py --epochs 20")
        print("\n2. O usar un modelo pre-entrenado (si tienes uno)")
        return None
    
    print(f"✓ Modelo encontrado: {model_path}")
    
    # Importar después de verificar que hay modelo
    try:
        from src.inference.virtual_tryon import VirtualTryOn, create_virtual_tryon_model
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        return None
    
    # Crear modelo de virtual try-on si es necesario
    if not model_path.endswith('virtual_tryon.pt'):
        print("🔄 Convirtiendo modelo a Virtual Try-On...")
        model_path = create_virtual_tryon_model(
            model_path,
            'models/virtual_tryon.pt'
        )
    
    # Inicializar sistema
    tryon = VirtualTryOn(model_path=model_path)
    
    print("✓ Sistema listo!\n")
    return tryon


def learn_from_catalog_interactive(tryon):
    """Modo interactivo para aprender de catálogo"""
    print("="*60)
    print("📚 APRENDER DE CATÁLOGO DE ROPA")
    print("="*60 + "\n")
    
    print("Organiza tu catálogo así:")
    print("  catalog/")
    print("    camisas/")
    print("      camisa1.jpg")
    print("      camisa2.jpg")
    print("    pantalones/")
    print("      pantalon1.jpg")
    print("      pantalon2.jpg\n")
    
    catalog_dir = input("📁 Ruta del directorio del catálogo: ").strip()
    
    if not os.path.exists(catalog_dir):
        print(f"❌ El directorio {catalog_dir} no existe")
        return False
    
    # Verificar si tiene subdirectorios o es un directorio plano
    subdirs = [d for d in os.listdir(catalog_dir) 
               if os.path.isdir(os.path.join(catalog_dir, d))]
    
    if len(subdirs) > 0:
        # Tiene categorías
        print(f"\n✓ Encontradas {len(subdirs)} categorías:")
        for i, subdir in enumerate(subdirs, 1):
            print(f"  {i}. {subdir}")
        
        print("\n¿Aprender de todas las categorías? (s/n): ", end='')
        choice = input().strip().lower()
        
        if choice in ['s', 'si', 'sí', 'y', 'yes']:
            for subdir in subdirs:
                subdir_path = os.path.join(catalog_dir, subdir)
                try:
                    tryon.learn_from_catalog(subdir_path, category_name=subdir)
                except Exception as e:
                    print(f"⚠️  Error aprendiendo de {subdir}: {e}")
        else:
            print("Selecciona categoría (1-{}): ".format(len(subdirs)), end='')
            try:
                idx = int(input().strip()) - 1
                if 0 <= idx < len(subdirs):
                    subdir = subdirs[idx]
                    subdir_path = os.path.join(catalog_dir, subdir)
                    tryon.learn_from_catalog(subdir_path, category_name=subdir)
                else:
                    print("❌ Índice inválido")
                    return False
            except ValueError:
                print("❌ Entrada inválida")
                return False
    else:
        # Directorio plano
        category_name = input("📝 Nombre de la categoría (ej: 'camisas'): ").strip()
        if not category_name:
            category_name = 'default'
        
        try:
            tryon.learn_from_catalog(catalog_dir, category_name=category_name)
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    # Guardar estilos aprendidos
    print("\n💾 Guardando estilos aprendidos...")
    tryon.save_styles()
    
    return True


def apply_style_interactive(tryon):
    """Modo interactivo para aplicar estilo"""
    print("="*60)
    print("🎨 APLICAR ESTILO A IMAGEN")
    print("="*60 + "\n")
    
    # Verificar estilos disponibles
    if len(tryon.catalog_styles) == 0:
        print("❌ No hay estilos aprendidos")
        print("💡 Primero usa la opción 1 para aprender de un catálogo")
        return False
    
    print("Estilos disponibles:")
    styles = list(tryon.catalog_styles.keys())
    for i, style in enumerate(styles, 1):
        print(f"  {i}. {style}")
    
    # Seleccionar estilo
    print(f"\nSelecciona estilo (1-{len(styles)}): ", end='')
    try:
        idx = int(input().strip()) - 1
        if 0 <= idx < len(styles):
            category_name = styles[idx]
        else:
            print("❌ Índice inválido")
            return False
    except ValueError:
        print("❌ Entrada inválida")
        return False
    
    # Imagen objetivo
    image_path = input("\n📷 Ruta de la imagen (modelo): ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ La imagen {image_path} no existe")
        return False
    
    # Output
    output_dir = 'outputs/virtual_tryon'
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"styled_{Path(image_path).stem}_{category_name}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    
    # Aplicar estilo
    try:
        styled_image = tryon.apply_to_image(
            image_path,
            category_name=category_name,
            output_path=output_path
        )
        
        print(f"\n✅ ¡Éxito!")
        print(f"📁 Resultado guardado en: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error aplicando estilo: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Virtual Try-On: Aprende de catálogos y aplica estilos a imágenes'
    )
    parser.add_argument(
        '--learn',
        type=str,
        metavar='CATALOG_DIR',
        help='Aprender de un directorio de catálogo'
    )
    parser.add_argument(
        '--apply',
        type=str,
        metavar='IMAGE_PATH',
        help='Aplicar estilo a una imagen'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='default',
        help='Categoría de estilo a usar (default: default)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Ruta de salida para la imagen estilizada'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Setup
    tryon = setup_virtual_tryon()
    if tryon is None:
        return 1
    
    # Intentar cargar estilos previos
    if os.path.exists('models/learned_styles.pt'):
        tryon.load_styles()
    
    # Modo línea de comandos
    if args.learn:
        print(f"📚 Aprendiendo de catálogo: {args.learn}")
        try:
            tryon.learn_from_catalog(args.learn, args.category)
            tryon.save_styles()
            print("✅ Catálogo aprendido exitosamente")
            return 0
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    if args.apply:
        if args.category not in tryon.catalog_styles:
            print(f"❌ Estilo '{args.category}' no disponible")
            print(f"Estilos disponibles: {list(tryon.catalog_styles.keys())}")
            return 1
        
        print(f"🎨 Aplicando estilo '{args.category}' a {args.apply}")
        try:
            output = args.output or f'outputs/virtual_tryon/styled_{Path(args.apply).name}'
            tryon.apply_to_image(args.apply, args.category, output)
            print(f"✅ Resultado guardado en: {output}")
            return 0
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    # Modo interactivo
    print("="*60)
    print("MENÚ PRINCIPAL")
    print("="*60)
    
    while True:
        print("\nOpciones:")
        print("  1. 📚 Aprender de catálogo de ropa")
        print("  2. 🎨 Aplicar estilo a imagen")
        print("  3. 💾 Ver estilos aprendidos")
        print("  4. 🚪 Salir")
        
        try:
            choice = input("\nSelecciona opción (1-4): ").strip()
            
            if choice == '1':
                learn_from_catalog_interactive(tryon)
            
            elif choice == '2':
                apply_style_interactive(tryon)
            
            elif choice == '3':
                print("\n📊 Estilos aprendidos:")
                if len(tryon.catalog_styles) == 0:
                    print("  (ninguno)")
                else:
                    for i, style in enumerate(tryon.catalog_styles.keys(), 1):
                        print(f"  {i}. {style}")
            
            elif choice == '4':
                print("\n👋 ¡Hasta luego!")
                break
            
            else:
                print("❌ Opción inválida")
        
        except KeyboardInterrupt:
            print("\n\n👋 Cancelado por usuario")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
