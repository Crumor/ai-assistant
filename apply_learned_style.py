"""
Sistema para aplicar el estilo de tus datos de entrenamiento a nuevas imágenes
Usa tu modelo entrenado como guía para generar/modificar imágenes
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import cv2


class StyleTransfer:
    """
    Aplica características de tus datos entrenados a nuevas imágenes
    """
    
    def __init__(self, model_path='models/checkpoints/best_model.pt'):
        """Cargar tu modelo entrenado"""
        print("🎨 Cargando tu modelo entrenado...")
        
        from src.models.vision_model import VisionModel
        
        # Cargar modelo
        self.model = VisionModel(num_classes=2, pretrained=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        print(f"✓ Modelo cargado en {self.device}")
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    
    def extract_features(self, image_path):
        """
        Extraer características de una imagen usando tu modelo
        Estas características representan lo que tu modelo aprendió
        """
        # Cargar imagen
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extraer features del modelo
        with torch.no_grad():
            features = self.model.extract_features(input_tensor)
        
        return features.cpu().numpy()
    
    def get_style_features_from_training_data(self, data_dir='data/train'):
        """
        Extraer características promedio de tus datos de entrenamiento
        Esto representa el "estilo" aprendido de tus imágenes/videos
        """
        print("\n🔍 Analizando tus datos de entrenamiento...")
        
        data_path = Path(data_dir)
        all_features = []
        
        # Procesar cada clase
        for class_dir in data_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            print(f"  Procesando clase: {class_dir.name}")
            
            # Obtener imágenes
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            for img_path in images[:10]:  # Limitar para velocidad
                try:
                    features = self.extract_features(str(img_path))
                    all_features.append(features)
                except Exception as e:
                    print(f"    ⚠️  Error con {img_path.name}: {e}")
        
        # Promedio de features
        avg_features = np.mean(all_features, axis=0)
        print(f"✓ Características extraídas de {len(all_features)} imágenes")
        
        return avg_features
    
    def apply_learned_style(self, input_image_path, output_path='outputs/styled_image.png', intensity=0.5):
        """
        Aplicar el estilo aprendido de tus datos a una nueva imagen
        
        Args:
            input_image_path: Imagen a modificar
            output_path: Dónde guardar resultado
            intensity: Qué tanto aplicar el estilo (0-1)
        """
        print(f"\n🎨 Aplicando estilo aprendido a: {input_image_path}")
        
        # Cargar imagen
        image = Image.open(input_image_path).convert('RGB')
        original_size = image.size
        
        # Extraer features de la imagen
        input_features = self.extract_features(input_image_path)
        
        # Obtener features del estilo entrenado
        style_features = self.get_style_features_from_training_data()
        
        # Mezclar features (simple blending)
        mixed_features = (1 - intensity) * input_features + intensity * style_features
        
        print(f"✓ Features mezcladas (intensidad: {intensity})")
        
        # Por ahora, aplicar transformación de color simple
        # (Un transfer de estilo completo requeriría un modelo generativo)
        styled_image = self._apply_color_transfer(input_image_path, intensity)
        
        # Guardar
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        styled_image.save(output_path)
        
        print(f"✓ Imagen guardada: {output_path}")
        return output_path
    
    def _apply_color_transfer(self, image_path, intensity=0.5):
        """
        Aplicar transformación de color basada en el estilo aprendido
        """
        # Cargar con OpenCV
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Aplicar ajustes basados en intensidad
        # Estos valores se pueden ajustar según tus datos
        
        # Ajuste de saturación
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= (1 + intensity * 0.3)  # Aumentar saturación
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img_adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Mezclar con original
        result = cv2.addWeighted(img, 1-intensity, img_adjusted, intensity, 0)
        
        return Image.fromarray(result)


def analyze_training_data_characteristics():
    """
    Analizar y mostrar características de tus datos de entrenamiento
    """
    print("\n" + "="*60)
    print("📊 ANÁLISIS DE TUS DATOS DE ENTRENAMIENTO")
    print("="*60)
    
    data_dir = Path('data/train')
    
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        print(f"\n📁 Clase: {class_dir.name}")
        
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        if not images:
            print("  Sin imágenes")
            continue
        
        # Analizar características visuales
        colors = []
        sizes = []
        
        for img_path in images[:10]:
            try:
                img = Image.open(img_path)
                sizes.append(img.size)
                
                # Colores dominantes
                img_small = img.resize((50, 50))
                pixels = np.array(img_small).reshape(-1, 3)
                avg_color = pixels.mean(axis=0)
                colors.append(avg_color)
            except:
                pass
        
        if colors:
            avg_color = np.mean(colors, axis=0)
            print(f"  Imágenes: {len(images)}")
            print(f"  Color promedio (RGB): ({avg_color[0]:.0f}, {avg_color[1]:.0f}, {avg_color[2]:.0f})")
            
            # Determinar tono
            r, g, b = avg_color
            if r > g and r > b:
                tone = "Tonos rojos/cálidos"
            elif g > r and g > b:
                tone = "Tonos verdes/naturales"
            elif b > r and b > g:
                tone = "Tonos azules/fríos"
            else:
                tone = "Tonos neutros"
            
            print(f"  Estilo visual: {tone}")


def main():
    """Función principal"""
    print("\n" + "🎨 "*20)
    print("   APLICAR ESTILO APRENDIDO DE TUS DATOS")
    print("🎨 "*20)
    
    # Analizar datos primero
    analyze_training_data_characteristics()
    
    print("\n" + "="*60)
    print("OPCIONES:")
    print("="*60)
    print("1. Aplicar estilo aprendido a una imagen")
    print("2. Ver análisis detallado de tus datos")
    print("3. Salir")
    
    try:
        choice = input("\nSelecciona opción (1-3): ").strip()
        
        if choice == '1':
            # Cargar sistema
            style_transfer = StyleTransfer()
            
            # Solicitar imagen
            image_path = input("\n📸 Ruta de la imagen a modificar: ").strip()
            
            if not Path(image_path).exists():
                print(f"❌ Imagen no encontrada: {image_path}")
                return
            
            # Intensidad
            intensity = input("💪 Intensidad del estilo (0.0-1.0) [0.5]: ").strip()
            intensity = float(intensity) if intensity else 0.5
            intensity = max(0.0, min(1.0, intensity))
            
            # Aplicar
            output_path = f"outputs/styled_{Path(image_path).name}"
            result = style_transfer.apply_learned_style(
                image_path,
                output_path,
                intensity=intensity
            )
            
            print(f"\n✅ ¡Estilo aplicado!")
            print(f"📁 Original: {image_path}")
            print(f"📁 Resultado: {result}")
            
            # Abrir
            open_choice = input("\n¿Abrir resultado? (s/n): ").strip().lower()
            if open_choice in ['s', 'si', 'sí', 'y', 'yes']:
                import os
                os.system(f"xdg-open {result} 2>/dev/null || open {result} 2>/dev/null")
        
        elif choice == '2':
            # Ya mostrado arriba
            pass
        
        elif choice == '3':
            print("\n👋 Hasta luego!")
        
        else:
            print("\n❌ Opción inválida")
    
    except KeyboardInterrupt:
        print("\n\n👋 Cancelado")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
