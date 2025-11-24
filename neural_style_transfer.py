#!/usr/bin/env python3
"""
Neural Style Transfer usando tu modelo entrenado
Sistema que REALMENTE aprende características visuales y las aplica
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
from src.models.vision_model import VisionModel


class NeuralStyleTransfer:
    """
    Style Transfer usando features de tu modelo entrenado
    Aprende características REALES (no solo colores)
    """
    
    def __init__(self, model_path='models/checkpoints/best_model.pt'):
        """Cargar modelo entrenado y preparar para extraer features"""
        print("\n" + "🎨 "*20)
        print("   NEURAL STYLE TRANSFER")
        print("   Usando tu modelo entrenado")
        print("🎨 "*20 + "\n")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}\n")
        
        # Cargar tu modelo entrenado
        print("📦 Cargando tu modelo entrenado...")
        self.model = VisionModel(num_classes=2, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✓ Modelo cargado\n")
        
        # Extraer capas intermedias para style transfer
        self.feature_layers = self._get_feature_extractors()
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
        ])
        
        self.denorm = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    
    def _get_feature_extractors(self):
        """
        Extraer capas intermedias del modelo para style features
        Estas capas capturan diferentes niveles de abstracción
        """
        # Obtener el backbone (ResNet50)
        backbone = self.model.backbone
        
        # Definir las capas que queremos usar
        # Usamos las capas residuales de ResNet
        layers = {
            'conv1': nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool
            ),
            'layer1': backbone.layer1,  # Residual block 1
            'layer2': backbone.layer2,  # Residual block 2
            'layer3': backbone.layer3,  # Residual block 3
            'layer4': backbone.layer4   # Residual block 4
        }
        
        return layers
    
    def extract_features(self, image_tensor, target_layers=None):
        """
        Extraer features de múltiples capas
        Estas representan el "estilo" en diferentes niveles
        """
        if target_layers is None:
            target_layers = ['conv1', 'layer1', 'layer2', 'layer3']
        
        features = {}
        
        # Normalizar la imagen
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        x = normalize(image_tensor)
        
        # Siempre procesar secuencialmente desde el inicio
        # conv1
        x = self.feature_layers['conv1'](x)
        if 'conv1' in target_layers:
            features['conv1'] = x
        
        # layer1
        x = self.feature_layers['layer1'](x)
        if 'layer1' in target_layers:
            features['layer1'] = x
        
        # layer2
        x = self.feature_layers['layer2'](x)
        if 'layer2' in target_layers:
            features['layer2'] = x
        
        # layer3
        x = self.feature_layers['layer3'](x)
        if 'layer3' in target_layers:
            features['layer3'] = x
        
        # layer4 (solo si se pide)
        if 'layer4' in target_layers:
            x = self.feature_layers['layer4'](x)
            features['layer4'] = x
        
        return features
    
    def gram_matrix(self, feature_map):
        """
        Calcular Gram Matrix para capturar correlaciones de features
        Esto representa el "estilo" (texturas, patrones)
        """
        batch, channels, height, width = feature_map.size()
        features = feature_map.view(batch * channels, height * width)
        gram = torch.mm(features, features.t())
        return gram / (batch * channels * height * width)
    
    def learn_style_from_dataset(self, data_dir='data/train'):
        """
        Aprender el estilo de TODAS tus imágenes de entrenamiento
        Extrae y promedia características de todo tu dataset
        """
        print("🔍 Analizando tus imágenes de entrenamiento...")
        
        data_path = Path(data_dir)
        style_features = {
            'conv1': [],
            'layer1': [],
            'layer2': [],
            'layer3': []
        }
        
        image_count = 0
        
        # Procesar cada clase
        for class_dir in data_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            print(f"  Procesando: {class_dir.name}")
            
            # Obtener imágenes
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            for img_path in images:
                try:
                    # Cargar imagen
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # Extraer features
                    with torch.no_grad():
                        features = self.extract_features(img_tensor)
                    
                    # Calcular gram matrices (representación del estilo)
                    for layer in features:
                        gram = self.gram_matrix(features[layer])
                        style_features[layer].append(gram)
                    
                    image_count += 1
                    
                except Exception as e:
                    print(f"    ⚠️ Error con {img_path.name}: {e}")
        
        # Promediar características de todas las imágenes
        print(f"\n✓ Analizadas {image_count} imágenes")
        print("📊 Calculando características promedio...\n")
        
        avg_style = {}
        for layer in style_features:
            if style_features[layer]:
                avg_style[layer] = torch.mean(torch.stack(style_features[layer]), dim=0)
        
        return avg_style
    
    def apply_style(
        self,
        content_image_path,
        output_path='outputs/styled_output.jpg',
        style_weight=1e6,
        content_weight=1,
        num_steps=300,
        learning_rate=0.01
    ):
        """
        Aplicar el estilo aprendido a una imagen nueva
        
        Args:
            content_image_path: Imagen a la que aplicar el estilo
            output_path: Dónde guardar resultado
            style_weight: Peso del estilo (más alto = más efecto)
            content_weight: Peso del contenido (preservar estructura)
            num_steps: Iteraciones de optimización
            learning_rate: Velocidad de aprendizaje
        """
        print(f"🎨 Aplicando estilo aprendido a: {content_image_path}\n")
        
        # 1. Aprender estilo de tus datos
        print("PASO 1: Aprendiendo estilo de tus datos de entrenamiento")
        style_targets = self.learn_style_from_dataset()
        
        # 2. Cargar imagen de contenido
        print("PASO 2: Cargando imagen de contenido")
        content_img = Image.open(content_image_path).convert('RGB')
        content_tensor = self.transform(content_img).unsqueeze(0).to(self.device)
        
        # Extraer features de contenido
        with torch.no_grad():
            content_features = self.extract_features(content_tensor, ['layer3'])
        
        print(f"✓ Imagen cargada: {content_tensor.shape}\n")
        
        # 3. Inicializar imagen de salida (copia del contenido)
        print("PASO 3: Inicializando optimización")
        output_img = content_tensor.clone().requires_grad_(True)
        
        # Optimizador
        optimizer = optim.LBFGS([output_img], lr=learning_rate)
        
        print(f"⚙️  Configuración:")
        print(f"  - Style weight: {style_weight}")
        print(f"  - Content weight: {content_weight}")
        print(f"  - Steps: {num_steps}")
        print(f"  - Learning rate: {learning_rate}\n")
        
        # 4. Optimización iterativa
        print("PASO 4: Aplicando estilo (esto puede tardar 1-2 minutos)...\n")
        
        steps = [0]
        
        def closure():
            """Función de pérdida para optimización"""
            optimizer.zero_grad()
            
            # Extraer features de la imagen actual
            output_features = self.extract_features(output_img)
            
            # Content loss (preservar estructura)
            content_loss = torch.mean(
                (output_features['layer3'] - content_features['layer3']) ** 2
            )
            
            # Style loss (aplicar estilo aprendido)
            style_loss = 0
            for layer in style_targets:
                output_gram = self.gram_matrix(output_features[layer])
                target_gram = style_targets[layer]
                style_loss += torch.mean((output_gram - target_gram) ** 2)
            
            # Loss total
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            # Backprop
            total_loss.backward()
            
            steps[0] += 1
            if steps[0] % 50 == 0:
                print(f"  Step {steps[0]}/{num_steps}")
                print(f"    Content Loss: {content_loss.item():.4f}")
                print(f"    Style Loss: {style_loss.item():.4f}")
                print(f"    Total Loss: {total_loss.item():.4f}\n")
            
            return total_loss
        
        # Ejecutar optimización
        for step in range(num_steps // 10):  # LBFGS hace ~10 pasos internos
            optimizer.step(closure)
            
            if steps[0] >= num_steps:
                break
        
        print("✓ Optimización completada\n")
        
        # 5. Guardar resultado
        print("PASO 5: Guardando resultado...")
        
        # Convertir tensor a imagen
        output_img_np = output_img.squeeze(0).detach().cpu()
        output_img_np = torch.clamp(output_img_np, 0, 1)
        output_img_np = output_img_np.permute(1, 2, 0).numpy()
        output_img_np = (output_img_np * 255).astype(np.uint8)
        
        # Guardar
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        output_pil = Image.fromarray(output_img_np)
        output_pil.save(output_path)
        
        print(f"✅ Resultado guardado: {output_path}\n")
        
        return output_path
    
    def batch_apply(self, input_dir, output_dir='outputs/batch_styled'):
        """Aplicar estilo a múltiples imágenes"""
        print(f"\n📦 Procesamiento por lotes")
        print(f"Entrada: {input_dir}")
        print(f"Salida: {output_dir}\n")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Buscar imágenes
        images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        
        if not images:
            print(f"❌ No se encontraron imágenes en {input_dir}")
            return
        
        print(f"✓ Encontradas {len(images)} imágenes\n")
        
        # Aprender estilo una sola vez
        style_targets = self.learn_style_from_dataset()
        
        # Procesar cada imagen
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] Procesando {img_path.name}...")
            
            output_file = output_path / f"styled_{img_path.name}"
            
            try:
                self.apply_style(img_path, output_file)
            except Exception as e:
                print(f"⚠️ Error: {e}\n")
        
        print(f"✅ Procesamiento completado!")
        print(f"📁 Resultados en: {output_dir}")


def main():
    """Función principal"""
    print("\n" + "🎨 "*20)
    print("   NEURAL STYLE TRANSFER")
    print("   Aprende de tus datos y aplica características reales")
    print("🎨 "*20 + "\n")
    
    print("="*60)
    print("OPCIONES:")
    print("="*60)
    print("1. Aplicar estilo a una imagen")
    print("2. Aplicar estilo a múltiples imágenes")
    print("3. Configuración avanzada")
    print("4. Ver información del modelo")
    print("5. Salir")
    
    try:
        choice = input("\nSelecciona opción (1-5): ").strip()
        
        if choice == '1':
            # Single image
            image_path = input("\n📸 Ruta de la imagen: ").strip()
            
            if not Path(image_path).exists():
                print(f"❌ Imagen no encontrada: {image_path}")
                return
            
            # Crear sistema
            nst = NeuralStyleTransfer()
            
            # Preguntar configuración
            print("\n⚙️  Configuración (presiona Enter para valores por defecto):")
            
            try:
                style_w = input("  Style weight [1000000]: ").strip()
                style_weight = float(style_w) if style_w else 1e6
                
                steps = input("  Pasos de optimización [300]: ").strip()
                num_steps = int(steps) if steps else 300
            except ValueError:
                print("⚠️ Usando valores por defecto")
                style_weight = 1e6
                num_steps = 300
            
            # Aplicar
            output = f"outputs/styled_{Path(image_path).name}"
            nst.apply_style(
                image_path,
                output_path=output,
                style_weight=style_weight,
                num_steps=num_steps
            )
            
            # Abrir
            import os
            os.system(f"xdg-open {output} 2>/dev/null || open {output} 2>/dev/null")
        
        elif choice == '2':
            # Batch
            input_dir = input("\n📁 Directorio con imágenes: ").strip()
            
            if not Path(input_dir).exists():
                print(f"❌ Directorio no encontrado: {input_dir}")
                return
            
            nst = NeuralStyleTransfer()
            nst.batch_apply(input_dir)
        
        elif choice == '3':
            # Advanced
            print("\n⚙️  CONFIGURACIÓN AVANZADA\n")
            print("Parámetros clave:")
            print("\n1. style_weight (peso del estilo)")
            print("   - Bajo (100,000): Efecto sutil")
            print("   - Medio (1,000,000): Balanceado")
            print("   - Alto (10,000,000): Efecto fuerte")
            
            print("\n2. num_steps (pasos de optimización)")
            print("   - 100: Rápido, menos calidad")
            print("   - 300: Balanceado (recomendado)")
            print("   - 500+: Lento, mejor calidad")
            
            print("\n3. learning_rate")
            print("   - 0.01: Estándar")
            print("   - 0.001: Más preciso, más lento")
        
        elif choice == '4':
            # Info
            print("\n📊 INFORMACIÓN DEL MODELO\n")
            
            try:
                checkpoint = torch.load('models/checkpoints/best_model.pt')
                print(f"✓ Modelo encontrado")
                print(f"  Época: {checkpoint.get('epoch', 'N/A')}")
                print(f"  Accuracy: {checkpoint.get('val_acc', 0):.2f}%")
                
                history = checkpoint.get('history', {})
                if history:
                    train_count = len(history.get('train_loss', []))
                    print(f"  Épocas entrenadas: {train_count}")
            except:
                print("❌ No se pudo cargar información del modelo")
        
        elif choice == '5':
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
