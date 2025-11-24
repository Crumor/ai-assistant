#!/usr/bin/env python3
"""
Aprende características de tus imágenes/videos de entrenamiento y aplícalas a nuevas imágenes
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.vision_model import VisionModel


class StyleLearner:
    def __init__(self, model_path='models/checkpoints/best_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Dispositivo: {self.device}")
        
        # Cargar modelo entrenado
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = VisionModel(num_classes=2)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.to_pil = transforms.ToPILImage()
    
    def extract_features(self, img_tensor):
        """Extrae features de capas intermedias"""
        with torch.no_grad():
            x = self.model.backbone.conv1(img_tensor)
            x = self.model.backbone.bn1(x)
            x = self.model.backbone.relu(x)
            x = self.model.backbone.maxpool(x)
            
            feat1 = self.model.backbone.layer1(x)
            feat2 = self.model.backbone.layer2(feat1)
            feat3 = self.model.backbone.layer3(feat2)
            feat4 = self.model.backbone.layer4(feat3)
            
        return [feat1, feat2, feat3, feat4]
    
    def gram_matrix(self, features):
        """Calcula matriz de Gram (captura estilo)"""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def learn_from_training_data(self, data_dir='data/train', max_samples=30):
        """Aprende características promedio del dataset de entrenamiento"""
        print(f"\n📚 Aprendiendo de: {data_dir}")
        
        # Buscar todas las imágenes
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(Path(data_dir).rglob(ext))
        
        image_paths = list(image_paths)[:max_samples]
        print(f"   Procesando {len(image_paths)} imágenes...")
        
        all_grams = [[] for _ in range(4)]  # 4 capas
        
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                features = self.extract_features(img_tensor)
                
                for layer_idx, feat in enumerate(features):
                    gram = self.gram_matrix(feat)
                    all_grams[layer_idx].append(gram)
                
                if (i + 1) % 10 == 0:
                    print(f"   Procesadas: {i + 1}/{len(image_paths)}")
                    
            except Exception as e:
                print(f"   ⚠️  Error en {img_path.name}: {e}")
                continue
        
        # Promediar todas las matrices de Gram
        self.style_grams = []
        for layer_grams in all_grams:
            if layer_grams:
                avg_gram = torch.stack(layer_grams).mean(dim=0)
                self.style_grams.append(avg_gram)
        
        print(f"   ✅ Estilo aprendido de {len(image_paths)} imágenes")
    
    def apply_to_image(self, input_path, output_path, iterations=200, style_weight=1e6):
        """Aplica el estilo aprendido a una nueva imagen"""
        print(f"\n🎨 Aplicando estilo a: {input_path}")
        
        # Cargar imagen
        img = Image.open(input_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        img_tensor = img_tensor.clone().requires_grad_(True)
        
        optimizer = torch.optim.LBFGS([img_tensor], max_iter=20)
        
        print(f"   Optimizando...")
        
        run = [0]
        while run[0] <= iterations:
            def closure():
                optimizer.zero_grad()
                
                # Extraer features
                x = self.model.backbone.conv1(img_tensor)
                x = self.model.backbone.bn1(x)
                x = self.model.backbone.relu(x)
                x = self.model.backbone.maxpool(x)
                
                feat1 = self.model.backbone.layer1(x)
                feat2 = self.model.backbone.layer2(feat1)
                feat3 = self.model.backbone.layer3(feat2)
                feat4 = self.model.backbone.layer4(feat3)
                
                features = [feat1, feat2, feat3, feat4]
                
                # Calcular pérdida de estilo
                style_loss = 0
                for feat, target_gram in zip(features, self.style_grams):
                    gram = self.gram_matrix(feat)
                    style_loss += nn.functional.mse_loss(gram, target_gram)
                
                loss = style_weight * style_loss
                loss.backward()
                
                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"   Iteración {run[0]}: loss={loss.item():.2f}")
                
                return loss
            
            optimizer.step(closure)
        
        # Guardar resultado
        output_tensor = img_tensor.squeeze(0).cpu().detach()
        
        # Desnormalizar
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        output_tensor = output_tensor * std + mean
        output_tensor = output_tensor.clamp(0, 1)
        
        result_img = self.to_pil(output_tensor)
        
        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_img.save(output_path)
        
        print(f"   ✅ Guardado: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Aprende de tu dataset y aplica a nuevas imágenes')
    parser.add_argument('--input', type=str, help='Imagen de entrada')
    parser.add_argument('--output', type=str, default='outputs/styled_result.jpg')
    parser.add_argument('--train-dir', type=str, default='data/train')
    parser.add_argument('--max-learn', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=200)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  🎨 APRENDIZAJE Y APLICACIÓN DE ESTILO")
    print("="*60)
    
    # Inicializar
    learner = StyleLearner()
    
    # Aprender del dataset de entrenamiento
    learner.learn_from_training_data(args.train_dir, args.max_learn)
    
    # Si no se especifica input, usar imagen de prueba
    if not args.input:
        test_imgs = list(Path('.').glob('*.png')) + list(Path('.').glob('*.jpg'))
        if test_imgs:
            args.input = str(test_imgs[0])
            print(f"\n💡 Usando imagen de prueba: {args.input}")
        else:
            print("\n❌ Especifica --input con la ruta de tu imagen")
            return
    
    # Aplicar estilo
    learner.apply_to_image(args.input, args.output, args.iterations)
    
    print("\n✅ ¡Completado!")
    print(f"📁 Resultado: {args.output}\n")


if __name__ == '__main__':
    main()
