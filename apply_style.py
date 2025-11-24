#!/usr/bin/env python3
"""
Aplicar características visuales aprendidas del dataset a nuevas imágenes
Usa el modelo entrenado para extraer y transferir estilo
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import argparse


class StyleExtractor:
    """Extrae características de estilo del modelo entrenado"""
    
    def __init__(self, model_path='models/checkpoints/best_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Usando: {self.device}")
        
        # Cargar modelo entrenado (VisionModel wrapper)
        from src.models.vision_model import VisionModel
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = VisionModel(num_classes=2)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Extraer capas intermedias del backbone
        self.feature_layers = nn.ModuleList([
            self.model.backbone.layer1,
            self.model.backbone.layer2,
            self.model.backbone.layer3,
            self.model.backbone.layer4
        ])
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path):
        """Extrae features de múltiples capas"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        features = []
        x = self.model.backbone.conv1(img_tensor)
        x = self.model.backbone.bn1(x)
        x = self.model.backbone.relu(x)
        x = self.model.backbone.maxpool(x)
        
        for layer in self.feature_layers:
            x = layer(x)
            features.append(x)
        
        return features
    
    def compute_gram_matrix(self, features):
        """Calcula matriz de Gram para capturar estilo"""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)


class StyleTransfer:
    """Aplica estilo aprendido a nueva imagen"""
    
    def __init__(self, extractor):
        self.extractor = extractor
        self.device = extractor.device
    
    def learn_from_dataset(self, data_dir='data/train/imagenes', max_images=50):
        """Aprende características del dataset"""
        print(f"\n📚 Aprendiendo de dataset: {data_dir}")
        
        image_paths = list(Path(data_dir).glob('*.jpg'))[:max_images]
        print(f"   Procesando {len(image_paths)} imágenes...")
        
        style_grams = []
        for img_path in image_paths:
            features = self.extractor.extract_features(img_path)
            grams = [self.extractor.compute_gram_matrix(f) for f in features]
            style_grams.append(grams)
        
        # Promediar características de todas las imágenes
        self.avg_style = []
        for layer_idx in range(len(style_grams[0])):
            layer_grams = torch.stack([g[layer_idx] for g in style_grams])
            self.avg_style.append(layer_grams.mean(dim=0))
        
        print("   ✅ Estilo aprendido")
        return self
    
    def apply_to_image(self, input_path, output_path, iterations=300, style_weight=1e6):
        """Aplica estilo aprendido a nueva imagen"""
        print(f"\n🎨 Aplicando estilo a: {input_path}")
        
        # Cargar imagen objetivo
        target_img = Image.open(input_path).convert('RGB')
        target_tensor = self.extractor.transform(target_img).unsqueeze(0).to(self.device)
        target_tensor.requires_grad_(True)
        
        optimizer = torch.optim.LBFGS([target_tensor])
        
        print(f"   Optimizando ({iterations} iteraciones)...")
        
        for i in range(iterations):
            def closure():
                optimizer.zero_grad()
                
                # Extraer features de imagen actual
                features = []
                x = self.extractor.model.backbone.conv1(target_tensor)
                x = self.extractor.model.backbone.bn1(x)
                x = self.extractor.model.backbone.relu(x)
                x = self.extractor.model.backbone.maxpool(x)
                
                for layer in self.extractor.feature_layers:
                    x = layer(x)
                    features.append(x)
                
                # Calcular pérdida de estilo
                style_loss = 0
                for feat, target_gram in zip(features, self.avg_style):
                    gram = self.extractor.compute_gram_matrix(feat)
                    style_loss += nn.functional.mse_loss(gram, target_gram)
                
                loss = style_weight * style_loss
                loss.backward()
                
                if i % 50 == 0:
                    print(f"   Iteración {i}: loss={loss.item():.2f}")
                
                return loss
            
            optimizer.step(closure)
        
        # Guardar resultado
        result = target_tensor.squeeze(0).cpu().detach()
        result = result.clamp(0, 1)
        result = transforms.ToPILImage()(result)
        result.save(output_path)
        
        print(f"   ✅ Guardado en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Aplicar estilo aprendido')
    parser.add_argument('--input', type=str, required=True, help='Imagen de entrada')
    parser.add_argument('--output', type=str, default='outputs/styled_output.jpg')
    parser.add_argument('--model', type=str, default='models/checkpoints/best_model.pt')
    parser.add_argument('--data-dir', type=str, default='data/train/imagenes')
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--max-learn', type=int, default=50, help='Máx imágenes para aprender')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  🎨 APLICADOR DE ESTILO APRENDIDO")
    print("="*60)
    
    # Crear directorio de salida
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Inicializar
    extractor = StyleExtractor(args.model)
    transfer = StyleTransfer(extractor)
    
    # Aprender del dataset
    transfer.learn_from_dataset(args.data_dir, args.max_learn)
    
    # Aplicar a nueva imagen
    transfer.apply_to_image(args.input, args.output, args.iterations)
    
    print("\n✅ Proceso completado")
    print(f"📁 Resultado: {args.output}\n")


if __name__ == '__main__':
    main()
