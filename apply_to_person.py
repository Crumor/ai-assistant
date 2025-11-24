#!/usr/bin/env python3
"""
Detecta personas en imagen y aplica características aprendidas de cuerpos del dataset
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


class PersonStyleTransfer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Dispositivo: {self.device}")
        
        # Detector de personas
        print("📥 Cargando detector de personas...")
        self.person_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )
        
        # Cargar modelo de segmentación de personas
        try:
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
            
            self.segmentation_model = maskrcnn_resnet50_fpn(
                weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            ).to(self.device)
            self.segmentation_model.eval()
            print("✅ Modelo de segmentación cargado")
        except Exception as e:
            print(f"⚠️  Segmentación avanzada no disponible: {e}")
            self.segmentation_model = None
    
    def detect_person(self, image_path):
        """Detecta personas en la imagen"""
        print(f"\n🔍 Detectando personas en: {image_path}")
        
        img = cv2.imread(str(image_path))
        if img is None:
            img_pil = Image.open(image_path).convert('RGB')
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar personas
        persons = self.person_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100)
        )
        
        if len(persons) > 0:
            print(f"   ✅ Detectadas {len(persons)} persona(s)")
            return img, persons
        
        # Si no detecta con Haar, usar segmentación
        if self.segmentation_model:
            return self._detect_with_maskrcnn(img)
        
        print("   ⚠️  No se detectaron personas, procesando imagen completa")
        return img, [(0, 0, img.shape[1], img.shape[0])]
    
    def _detect_with_maskrcnn(self, img):
        """Detecta personas con Mask R-CNN"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.segmentation_model(img_tensor)[0]
        
        # Filtrar solo personas (clase 1 en COCO)
        person_indices = predictions['labels'] == 1
        if person_indices.sum() > 0:
            boxes = predictions['boxes'][person_indices].cpu().numpy()
            persons = [(int(x1), int(y1), int(x2-x1), int(y2-y1)) 
                      for x1, y1, x2, y2 in boxes]
            print(f"   ✅ Detectadas {len(persons)} persona(s) con Mask R-CNN")
            return img, persons
        
        return img, [(0, 0, img.shape[1], img.shape[0])]
    
    def learn_from_bodies(self, data_dir='data/train', max_samples=30):
        """Aprende características de cuerpos del dataset"""
        print(f"\n📚 Aprendiendo de cuerpos en: {data_dir}")
        
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(Path(data_dir).rglob(ext))
        
        image_paths = list(image_paths)[:max_samples]
        
        body_features = []
        valid_count = 0
        
        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                persons = self.person_detector.detectMultiScale(gray, 1.1, 3)
                
                if len(persons) > 0:
                    # Extraer región de la persona
                    x, y, w, h = persons[0]
                    person_region = img[y:y+h, x:x+w]
                    
                    # Calcular características (histograma de color, textura)
                    hist = cv2.calcHist([person_region], [0, 1, 2], None, 
                                       [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    body_features.append(hist)
                    valid_count += 1
                    
            except Exception as e:
                continue
        
        if body_features:
            self.avg_body_features = np.mean(body_features, axis=0)
            print(f"   ✅ Aprendido de {valid_count} cuerpos")
        else:
            print("   ⚠️  No se detectaron cuerpos en el dataset")
            self.avg_body_features = None
    
    def apply_to_person(self, input_path, output_path):
        """Aplica características aprendidas a personas en la imagen"""
        print(f"\n🎨 Aplicando a: {input_path}")
        
        img, persons = self.detect_person(input_path)
        
        if self.avg_body_features is None:
            print("   ⚠️  No hay características aprendidas")
            return
        
        result = img.copy()
        
        for i, (x, y, w, h) in enumerate(persons):
            print(f"   Procesando persona {i+1}/{len(persons)}...")
            
            # Extraer región de la persona
            person_region = img[y:y+h, x:x+w]
            
            # Aplicar características aprendidas
            # Método 1: Ajuste de histograma
            person_hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)
            
            # Transferir características de color/textura
            alpha = 0.6  # Intensidad de aplicación
            
            # Calcular histograma objetivo
            target_hist = self.avg_body_features.reshape(8, 8, 8)
            
            # Aplicar transformación de color
            for c in range(3):
                channel = person_hsv[:, :, c]
                hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                cdf_normalized = cdf * hist.max() / cdf.max()
                
                # Aplicar ecualización adaptativa
                channel_eq = np.interp(channel.flatten(), bins[:-1], cdf_normalized)
                person_hsv[:, :, c] = channel_eq.reshape(channel.shape)
            
            # Convertir de vuelta
            person_modified = cv2.cvtColor(person_hsv, cv2.COLOR_HSV2BGR)
            
            # Blend con original
            person_modified = cv2.addWeighted(person_region, 1-alpha, person_modified, alpha, 0)
            
            # Reemplazar en imagen
            result[y:y+h, x:x+w] = person_modified
        
        # Guardar
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"   ✅ Guardado: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Aplica características de cuerpos aprendidos')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/person_styled.jpg')
    parser.add_argument('--train-dir', type=str, default='data/train')
    parser.add_argument('--max-learn', type=int, default=30)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  👤 APLICACIÓN A PERSONAS")
    print("="*60)
    
    processor = PersonStyleTransfer()
    processor.learn_from_bodies(args.train_dir, args.max_learn)
    processor.apply_to_person(args.input, args.output)
    
    print("\n✅ Completado!")
    print(f"📁 Resultado: {args.output}\n")


if __name__ == '__main__':
    main()
