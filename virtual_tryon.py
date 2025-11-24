#!/usr/bin/env python3
"""
Virtual Try-On: Cambia la ropa de una persona por la ropa de otra imagen
Ejemplo: Persona con smoking → Persona con traje de baño
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import requests
from io import BytesIO


class VirtualTryOn:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Dispositivo: {self.device}")
        
        print("\n📥 Cargando modelos...")
        print("   Esto puede tardar la primera vez...")
        
        # Segmentación de personas
        from torchvision.models.segmentation import deeplabv3_resnet101
        from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
        
        self.segmentation = deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.DEFAULT
        ).to(self.device)
        self.segmentation.eval()
        
        # Detector de personas
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
        
        self.detector = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(self.device)
        self.detector.eval()
        
        print("✅ Modelos cargados")
    
    def segment_person(self, image_path):
        """Segmenta la persona y su ropa"""
        print(f"\n🔍 Segmentando: {image_path}")
        
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        # Preparar para segmentación
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.segmentation(input_tensor)['out'][0]
        
        # Clase 15 = persona en COCO
        person_mask = output.argmax(0) == 15
        person_mask = person_mask.cpu().numpy().astype(np.uint8) * 255
        
        # Detectar bounding box de persona
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            detections = self.detector(img_tensor)[0]
        
        # Filtrar personas
        person_boxes = detections['boxes'][detections['labels'] == 1].cpu().numpy()
        
        if len(person_boxes) > 0:
            box = person_boxes[0].astype(int)
            print(f"   ✅ Persona detectada en: {box}")
            return img_np, person_mask, box
        
        print("   ⚠️  No se detectó persona claramente")
        return img_np, person_mask, None
    
    def extract_clothing(self, image_path):
        """Extrae la región de ropa de la imagen"""
        print(f"\n👔 Extrayendo ropa de: {image_path}")
        
        img, mask, box = self.segment_person(image_path)
        
        if box is None:
            return None, None
        
        x1, y1, x2, y2 = box
        
        # Región del torso (aproximación)
        torso_y1 = y1 + int((y2 - y1) * 0.2)  # Después de la cabeza
        torso_y2 = y1 + int((y2 - y1) * 0.7)  # Antes de las piernas
        
        clothing_region = img[torso_y1:torso_y2, x1:x2]
        clothing_mask = mask[torso_y1:torso_y2, x1:x2]
        
        print(f"   ✅ Ropa extraída: {clothing_region.shape}")
        return clothing_region, (x1, torso_y1, x2, torso_y2)
    
    def apply_clothing(self, person_path, clothing_path, output_path):
        """Aplica la ropa de clothing_path a la persona en person_path"""
        print("\n" + "="*60)
        print("  👔 CAMBIANDO ROPA")
        print("="*60)
        
        # Extraer ropa de la imagen fuente
        clothing, _ = self.extract_clothing(clothing_path)
        
        if clothing is None:
            print("❌ No se pudo extraer ropa de la imagen fuente")
            return
        
        # Segmentar persona objetivo
        person_img, person_mask, person_box = self.segment_person(person_path)
        
        if person_box is None:
            print("❌ No se detectó persona en imagen objetivo")
            return
        
        x1, y1, x2, y2 = person_box
        
        # Región del torso en persona objetivo
        torso_y1 = y1 + int((y2 - y1) * 0.2)
        torso_y2 = y1 + int((y2 - y1) * 0.7)
        
        # Redimensionar ropa para ajustar
        target_h = torso_y2 - torso_y1
        target_w = x2 - x1
        
        clothing_resized = cv2.resize(clothing, (target_w, target_h))
        
        # Crear resultado
        result = person_img.copy()
        
        # Aplicar ropa con blending suave
        print("\n🎨 Aplicando ropa...")
        
        # Crear máscara de blend
        blend_mask = np.ones((target_h, target_w, 3), dtype=np.float32) * 0.85
        
        # Blend
        roi = result[torso_y1:torso_y2, x1:x2]
        blended = (clothing_resized * blend_mask + roi * (1 - blend_mask)).astype(np.uint8)
        result[torso_y1:torso_y2, x1:x2] = blended
        
        # Guardar
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_img = Image.fromarray(result)
        result_img.save(output_path)
        
        print(f"   ✅ Guardado: {output_path}")
        
        # Guardar visualización
        vis_path = str(Path(output_path).parent / f"vis_{Path(output_path).name}")
        vis = result.copy()
        cv2.rectangle(vis, (x1, torso_y1), (x2, torso_y2), (0, 255, 0), 2)
        cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"   ✅ Visualización: {vis_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Virtual Try-On: Cambiar ropa')
    parser.add_argument('--clothing', type=str, required=True, 
                       help='Imagen con la ropa que quieres (ej: traje de baño)')
    parser.add_argument('--person', type=str, required=True,
                       help='Imagen de la persona a cambiar (ej: con smoking)')
    parser.add_argument('--output', type=str, default='outputs/tryon_result.jpg')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  🎽 VIRTUAL TRY-ON")
    print("="*60)
    print(f"\n📸 Ropa fuente: {args.clothing}")
    print(f"👤 Persona: {args.person}")
    
    tryon = VirtualTryOn()
    tryon.apply_clothing(args.person, args.clothing, args.output)
    
    print("\n✅ ¡Completado!")
    print(f"📁 Resultado: {args.output}\n")


if __name__ == '__main__':
    main()
