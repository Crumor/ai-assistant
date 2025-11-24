#!/usr/bin/env python3
"""
Virtual Try-On real usando IA generativa
Genera la ropa de forma realista, no solo pega imágenes
"""

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import cv2
from pathlib import Path


class AITryOn:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Dispositivo: {self.device}")
        
        print("\n📥 Cargando Stable Diffusion Inpainting...")
        print("   Primera vez: descarga ~5GB, puede tardar...")
        
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            safety_checker=None
        ).to(self.device)
        
        # Optimizar para GPU pequeña
        if self.device.type == 'cuda':
            self.pipe.enable_attention_slicing()
        
        print("✅ Modelo cargado")
    
    def detect_clothing_area(self, image_path):
        """Detecta área de ropa en la persona"""
        print(f"\n🔍 Detectando área de ropa en: {Path(image_path).name}")
        
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Usar segmentación simple para detectar torso
        from torchvision.models.segmentation import deeplabv3_resnet101
        from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
        from torchvision import transforms
        
        model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(self.device)
        model.eval()
        
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_pil = Image.fromarray(img_rgb)
        input_tensor = preprocess(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        
        # Máscara de persona
        person_mask = output.argmax(0) == 15
        person_mask = person_mask.cpu().numpy().astype(np.uint8) * 255
        
        # Crear máscara de torso (área de ropa)
        h, w = person_mask.shape
        torso_mask = np.zeros_like(person_mask)
        
        # Encontrar región de persona
        coords = np.where(person_mask > 0)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Región del torso (excluyendo cabeza y piernas)
            torso_y_start = y_min + int((y_max - y_min) * 0.15)
            torso_y_end = y_min + int((y_max - y_min) * 0.65)
            
            torso_mask[torso_y_start:torso_y_end, x_min:x_max] = 255
            
            # Suavizar bordes
            torso_mask = cv2.GaussianBlur(torso_mask, (21, 21), 0)
            
            print(f"   ✅ Área detectada: {x_max-x_min}x{torso_y_end-torso_y_start}px")
            return img_pil, Image.fromarray(torso_mask)
        
        print("   ⚠️  No se detectó persona")
        return img_pil, None
    
    def analyze_clothing(self, clothing_image_path):
        """Analiza la ropa en la imagen fuente para generar prompt"""
        print(f"\n👔 Analizando ropa en: {Path(clothing_image_path).name}")
        
        # Usar CLIP para entender qué tipo de ropa es
        from transformers import CLIPProcessor, CLIPModel
        
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        img = Image.open(clothing_image_path).convert('RGB')
        
        # Tipos de ropa a detectar
        clothing_types = [
            "swimsuit", "bikini", "swimming trunks",
            "suit", "tuxedo", "formal wear",
            "t-shirt", "shirt", "blouse",
            "dress", "skirt",
            "jacket", "coat",
            "casual wear", "sportswear"
        ]
        
        inputs = clip_processor(
            text=clothing_types,
            images=img,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
        
        # Top 3 tipos detectados
        top_indices = probs.argsort(descending=True)[:3]
        detected = [clothing_types[i] for i in top_indices]
        
        print(f"   ✅ Detectado: {', '.join(detected)}")
        return detected[0]
    
    def generate_tryon(self, person_path, clothing_path, output_path):
        """Genera el try-on usando IA"""
        print("\n" + "="*60)
        print("  🎨 GENERANDO VIRTUAL TRY-ON CON IA")
        print("="*60)
        
        # Detectar área de ropa en persona
        person_img, mask = self.detect_clothing_area(person_path)
        
        if mask is None:
            print("❌ No se pudo detectar persona")
            return
        
        # Analizar tipo de ropa
        clothing_type = self.analyze_clothing(clothing_path)
        
        # Crear prompt
        prompt = f"person wearing {clothing_type}, high quality photo, realistic, detailed clothing, professional photography"
        negative_prompt = "blurry, distorted, low quality, cartoon, drawing, deformed"
        
        print(f"\n🎨 Generando con prompt: '{prompt}'")
        print("   Esto puede tardar 30-60 segundos...")
        
        # Redimensionar para GPU pequeña
        person_img = person_img.resize((512, 512))
        mask = mask.resize((512, 512))
        
        # Generar
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=person_img,
            mask_image=mask,
            num_inference_steps=30,
            guidance_scale=7.5,
            strength=0.8
        ).images[0]
        
        # Guardar
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        
        print(f"\n   ✅ Guardado: {output_path}")
        
        # Guardar máscara para referencia
        mask_path = str(Path(output_path).parent / f"mask_{Path(output_path).name}")
        mask.save(mask_path)
        print(f"   ✅ Máscara: {mask_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Virtual Try-On con IA generativa')
    parser.add_argument('--clothing', type=str, required=True,
                       help='Imagen con la ropa (ej: traje de baño)')
    parser.add_argument('--person', type=str, required=True,
                       help='Imagen de la persona')
    parser.add_argument('--output', type=str, default='outputs/ai_tryon.jpg')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  🤖 VIRTUAL TRY-ON CON IA")
    print("="*60)
    
    tryon = AITryOn()
    tryon.generate_tryon(args.person, args.clothing, args.output)
    
    print("\n✅ ¡Completado!")
    print(f"📁 Resultado: {args.output}\n")


if __name__ == '__main__':
    main()
