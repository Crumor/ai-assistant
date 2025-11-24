#!/usr/bin/env python3
"""
Genera/modifica imágenes basándose en el CONTENIDO aprendido del dataset
No solo estilo visual, sino conceptos y objetos
"""

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.models.vision_model import VisionModel


class ContentBasedGenerator:
    def __init__(self, model_path='models/checkpoints/best_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Dispositivo: {self.device}")
        
        # Cargar tu modelo entrenado
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classifier = VisionModel(num_classes=2)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        print("📥 Cargando Stable Diffusion...")
        self.generator = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        ).to(self.device)
        
    def analyze_training_content(self, data_dir='data/train', max_samples=50):
        """Analiza QUÉ contienen las imágenes de entrenamiento"""
        print(f"\n🔍 Analizando contenido de: {data_dir}")
        
        from transformers import CLIPProcessor, CLIPModel
        
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(Path(data_dir).rglob(ext))
        
        image_paths = list(image_paths)[:max_samples]
        print(f"   Analizando {len(image_paths)} imágenes...")
        
        # Conceptos comunes a detectar
        concepts = [
            "person", "clothing", "fashion", "model", "portrait",
            "product", "object", "scene", "landscape", "indoor",
            "outdoor", "animal", "food", "building", "vehicle"
        ]
        
        concept_scores = {c: 0 for c in concepts}
        
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                inputs = clip_processor(
                    text=concepts,
                    images=img,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)[0]
                
                for i, concept in enumerate(concepts):
                    concept_scores[concept] += probs[i].item()
                    
            except Exception as e:
                continue
        
        # Conceptos dominantes
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        self.learned_concepts = [c for c, s in sorted_concepts[:5]]
        
        print(f"\n   ✅ Conceptos aprendidos: {', '.join(self.learned_concepts)}")
        return self.learned_concepts
    
    def generate_based_on_content(self, input_image, output_path, strength=0.75):
        """Genera imagen basándose en contenido aprendido"""
        print(f"\n🎨 Generando basado en contenido aprendido...")
        
        img = Image.open(input_image).convert('RGB')
        
        # Crear prompt basado en conceptos aprendidos
        prompt = f"high quality photo of {', '.join(self.learned_concepts)}, professional, detailed"
        
        print(f"   Prompt: {prompt}")
        print(f"   Strength: {strength} (0=original, 1=completamente nuevo)")
        
        result = self.generator(
            prompt=prompt,
            image=img,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=50
        ).images[0]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        
        print(f"   ✅ Guardado: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Genera basándose en contenido aprendido')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/generated_content.jpg')
    parser.add_argument('--strength', type=float, default=0.75, help='0-1: cuánto modificar')
    parser.add_argument('--max-learn', type=int, default=50)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  🧠 GENERACIÓN BASADA EN CONTENIDO APRENDIDO")
    print("="*60)
    
    generator = ContentBasedGenerator()
    generator.analyze_training_content(max_samples=args.max_learn)
    generator.generate_based_on_content(args.input, args.output, args.strength)
    
    print("\n✅ Completado!")
    print(f"📁 Resultado: {args.output}\n")


if __name__ == '__main__':
    main()
