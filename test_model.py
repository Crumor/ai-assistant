#!/usr/bin/env python3
"""
Script para probar el modelo entrenado con nuevas imágenes/videos
"""

import torch
import sys
from pathlib import Path
from PIL import Image
from torchvision import transforms
from src.models.vision_model import VisionModel


def load_trained_model(checkpoint_path='models/checkpoints/best_model.pt', num_classes=2):
    """Cargar el mejor modelo entrenado"""
    print(f"Cargando modelo desde {checkpoint_path}...")
    
    # Crear modelo
    model = VisionModel(num_classes=num_classes, pretrained=False)
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Modelo cargado (Accuracy: {checkpoint.get('val_acc', 0):.2f}%)")
    
    return model, checkpoint


def predict_image(model, image_path, class_names=['imagenes', 'videos']):
    """Predecir la clase de una imagen"""
    
    # Transformaciones (las mismas que en entrenamiento)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Cargar imagen
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error al cargar imagen: {e}")
        return None
    
    # Preprocesar
    input_tensor = transform(image).unsqueeze(0)
    
    # Predecir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
    
    predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Clase {predicted_idx}"
    
    return {
        'clase': predicted_class,
        'confianza': confidence * 100,
        'probabilidades': {class_names[i]: probabilities[i].item() * 100 for i in range(len(class_names))}
    }


def show_model_info(checkpoint_path='models/checkpoints/best_model.pt'):
    """Mostrar información del modelo entrenado"""
    print("\n" + "="*60)
    print("📊 INFORMACIÓN DEL MODELO")
    print("="*60 + "\n")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Época final: {checkpoint.get('epoch', 'N/A')}")
    print(f"Accuracy de validación: {checkpoint.get('val_acc', 0):.2f}%")
    
    history = checkpoint.get('history', {})
    
    if history:
        print(f"\nHistorial de entrenamiento:")
        print(f"  Épocas completadas: {len(history.get('train_loss', []))}")
        print(f"  Mejor train accuracy: {max(history.get('train_acc', [0])):.2f}%")
        print(f"  Mejor val accuracy: {max(history.get('val_acc', [0])):.2f}%")
        print(f"  Loss final (train): {history.get('train_loss', [0])[-1]:.6f}")
        print(f"  Loss final (val): {history.get('val_loss', [0])[-1]:.6f}")


def main():
    """Función principal"""
    print("\n" + "🤖 "*20)
    print("   PRUEBA TU MODELO ENTRENADO")
    print("🤖 "*20)
    
    # Mostrar info del modelo
    show_model_info()
    
    # Cargar modelo
    model, checkpoint = load_trained_model()
    
    # Clases (detectar automáticamente)
    class_names = ['imagenes', 'videos']  # Ajusta según tus clases
    
    print("\n" + "="*60)
    print("🔮 HACER PREDICCIONES")
    print("="*60)
    print("\nOpciones:")
    print("1. Probar con una imagen específica")
    print("2. Probar con todas las imágenes de validación")
    print("3. Salir")
    
    try:
        choice = input("\nSelecciona una opción (1-3): ").strip()
        
        if choice == '1':
            image_path = input("\nRuta de la imagen: ").strip()
            
            if not Path(image_path).exists():
                print(f"❌ La imagen {image_path} no existe")
                return
            
            print(f"\n🔍 Analizando {image_path}...")
            result = predict_image(model, image_path, class_names)
            
            if result:
                print(f"\n✓ Predicción: {result['clase']}")
                print(f"  Confianza: {result['confianza']:.2f}%")
                print(f"\n  Probabilidades por clase:")
                for cls, prob in result['probabilidades'].items():
                    print(f"    {cls}: {prob:.2f}%")
        
        elif choice == '2':
            print("\n🔍 Probando con imágenes de validación...")
            
            val_dir = Path('data/val')
            correct = 0
            total = 0
            
            for class_dir in val_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                true_class = class_dir.name
                
                for img_file in class_dir.glob('*.jpg'):
                    result = predict_image(model, str(img_file), class_names)
                    
                    if result:
                        total += 1
                        predicted_class = result['clase']
                        
                        is_correct = predicted_class == true_class
                        if is_correct:
                            correct += 1
                        
                        status = "✓" if is_correct else "✗"
                        print(f"  {status} {img_file.name}: {predicted_class} ({result['confianza']:.1f}%)")
            
            if total > 0:
                accuracy = (correct / total) * 100
                print(f"\n📊 Resultados:")
                print(f"  Correctas: {correct}/{total}")
                print(f"  Accuracy: {accuracy:.2f}%")
        
        elif choice == '3':
            print("\n👋 Hasta luego!")
            return
        
        else:
            print("\n❌ Opción inválida")
    
    except KeyboardInterrupt:
        print("\n\n👋 Cancelado")
        sys.exit(0)


if __name__ == "__main__":
    main()
