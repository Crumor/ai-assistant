#!/usr/bin/env python3
"""
Virtual Try-On Básico
Sistema simple de cambio de ropa usando técnicas de computer vision

IMPORTANTE: Este es un prototipo básico. Para resultados profesionales
necesitas usar modelos especializados como HR-VITON o VTON-HD.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


class BasicVirtualTryOn:
    """
    Implementación básica de Virtual Try-On usando OpenCV
    
    Limitaciones:
    - Resultados no fotorealistas
    - Requiere imágenes con fondo simple
    - Mejores resultados con poses frontales
    """
    
    def __init__(self):
        print("\n" + "👔 "*20)
        print("   VIRTUAL TRY-ON BÁSICO")
        print("👔 "*20 + "\n")
        
        print("⚠️  NOTA: Este es un prototipo básico")
        print("   Para resultados profesionales usa HR-VITON o APIs\n")
        
        # Cargar detector de personas (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        self.body_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Si no está disponible, usar alternativa
        if self.body_cascade.empty():
            print("⚠️  Usando detección alternativa")
    
    def detect_person(self, image):
        """
        Detectar persona en la imagen
        
        Returns:
            (x, y, w, h) del bounding box o None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar cuerpos
        bodies = self.body_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(100, 100)
        )
        
        if len(bodies) > 0:
            # Devolver el más grande
            areas = [w * h for (x, y, w, h) in bodies]
            largest_idx = np.argmax(areas)
            return bodies[largest_idx]
        
        return None
    
    def segment_person_simple(self, image):
        """
        Segmentación simple usando GrabCut
        
        Returns:
            Máscara de la persona
        """
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Área inicial (centro de la imagen)
        h, w = image.shape[:2]
        rect = (w//4, h//8, w//2, 7*h//8)
        
        try:
            # GrabCut
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Crear máscara binaria
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            return mask2
        except:
            print("⚠️  Error en segmentación, usando máscara simple")
            # Máscara simple del centro
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h//8:7*h//8, w//4:3*w//4] = 1
            return mask
    
    def detect_clothing_region(self, image, person_mask):
        """
        Detectar región de la prenda (torso superior)
        
        Returns:
            (x, y, w, h) de la región de ropa
        """
        # Encontrar contornos de la persona
        contours, _ = cv2.findContours(
            person_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            h, w = image.shape[:2]
            return (w//4, h//4, w//2, h//3)
        
        # Obtener bounding box más grande
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        
        # Región del torso (tercio superior del cuerpo)
        cloth_y = y + int(h * 0.15)
        cloth_h = int(h * 0.4)
        
        return (x, cloth_y, w, cloth_h)
    
    def prepare_garment(self, garment_image, target_size):
        """
        Preparar prenda para superposición
        
        Args:
            garment_image: Imagen de la prenda
            target_size: (width, height) objetivo
        
        Returns:
            Prenda redimensionada y con máscara
        """
        # Redimensionar
        garment_resized = cv2.resize(garment_image, target_size)
        
        # Crear máscara simple (basada en fondo blanco)
        gray = cv2.cvtColor(garment_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Limpiar máscara
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return garment_resized, mask
    
    def blend_garment(self, person_image, garment_image, garment_mask, position):
        """
        Mezclar prenda con imagen de persona
        
        Args:
            person_image: Imagen de la persona
            garment_image: Imagen de la prenda
            garment_mask: Máscara de la prenda
            position: (x, y, w, h) donde colocar
        
        Returns:
            Imagen resultante
        """
        result = person_image.copy()
        x, y, w, h = position
        
        # Asegurar que está dentro de límites
        x = max(0, x)
        y = max(0, y)
        w = min(w, person_image.shape[1] - x)
        h = min(h, person_image.shape[0] - y)
        
        # Preparar prenda
        garment_resized, mask = self.prepare_garment(garment_image, (w, h))
        
        # Crear región de interés
        roi = result[y:y+h, x:x+w]
        
        # Invertir máscara
        mask_inv = cv2.bitwise_not(mask)
        
        # Extraer fondo
        background = cv2.bitwise_and(roi, roi, mask=mask_inv)
        
        # Extraer prenda
        foreground = cv2.bitwise_and(garment_resized, garment_resized, mask=mask)
        
        # Combinar
        combined = cv2.add(background, foreground)
        
        # Suavizar bordes
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        combined = (combined * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
        
        # Colocar en resultado
        result[y:y+h, x:x+w] = combined
        
        return result
    
    def try_on(self, person_image_path, garment_image_path, output_path='outputs/tryon_result.jpg'):
        """
        Proceso completo de Virtual Try-On
        
        Args:
            person_image_path: Ruta a imagen de la persona
            garment_image_path: Ruta a imagen de la prenda
            output_path: Dónde guardar resultado
        
        Returns:
            Imagen resultante
        """
        print(f"\n🔄 Procesando Try-On...")
        print(f"📸 Persona: {person_image_path}")
        print(f"👕 Prenda: {garment_image_path}")
        
        # Cargar imágenes
        person_img = cv2.imread(str(person_image_path))
        garment_img = cv2.imread(str(garment_image_path))
        
        if person_img is None or garment_img is None:
            print("❌ Error al cargar imágenes")
            return None
        
        print("✓ Imágenes cargadas")
        
        # 1. Segmentar persona
        print("🔍 Segmentando persona...")
        person_mask = self.segment_person_simple(person_img)
        print("✓ Persona detectada")
        
        # 2. Detectar región de ropa
        print("👔 Detectando región de ropa...")
        cloth_region = self.detect_clothing_region(person_img, person_mask)
        print(f"✓ Región detectada: {cloth_region}")
        
        # 3. Superponer prenda
        print("🎨 Aplicando prenda...")
        result = self.blend_garment(person_img, garment_img, person_mask, cloth_region)
        print("✓ Prenda aplicada")
        
        # 4. Guardar resultado
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"\n✅ Resultado guardado: {output_path}")
        
        return result
    
    def batch_try_on(self, person_image, garments_dir, output_dir='outputs/batch_tryon'):
        """
        Probar múltiples prendas en una persona
        
        Args:
            person_image: Ruta a imagen de persona
            garments_dir: Directorio con prendas
            output_dir: Directorio de salida
        """
        print(f"\n📦 Try-On por lotes")
        print(f"📸 Persona: {person_image}")
        print(f"👔 Catálogo: {garments_dir}\n")
        
        garments_path = Path(garments_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Buscar prendas
        garments = list(garments_path.glob('*.jpg')) + list(garments_path.glob('*.png'))
        
        if not garments:
            print(f"❌ No se encontraron prendas en {garments_dir}")
            return
        
        print(f"✓ Encontradas {len(garments)} prendas\n")
        
        # Procesar cada prenda
        for i, garment in enumerate(garments, 1):
            print(f"[{i}/{len(garments)}] Procesando {garment.name}...")
            
            output_file = output_path / f"result_{i:03d}_{garment.stem}.jpg"
            
            try:
                self.try_on(person_image, garment, output_file)
            except Exception as e:
                print(f"⚠️  Error con {garment.name}: {e}")
        
        print(f"\n✅ Procesamiento completado!")
        print(f"📁 Resultados en: {output_dir}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Virtual Try-On Básico')
    
    parser.add_argument('--person', type=str, help='Imagen de la persona')
    parser.add_argument('--garment', type=str, help='Imagen de la prenda')
    parser.add_argument('--catalog', type=str, help='Directorio con catálogo de prendas')
    parser.add_argument('--output', type=str, default='outputs/tryon_result.jpg',
                       help='Ruta de salida')
    
    args = parser.parse_args()
    
    # Crear sistema
    tryon = BasicVirtualTryOn()
    
    # Modo interactivo si no hay argumentos
    if not args.person:
        print("="*60)
        print("OPCIONES:")
        print("="*60)
        print("1. Try-On simple (1 persona + 1 prenda)")
        print("2. Try-On por lotes (1 persona + catálogo)")
        print("3. Demo con imágenes de ejemplo")
        print("4. Salir")
        
        try:
            choice = input("\nSelecciona opción (1-4): ").strip()
            
            if choice == '1':
                person = input("\n📸 Ruta imagen persona: ").strip()
                garment = input("👕 Ruta imagen prenda: ").strip()
                
                if not person or not garment:
                    print("❌ Rutas vacías")
                    return
                
                tryon.try_on(person, garment)
                
                # Abrir resultado
                import os
                os.system("xdg-open outputs/tryon_result.jpg 2>/dev/null || open outputs/tryon_result.jpg 2>/dev/null")
            
            elif choice == '2':
                person = input("\n📸 Ruta imagen persona: ").strip()
                catalog = input("📁 Ruta catálogo prendas: ").strip()
                
                if not person or not catalog:
                    print("❌ Rutas vacías")
                    return
                
                tryon.batch_try_on(person, catalog)
            
            elif choice == '3':
                print("\n💡 Para demo, necesitas:")
                print("  1. Crear data/catalog/ con prendas")
                print("  2. Crear data/models/ con modelos")
                print("\nEjemplo de estructura:")
                print("data/")
                print("  catalog/")
                print("    shirt1.jpg")
                print("    shirt2.jpg")
                print("  models/")
                print("    model1.jpg")
            
            elif choice == '4':
                print("\n👋 Hasta luego!")
            
            else:
                print("\n❌ Opción inválida")
        
        except KeyboardInterrupt:
            print("\n\n👋 Cancelado")
    
    else:
        # Modo por argumentos
        if args.catalog:
            # Batch mode
            tryon.batch_try_on(args.person, args.catalog)
        elif args.garment:
            # Single mode
            tryon.try_on(args.person, args.garment, args.output)
        else:
            print("❌ Debes especificar --garment o --catalog")


if __name__ == "__main__":
    main()
