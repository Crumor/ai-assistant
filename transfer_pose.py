#!/usr/bin/env python3
"""
Transfiere la pose/cuerpo de una imagen fuente a una imagen objetivo
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


class PoseTransfer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Dispositivo: {self.device}")
        
        # Cargar detector de pose
        print("📥 Cargando detector de pose...")
        from torchvision.models.detection import keypointrcnn_resnet50_fpn
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
        
        self.pose_model = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(self.device)
        self.pose_model.eval()
        print("✅ Modelo cargado")
    
    def extract_pose(self, image_path):
        """Extrae keypoints de pose de la imagen"""
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.pose_model(img_tensor)[0]
        
        if len(predictions['keypoints']) > 0:
            keypoints = predictions['keypoints'][0].cpu().numpy()
            scores = predictions['keypoints_scores'][0].cpu().numpy()
            box = predictions['boxes'][0].cpu().numpy()
            return img, keypoints, scores, box
        
        return img, None, None, None
    
    def transfer(self, source_path, target_path, output_path):
        """Transfiere pose de source a target"""
        print(f"\n🎯 Fuente: {source_path}")
        print(f"🎯 Objetivo: {target_path}")
        
        # Extraer poses
        print("\n📍 Extrayendo pose de fuente...")
        source_img, source_kp, source_scores, source_box = self.extract_pose(source_path)
        
        if source_kp is None:
            print("❌ No se detectó persona en imagen fuente")
            return
        
        print(f"   ✅ Detectados {len(source_kp)} keypoints")
        
        print("\n📍 Extrayendo pose de objetivo...")
        target_img, target_kp, target_scores, target_box = self.extract_pose(target_path)
        
        if target_kp is None:
            print("❌ No se detectó persona en imagen objetivo")
            return
        
        print(f"   ✅ Detectados {len(target_kp)} keypoints")
        
        # Transferir
        print("\n🔄 Transfiriendo pose...")
        
        # Extraer región de la persona fuente
        x1, y1, x2, y2 = map(int, source_box)
        source_person = source_img[y1:y2, x1:x2]
        
        # Redimensionar para ajustar a objetivo
        tx1, ty1, tx2, ty2 = map(int, target_box)
        target_h, target_w = ty2 - ty1, tx2 - tx1
        source_resized = cv2.resize(source_person, (target_w, target_h))
        
        # Crear resultado
        result = target_img.copy()
        
        # Reemplazar región
        result[ty1:ty2, tx1:tx2] = source_resized
        
        # Guardar
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, result)
        
        print(f"   ✅ Guardado: {output_path}")
        
        # Guardar visualización de keypoints
        vis_path = str(Path(output_path).parent / f"vis_{Path(output_path).name}")
        self._visualize_keypoints(result, target_kp, vis_path)
        print(f"   ✅ Visualización: {vis_path}")
    
    def _visualize_keypoints(self, img, keypoints, output_path):
        """Visualiza keypoints en la imagen"""
        vis = img.copy()
        
        # Conexiones de skeleton
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Cabeza
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Brazos
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Piernas
        ]
        
        # Dibujar conexiones
        for i, j in skeleton:
            if i < len(keypoints) and j < len(keypoints):
                pt1 = tuple(map(int, keypoints[i][:2]))
                pt2 = tuple(map(int, keypoints[j][:2]))
                if keypoints[i][2] > 0.5 and keypoints[j][2] > 0.5:
                    cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
        
        # Dibujar keypoints
        for kp in keypoints:
            if kp[2] > 0.5:
                pt = tuple(map(int, kp[:2]))
                cv2.circle(vis, pt, 3, (0, 0, 255), -1)
        
        cv2.imwrite(output_path, vis)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Transferir pose de una imagen a otra')
    parser.add_argument('--source', type=str, required=True, help='Imagen con la pose a copiar')
    parser.add_argument('--target', type=str, required=True, help='Imagen donde aplicar')
    parser.add_argument('--output', type=str, default='outputs/pose_transferred.jpg')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  🤸 TRANSFERENCIA DE POSE")
    print("="*60)
    
    transfer = PoseTransfer()
    transfer.transfer(args.source, args.target, args.output)
    
    print("\n✅ Completado!\n")


if __name__ == '__main__':
    main()
