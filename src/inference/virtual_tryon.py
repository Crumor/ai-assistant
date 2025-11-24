"""
Módulo de inferencia para aplicar estilos aprendidos a nuevas imágenes
Virtual Try-On y Style Transfer
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os


class StyleTransferModel(nn.Module):
    """
    Modelo para transferir estilos aprendidos a nuevas imágenes.
    Útil para virtual try-on (probador virtual de ropa).
    """
    
    def __init__(self, vision_model, style_dim=512):
        """
        Args:
            vision_model: Modelo de visión pre-entrenado
            style_dim: Dimensión del espacio de estilo
        """
        super(StyleTransferModel, self).__init__()
        
        self.vision_model = vision_model
        
        # Encoder de estilo (extrae características del catálogo)
        self.style_encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, style_dim),
            nn.Tanh()
        )
        
        # Decoder para aplicar estilo a imagen objetivo
        self.style_decoder = nn.Sequential(
            nn.Linear(2048 + style_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * 224 * 224),  # Reconstruir imagen
            nn.Tanh()
        )
    
    def encode_style(self, catalog_images):
        """
        Extraer representación de estilo del catálogo
        
        Args:
            catalog_images: Tensor [N, 3, H, W] con imágenes del catálogo
        
        Returns:
            style_vector: Tensor [style_dim] con representación del estilo
        """
        # Extraer features del catálogo
        features = self.vision_model.extract_features(catalog_images)
        
        # Ensure features has batch dimension
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        # Promediar features de todas las imágenes del catálogo
        avg_features = features.mean(dim=0)
        
        # Codificar estilo
        style = self.style_encoder(avg_features)
        
        return style
    
    def apply_style(self, target_image, style_vector):
        """
        Aplicar estilo a una imagen objetivo
        
        Args:
            target_image: Tensor [1, 3, H, W] imagen donde aplicar el estilo
            style_vector: Tensor [style_dim] estilo a aplicar
        
        Returns:
            styled_image: Tensor [1, 3, H, W] imagen con estilo aplicado
        """
        # Extraer features de la imagen objetivo
        target_features = self.vision_model.extract_features(target_image)
        
        # Expandir style_vector para concatenar
        if len(target_features.shape) == 1:
            target_features = target_features.unsqueeze(0)
        
        style_vector = style_vector.unsqueeze(0) if len(style_vector.shape) == 1 else style_vector
        
        # Concatenar features de la imagen con el vector de estilo
        combined = torch.cat([target_features, style_vector], dim=1)
        
        # Decodificar para obtener imagen estilizada
        styled = self.style_decoder(combined)
        
        # Reshape a imagen
        styled = styled.view(-1, 3, 224, 224)
        
        return styled
    
    def forward(self, catalog_images, target_image):
        """
        Pipeline completo: aprender del catálogo y aplicar a imagen objetivo
        """
        style = self.encode_style(catalog_images)
        result = self.apply_style(target_image, style)
        return result


class VirtualTryOn:
    """
    Sistema de probador virtual que aprende de un catálogo y aplica a modelos
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path: Ruta al modelo entrenado (opcional)
            device: 'cuda' o 'cpu' (None = auto-detect)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.catalog_styles = {}
        
        # Transformaciones estándar
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
        ])
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Cargar modelo pre-entrenado"""
        from src.models.vision_model import VisionModel
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Crear modelo base
        num_classes = checkpoint.get('num_classes', 10)
        base_model = VisionModel(num_classes=num_classes, pretrained=False)
        
        # Si el checkpoint tiene el modelo completo de style transfer
        if 'style_transfer_state_dict' in checkpoint:
            self.model = StyleTransferModel(base_model)
            self.model.load_state_dict(checkpoint['style_transfer_state_dict'])
        else:
            # Usar modelo de visión básico
            base_model.load_state_dict(checkpoint['model_state_dict'])
            self.model = StyleTransferModel(base_model)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Modelo cargado desde {model_path}")
    
    def learn_from_catalog(self, catalog_dir, category_name='default'):
        """
        Aprender estilos de un directorio de catálogo
        
        Args:
            catalog_dir: Directorio con imágenes del catálogo
            category_name: Nombre de la categoría (ej: 'camisas', 'pantalones')
        """
        if self.model is None:
            raise ValueError("Primero debes cargar un modelo con load_model()")
        
        print(f"📚 Aprendiendo de catálogo: {catalog_dir}")
        
        # Cargar imágenes del catálogo
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        catalog_images = []
        
        if not os.path.isdir(catalog_dir):
            raise ValueError(f"Catalog directory '{catalog_dir}' does not exist or is not a directory")
        
        for filename in os.listdir(catalog_dir):
            if filename.lower().endswith(image_exts):
                img_path = os.path.join(catalog_dir, filename)
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)
                catalog_images.append(img_tensor)
        
        if len(catalog_images) == 0:
            raise ValueError(f"No se encontraron imágenes en {catalog_dir}")
        
        print(f"  Encontradas {len(catalog_images)} imágenes")
        
        # Apilar imágenes
        catalog_batch = torch.stack(catalog_images).to(self.device)
        
        # Extraer estilo
        with torch.no_grad():
            style_vector = self.model.encode_style(catalog_batch)
        
        # Guardar estilo
        self.catalog_styles[category_name] = style_vector.cpu()
        
        print(f"✓ Estilo '{category_name}' aprendido")
        
        return style_vector
    
    def apply_to_image(self, image_path, category_name='default', output_path=None):
        """
        Aplicar estilo aprendido a una imagen nueva
        
        Args:
            image_path: Ruta de la imagen objetivo (ej: foto de un modelo)
            category_name: Categoría de estilo a aplicar
            output_path: Dónde guardar el resultado (opcional)
        
        Returns:
            styled_image: PIL Image con el estilo aplicado
        """
        if self.model is None:
            raise ValueError("Primero debes cargar un modelo con load_model()")
        
        if category_name not in self.catalog_styles:
            raise ValueError(f"Estilo '{category_name}' no aprendido. Usa learn_from_catalog() primero")
        
        print(f"🎨 Aplicando estilo '{category_name}' a {image_path}")
        
        # Cargar imagen objetivo
        target_img = Image.open(image_path).convert('RGB')
        target_tensor = self.transform(target_img).unsqueeze(0).to(self.device)
        
        # Obtener estilo
        style_vector = self.catalog_styles[category_name].to(self.device)
        
        # Aplicar estilo
        with torch.no_grad():
            styled_tensor = self.model.apply_style(target_tensor, style_vector)
        
        # Convertir a imagen
        styled_tensor = styled_tensor.squeeze(0).cpu()
        styled_tensor = self.inverse_transform(styled_tensor)
        styled_tensor = torch.clamp(styled_tensor, 0, 1)
        
        # Convertir a PIL
        styled_array = styled_tensor.permute(1, 2, 0).numpy()
        styled_image = Image.fromarray((styled_array * 255).astype(np.uint8))
        
        # Guardar si se especifica
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create if there's a directory component
                os.makedirs(output_dir, exist_ok=True)
            styled_image.save(output_path)
            print(f"✓ Resultado guardado en {output_path}")
        
        return styled_image
    
    def save_styles(self, styles_path='models/learned_styles.pt'):
        """Guardar estilos aprendidos"""
        os.makedirs(os.path.dirname(styles_path), exist_ok=True)
        torch.save(self.catalog_styles, styles_path)
        print(f"✓ Estilos guardados en {styles_path}")
    
    def load_styles(self, styles_path='models/learned_styles.pt'):
        """Cargar estilos previamente aprendidos"""
        if os.path.exists(styles_path):
            self.catalog_styles = torch.load(styles_path, map_location='cpu')
            print(f"✓ Estilos cargados desde {styles_path}")
            print(f"  Categorías disponibles: {list(self.catalog_styles.keys())}")
        else:
            print(f"⚠️  No se encontró archivo de estilos en {styles_path}")


def create_virtual_tryon_model(base_model_path, output_path='models/virtual_tryon.pt'):
    """
    Crear y guardar un modelo de virtual try-on desde un modelo base
    
    Args:
        base_model_path: Ruta al modelo de visión entrenado
        output_path: Dónde guardar el nuevo modelo
    """
    from src.models.vision_model import VisionModel
    
    print("🔧 Creando modelo de Virtual Try-On...")
    
    # Cargar modelo base
    try:
        checkpoint = torch.load(base_model_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"❌ Error al cargar el checkpoint del modelo base desde '{base_model_path}': {e}")
    num_classes = checkpoint.get('num_classes', 10)
    
    base_model = VisionModel(num_classes=num_classes, pretrained=False)
    try:
        base_model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        raise RuntimeError(f"❌ Error al cargar el state_dict del modelo base: {e}")
    
    # Crear modelo de style transfer
    style_model = StyleTransferModel(base_model)
    
    # Guardar
    new_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict'],
        'style_transfer_state_dict': style_model.state_dict(),
        'num_classes': num_classes,
        'val_acc': checkpoint.get('val_acc', 0)
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(new_checkpoint, output_path)
    
    print(f"✓ Modelo de Virtual Try-On guardado en {output_path}")
    
    return output_path
