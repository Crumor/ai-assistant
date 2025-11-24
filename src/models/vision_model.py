"""
Modelo base de visión computacional usando PyTorch
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class VisionModel(nn.Module):
    """
    Modelo base para procesamiento de imágenes.
    Usa ResNet50 pre-entrenado como backbone.
    """
    
    def __init__(self, num_classes=1000, pretrained=True):
        super(VisionModel, self).__init__()
        
        # Cargar modelo pre-entrenado
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Obtener el número de features de la última capa
        num_features = self.backbone.fc.in_features
        
        # Reemplazar la última capa fully connected
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor de imágenes [batch_size, 3, height, width]
        
        Returns:
            Tensor de predicciones [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def extract_features(self, x):
        """
        Extraer features sin la capa de clasificación
        
        Args:
            x: Tensor de imágenes [batch_size, 3, height, width]
        
        Returns:
            Tensor de features [batch_size, 2048]
        """
        # Remover temporalmente la capa fc
        modules = list(self.backbone.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        features = feature_extractor(x)
        return features.squeeze()


class MultiModalModel(nn.Module):
    """
    Modelo multimodal para procesar imágenes y texto juntos.
    Similar a CLIP de OpenAI.
    """
    
    def __init__(self, vision_dim=2048, text_dim=768, embedding_dim=512):
        super(MultiModalModel, self).__init__()
        
        # Encoder de visión
        self.vision_encoder = VisionModel(num_classes=vision_dim)
        
        # Proyección de visión a espacio compartido
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Proyección de texto a espacio compartido
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
    
    def encode_image(self, image):
        """Codificar imagen a embedding"""
        features = self.vision_encoder.extract_features(image)
        return self.vision_projection(features)
    
    def encode_text(self, text_features):
        """Codificar texto a embedding"""
        return self.text_projection(text_features)
    
    def forward(self, image, text_features):
        """
        Forward pass para contrastive learning
        
        Args:
            image: Tensor de imágenes
            text_features: Features de texto pre-procesadas
        
        Returns:
            Similitud entre imágenes y texto
        """
        image_embeddings = self.encode_image(image)
        text_embeddings = self.encode_text(text_features)
        
        # Normalizar embeddings
        image_embeddings = nn.functional.normalize(image_embeddings, dim=-1)
        text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)
        
        # Calcular similitud
        logits = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature
        
        return logits


if __name__ == "__main__":
    # Ejemplo de uso
    print("Creando modelo de visión...")
    model = VisionModel(num_classes=10)
    
    # Crear input dummy
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Shape del output: {output.shape}")
    
    # Extraer features
    features = model.extract_features(dummy_input)
    print(f"Shape de features: {features.shape}")
