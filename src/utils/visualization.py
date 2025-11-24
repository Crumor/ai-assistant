"""
Utilidades para visualización de resultados
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os


def plot_training_history(history_path, save_path=None):
    """
    Plotear historia de entrenamiento
    
    Args:
        history_path: Ruta al archivo JSON con la historia
        save_path: Ruta opcional para guardar el plot
    """
    # Cargar historia
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Pérdida durante el entrenamiento')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Precisión durante el entrenamiento')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot guardado en: {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plotear matriz de confusión
    
    Args:
        cm: Matriz de confusión (numpy array)
        class_names: Nombres de las clases
        save_path: Ruta opcional para guardar
    """
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusión')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de confusión guardada en: {save_path}")
    
    plt.show()


def visualize_predictions(model, dataloader, device, num_images=16, class_names=None):
    """
    Visualizar predicciones del modelo
    
    Args:
        model: Modelo entrenado
        dataloader: DataLoader con imágenes
        device: Device (cuda/cpu)
        num_images: Número de imágenes a mostrar
        class_names: Nombres de las clases
    """
    model.eval()
    
    images_shown = 0
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Mover a CPU para visualización
            inputs = inputs.cpu()
            labels = labels.cpu()
            predicted = predicted.cpu()
            
            for i in range(inputs.size(0)):
                if images_shown >= num_images:
                    break
                
                # Denormalizar imagen
                img = inputs[i].permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                # Plotear
                axes[images_shown].imshow(img)
                axes[images_shown].axis('off')
                
                # Título con predicción
                label_text = class_names[labels[i]] if class_names else f"Clase {labels[i]}"
                pred_text = class_names[predicted[i]] if class_names else f"Clase {predicted[i]}"
                
                color = 'green' if labels[i] == predicted[i] else 'red'
                axes[images_shown].set_title(
                    f"Real: {label_text}\nPred: {pred_text}",
                    color=color,
                    fontsize=8
                )
                
                images_shown += 1
            
            if images_shown >= num_images:
                break
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import torch
    # Ejemplo de uso
    print("Módulo de visualización listo")
