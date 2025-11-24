#!/usr/bin/env python3
"""
Script para visualizar el progreso del entrenamiento
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def plot_training_history(history_file):
    """Visualizar historia de entrenamiento"""
    
    print(f"\n📊 Cargando historia desde {history_file}...")
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Entrenamiento del Modelo', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Pérdida (Loss)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_yscale('log')  # Escala logarítmica para loss
    
    # Plot 2: Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='Val Accuracy', linewidth=2, markersize=6)
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Precisión (Accuracy)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 105])
    
    # Añadir línea en 100%
    axes[1].axhline(y=100, color='g', linestyle='--', alpha=0.5, label='100%')
    
    plt.tight_layout()
    
    # Guardar
    output_file = 'outputs/training_history.png'
    Path('outputs').mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada en: {output_file}")
    
    # Mostrar
    plt.show()
    
    # Imprimir estadísticas
    print("\n" + "="*60)
    print("📈 ESTADÍSTICAS DEL ENTRENAMIENTO")
    print("="*60)
    print(f"\nÉpocas totales: {len(epochs)}")
    print(f"\nTrain Accuracy:")
    print(f"  Inicial: {history['train_acc'][0]:.2f}%")
    print(f"  Final: {history['train_acc'][-1]:.2f}%")
    print(f"  Mejor: {max(history['train_acc']):.2f}%")
    print(f"\nValidation Accuracy:")
    print(f"  Inicial: {history['val_acc'][0]:.2f}%")
    print(f"  Final: {history['val_acc'][-1]:.2f}%")
    print(f"  Mejor: {max(history['val_acc']):.2f}%")
    print(f"\nLoss Final:")
    print(f"  Train: {history['train_loss'][-1]:.2e}")
    print(f"  Val: {history['val_loss'][-1]:.2e}")
    
    # Verificar overfitting
    train_acc_final = history['train_acc'][-1]
    val_acc_final = history['val_acc'][-1]
    diff = train_acc_final - val_acc_final
    
    print(f"\n⚠️  Análisis de Overfitting:")
    print(f"  Diferencia Train-Val: {diff:.2f}%")
    
    if diff > 10:
        print("  ⚠️  Posible overfitting detectado")
        print("  💡 Recomendaciones:")
        print("     - Aumentar data augmentation")
        print("     - Conseguir más datos")
        print("     - Usar regularization (dropout)")
    elif diff < 2:
        print("  ✓ No hay overfitting significativo")
    else:
        print("  ✓ Overfitting mínimo (aceptable)")


def main():
    """Función principal"""
    print("\n" + "📊 "*20)
    print("   VISUALIZACIÓN DE ENTRENAMIENTO")
    print("📊 "*20)
    
    # Buscar archivos de historia
    log_dir = Path('logs')
    history_files = list(log_dir.glob('history_*.json'))
    
    if not history_files:
        print("\n❌ No se encontraron archivos de historia en logs/")
        return
    
    # Usar el más reciente
    latest_file = max(history_files, key=lambda p: p.stat().st_mtime)
    
    print(f"\n✓ Archivo más reciente: {latest_file.name}")
    
    # Visualizar
    plot_training_history(str(latest_file))
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Cancelado")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
