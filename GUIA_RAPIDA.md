# 🚀 Guía Rápida de Uso

## Ya tienes tus datos en `data/train`? ¡Perfecto!

### Opción 1: Inicio Ultra-Rápido (Recomendado)

```bash
python quick_start.py
```

Este script:
- ✓ Detecta automáticamente tu configuración
- ✓ Verifica tus datos
- ✓ Crea el split de validación si no existe
- ✓ Sugiere la mejor configuración para tu hardware
- ✓ Inicia el entrenamiento automáticamente

---

### Opción 2: Entrenamiento Manual

```bash
# Entrenamiento básico
python train.py

# Con parámetros personalizados
python train.py --epochs 50 --batch-size 32 --lr 0.001
```

#### Parámetros disponibles:

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--data-dir` | Directorio con los datos | `data` |
| `--epochs` | Número de épocas | `20` |
| `--batch-size` | Tamaño del batch | `32` |
| `--lr` | Learning rate | `0.001` |
| `--optimizer` | Optimizador (adam/sgd) | `adam` |
| `--pretrained` | Usar modelo pre-entrenado | `True` |
| `--image-size` | Tamaño de imágenes | `224` |

---

## 📁 Estructura de Datos Esperada

```
data/
├── train/
│   ├── clase1/
│   │   ├── imagen1.jpg
│   │   ├── imagen2.jpg
│   │   └── video1.mp4
│   ├── clase2/
│   │   ├── imagen1.jpg
│   │   └── video1.mp4
│   └── clase3/
│       └── ...
└── val/  (opcional, se crea automáticamente)
    ├── clase1/
    ├── clase2/
    └── clase3/
```

---

## 🎯 Ejemplos de Uso

### 1. Entrenamiento Rápido (10 épocas)
```bash
python train.py --epochs 10
```

### 2. Entrenamiento con GPU potente
```bash
python train.py --epochs 50 --batch-size 64 --num-workers 8
```

### 3. Entrenamiento sin modelo pre-entrenado (desde cero)
```bash
python train.py --no-pretrained --epochs 100
```

### 4. Ajuste fino con learning rate bajo
```bash
python train.py --lr 0.0001 --epochs 30
```

---

## 📊 Resultados del Entrenamiento

Después del entrenamiento encontrarás:

### 1. Checkpoints del modelo
```
models/checkpoints/
├── best_model.pt          # Mejor modelo (mayor accuracy)
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
└── ...
```

### 2. Logs de entrenamiento
```
logs/
└── history_YYYYMMDD_HHMMSS.json  # Historia completa
```

---

## 🔍 Verificar Progreso

Durante el entrenamiento verás:

```
Época 1/20
──────────────────────────────────────
Training: 100%|████████| 50/50 [00:45<00:00]
  loss: 0.8234  acc: 72.50%

Validation: 100%|████████| 10/10 [00:08<00:00]
  loss: 0.6891  acc: 78.30%

Resumen época 1:
  Train Loss: 0.8234 | Train Acc: 72.50%
  Val Loss:   0.6891 | Val Acc:   78.30%

✓ Mejor modelo guardado con accuracy: 78.30%
```

---

## 🎓 Tips para Mejor Entrenamiento

### 1. **Con GPU** (NVIDIA)
- Batch size: 32-64
- Num workers: 4-8
- Epochs: 20-50

### 2. **Sin GPU** (CPU)
- Batch size: 8-16
- Num workers: 2-4
- Epochs: 10-20 (será más lento)

### 3. **Pocos datos** (< 1000 imágenes)
- Usar `--pretrained` (transfer learning)
- Learning rate bajo: `--lr 0.0001`
- Data augmentation está activado automáticamente

### 4. **Muchos datos** (> 10,000 imágenes)
- Puedes aumentar batch size
- Considerar entrenar desde cero: `--no-pretrained`
- Más épocas: `--epochs 100`

---

## 🐛 Problemas Comunes

### "CUDA out of memory"
```bash
# Reduce el batch size
python train.py --batch-size 16
```

### "No module named ..."
```bash
# Activa el entorno virtual
source venv/bin/activate
pip install -r requirements.txt
```

### Dataset no encontrado
```bash
# Verifica que exista data/train/ con subdirectorios
ls -la data/train/
```

---

## 📈 Siguiente Paso: Evaluación

Una vez entrenado, puedes:

1. **Cargar el mejor modelo**
```python
import torch
from src.models.vision_model import VisionModel

model = VisionModel(num_classes=10)
checkpoint = torch.load('models/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

2. **Hacer predicciones**
```python
# Ver notebook: notebooks/01_getting_started.ipynb
```

3. **Visualizar resultados**
```python
from src.utils.visualization import plot_training_history

plot_training_history('logs/history_*.json')
```

---

## 🆘 Ayuda

¿Problemas? Revisa:
- README.md - Documentación completa
- notebooks/01_getting_started.ipynb - Tutorial interactivo
- Issues en GitHub (si aplica)

¡Buena suerte con tu entrenamiento! 🎉
