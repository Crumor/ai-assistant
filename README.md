# AI Vision Assistant

Proyecto de IA para procesamiento de imágenes y videos.

## Objetivos
- Entrenar modelos con imágenes y videos
- Implementar reconocimiento visual similar a Meta AI
- Crear sistema multimodal (visión + texto)
- **Virtual Try-On**: Aprender de catálogos y aplicar estilos a imágenes

## Tecnologías
- Python 3.10+
- PyTorch / TensorFlow
- OpenCV
- Transformers (Hugging Face)

## Estructura del Proyecto
```
ai-assistant/
├── data/              # Datasets de imágenes y videos
├── models/            # Modelos entrenados
├── notebooks/         # Jupyter notebooks para experimentación
├── src/
│   ├── data/         # Procesamiento de datos
│   ├── models/       # Arquitecturas de modelos
│   ├── training/     # Scripts de entrenamiento
│   └── inference/    # Inferencia y predicción
├── tests/            # Tests unitarios
└── requirements.txt  # Dependencias
```

## Próximos pasos
1. Configurar entorno virtual
2. Instalar dependencias básicas
3. Descargar dataset inicial
4. Implementar pipeline de procesamiento
5. Entrenar modelo baseline

## 🆕 Nuevas Funcionalidades

### Virtual Try-On (Probador Virtual)
Sistema de IA que aprende de catálogos de ropa y aplica los estilos a fotos:

```bash
# Entrenar modelo base
python quick_start.py

# Usar Virtual Try-On
python virtual_tryon.py
```

**Ver guía completa**: [VIRTUAL_TRYON.md](VIRTUAL_TRYON.md)

### Características principales:
- 👔 Aprende de catálogos de ropa/imágenes
- 🎨 Aplica estilos aprendidos a nuevas imágenes
- 📚 Soporte para múltiples categorías (camisas, pantalones, etc.)
- 💾 Guarda y reutiliza estilos aprendidos
- 🖼️ Procesa imágenes y videos
