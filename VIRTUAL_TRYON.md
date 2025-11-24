# 👔 Virtual Try-On: Probador Virtual de Ropa

## 🎯 ¿Qué es esto?

Sistema de IA que **aprende de catálogos de ropa** y **aplica los estilos a fotos de modelos**.

### Ejemplo de uso:

1. **Aprende** de un catálogo de camisas, pantalones, vestidos, etc.
2. **Comparte** una foto de un modelo
3. **La IA cambia** la ropa del modelo según lo aprendido

---

## 🚀 Inicio Rápido

### Paso 1: Entrenar modelo base (si no lo has hecho)

```bash
# Organiza tus datos de entrenamiento
python organize_data.py /ruta/a/tus/datos

# Entrena el modelo
python quick_start.py
```

### Paso 2: Usar Virtual Try-On

```bash
# Modo interactivo (recomendado)
python virtual_tryon.py

# O modo línea de comandos
python virtual_tryon.py --learn catalog/camisas --category camisas
python virtual_tryon.py --apply modelo.jpg --category camisas --output resultado.jpg
```

---

## 📁 Organizar tu Catálogo

### Opción A: Por categorías (recomendado)

```
catalog/
  camisas/
    camisa_roja.jpg
    camisa_azul.jpg
    camisa_rayas.jpg
  pantalones/
    pantalon_negro.jpg
    pantalon_jeans.jpg
  vestidos/
    vestido_floral.jpg
    vestido_negro.jpg
```

### Opción B: Categoría única

```
catalog/camisas/
  camisa1.jpg
  camisa2.jpg
  camisa3.jpg
```

---

## 💡 Ejemplos de Uso

### Ejemplo 1: Catálogo de Camisas

```bash
# 1. Aprender del catálogo
python virtual_tryon.py --learn catalog/camisas --category camisas

# 2. Aplicar a foto de modelo
python virtual_tryon.py --apply modelo1.jpg --category camisas --output modelo_camisa_nueva.jpg
```

### Ejemplo 2: Múltiples Categorías

```bash
# Aprender varias categorías
python virtual_tryon.py --learn catalog/camisas --category camisas
python virtual_tryon.py --learn catalog/pantalones --category pantalones
python virtual_tryon.py --learn catalog/vestidos --category vestidos

# Aplicar diferentes estilos a la misma imagen
python virtual_tryon.py --apply modelo.jpg --category camisas --output modelo_camisa.jpg
python virtual_tryon.py --apply modelo.jpg --category pantalones --output modelo_pantalon.jpg
```

### Ejemplo 3: Modo Interactivo

```bash
python virtual_tryon.py

# El sistema te guiará paso a paso:
# 1. Selecciona "Aprender de catálogo"
# 2. Ingresa la ruta de tu catálogo
# 3. Selecciona "Aplicar estilo a imagen"
# 4. Elige la foto del modelo
# 5. ¡Listo!
```

---

## 🎨 Cómo Funciona

### 1. **Fase de Aprendizaje**
```
Catálogo de Ropa → Modelo de IA → Extrae Características → Guarda Estilo
```

El modelo analiza:
- Colores dominantes
- Texturas y patrones
- Formas y siluetas
- Detalles y ornamentos

### 2. **Fase de Aplicación**
```
Foto de Modelo + Estilo Aprendido → Transfer de Estilo → Imagen Estilizada
```

El modelo:
- Detecta la figura humana
- Identifica áreas de ropa
- Aplica el estilo aprendido
- Mantiene la pose original

---

## 📊 Mejores Prácticas

### ✅ Para Mejores Resultados:

1. **Catálogo de Calidad**
   - Usa imágenes claras y bien iluminadas
   - Mínimo 5-10 imágenes por categoría
   - Ideal: 20-50 imágenes

2. **Variedad**
   - Incluye diferentes ángulos
   - Diferentes colores de la misma prenda
   - Diferentes estilos dentro de la categoría

3. **Imágenes de Prueba**
   - Fotos frontales funcionan mejor
   - Buena iluminación
   - Pose clara y visible

### ❌ Evitar:

- Imágenes borrosas o de baja calidad
- Catálogos con menos de 5 imágenes
- Mezclar categorías muy diferentes

---

## 🔧 Configuración Avanzada

### Entrenar Modelo Específico para Ropa

Si quieres entrenar un modelo específicamente para reconocer prendas:

```bash
# Organiza tus datos por tipo de prenda
data/
  train/
    camisas/
      camisa1.jpg
      camisa2.jpg
    pantalones/
      pantalon1.jpg
    vestidos/
      vestido1.jpg

# Entrena
python train.py --epochs 50 --batch-size 32
```

### Ajustar Parámetros del Modelo

Edita `src/inference/virtual_tryon.py`:

```python
# Cambiar dimensión del espacio de estilo
model = StyleTransferModel(base_model, style_dim=1024)  # Default: 512

# Ajustar frames extraídos de videos
dataset = VideoImageDataset(data_dir, frames_per_video=32)  # Default: 16
```

---

## 🎯 Casos de Uso

### 1. **E-commerce de Moda**
- Muestra cómo se vería la ropa en diferentes modelos
- Prueba virtual antes de comprar

### 2. **Diseño de Moda**
- Visualiza diseños en diferentes contextos
- Experimenta con combinaciones

### 3. **Personalización**
- Aplica estilos específicos a fotos personales
- Crea catálogos personalizados

### 4. **Producción Fotográfica**
- Ahorra tiempo en sesiones de fotos
- Prueba looks rápidamente

---

## 📚 API de Python

### Uso Programático

```python
from src.inference.virtual_tryon import VirtualTryOn

# Inicializar
tryon = VirtualTryOn(model_path='models/virtual_tryon.pt')

# Aprender de catálogo
tryon.learn_from_catalog('catalog/camisas', category_name='camisas')

# Guardar estilos
tryon.save_styles('models/my_styles.pt')

# Aplicar a imagen
styled_image = tryon.apply_to_image(
    image_path='modelo.jpg',
    category_name='camisas',
    output_path='resultado.jpg'
)

# Cargar estilos previamente aprendidos
tryon.load_styles('models/my_styles.pt')
```

### Ejemplo: Procesar Múltiples Imágenes

```python
from pathlib import Path
from src.inference.virtual_tryon import VirtualTryOn

tryon = VirtualTryOn(model_path='models/virtual_tryon.pt')
tryon.load_styles('models/learned_styles.pt')

# Procesar todas las imágenes en un directorio
input_dir = Path('modelos/')
output_dir = Path('outputs/virtual_tryon/')
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob('*.jpg'):
    output_path = output_dir / f"styled_{img_path.name}"
    tryon.apply_to_image(
        str(img_path),
        category_name='camisas',
        output_path=str(output_path)
    )
    print(f"✓ Procesado: {img_path.name}")
```

---

## 🔍 Solución de Problemas

### Error: "No se encontró modelo entrenado"

```bash
# Entrena primero un modelo base
python train.py --epochs 20
```

### Error: "Estilo 'X' no disponible"

```bash
# Aprende el estilo primero
python virtual_tryon.py --learn catalog/X --category X
```

### Resultados no satisfactorios

1. **Aumenta el catálogo**: Más imágenes = mejores resultados
2. **Mejora la calidad**: Usa imágenes de alta resolución
3. **Entrena más tiempo**: Más épocas en el entrenamiento base
4. **Ajusta parámetros**: Modifica `style_dim` en el modelo

### Error: "CUDA out of memory"

```python
# Usa CPU en lugar de GPU
tryon = VirtualTryOn(model_path='...', device='cpu')
```

---

## 🎓 Conceptos Técnicos

### Arquitectura

```
Modelo Base (ResNet50) 
    ↓
Style Encoder (extrae características del catálogo)
    ↓
Style Vector (representación del estilo)
    ↓
Style Decoder (aplica estilo a imagen objetivo)
    ↓
Imagen Estilizada
```

### Transfer Learning

El sistema usa transfer learning:
1. Modelo pre-entrenado en ImageNet
2. Fine-tuned en tu dataset
3. Style encoder aprende características específicas
4. Style decoder reconstruye con nuevo estilo

---

## 📈 Roadmap

Próximas mejoras:

- [ ] Mejor detección de áreas de ropa
- [ ] Soporte para múltiples prendas simultáneas
- [ ] Integración con modelos de segmentación
- [ ] UI web interactiva
- [ ] API REST para integración

---

## 🤝 Contribuir

¿Ideas para mejorar? ¡Abre un issue o pull request!

---

## 📄 Licencia

Ver LICENSE en el repositorio principal.

---

## 🆘 Ayuda

¿Problemas? Revisa:
- Esta guía completa
- `README.md` principal
- Issues en GitHub
- Ejemplos en `notebooks/`

---

¡Disfruta creando con Virtual Try-On! 👔✨
