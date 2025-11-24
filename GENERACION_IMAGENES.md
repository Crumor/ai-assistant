# 🎨 Generación de Imágenes con IA

## 🚀 Opción 1: Generador Simple (Recomendado para empezar)

**Sin GPU, gratis, sin registro:**

```bash
python generate_images_simple.py
```

### Características:
- ✓ Usa API gratuita (Pollinations.ai)
- ✓ No requiere GPU
- ✓ No requiere cuenta ni API key
- ✓ Genera en 10-30 segundos
- ✓ Acepta prompts en español e inglés

### Ejemplo:
```
📝 Describe la imagen: a cute cat wearing a wizard hat
🎨 Generando imagen...
✓ Guardada: outputs/generated/ai_generated_20251124_123456_1.png
✅ ¡Imagen generada exitosamente!
```

---

## 🎨 Opción 2: Stable Diffusion Local (Mejor calidad)

**Requiere GPU, mejor calidad:**

### 1. Instalar dependencias adicionales:
```bash
pip install diffusers transformers accelerate
```

### 2. Ejecutar:
```bash
python generate_images.py
```

### Características:
- ✓ Modelos de Stable Diffusion (alta calidad)
- ✓ Control total sobre parámetros
- ✓ Sin límites de uso
- ✗ Requiere GPU (mínimo 4 GB VRAM)
- ✗ Primera vez descarga ~5 GB

### Parámetros avanzados:
- **num_inference_steps**: Más pasos = mejor calidad (30-50)
- **guidance_scale**: Qué tan literal seguir el prompt (7-15)
- **width/height**: Resolución (512, 768, 1024)

---

## 💡 Tips para Mejores Prompts

### ✅ Buenos prompts:
```
"a photorealistic portrait of a cat, professional photography, high detail"
"beautiful landscape with mountains at sunset, digital art, 4k"
"futuristic cyberpunk city with neon lights, highly detailed"
"cute cartoon character, pixar style, colorful"
```

### ❌ Prompts vagos:
```
"cat" → Muy simple
"imagen bonita" → Muy vago
"cosa rara" → No descriptivo
```

### 🎯 Estructura recomendada:
```
[Sujeto principal] + [Estilo] + [Detalles] + [Calidad]

Ejemplo:
"ancient castle in the mountains, fantasy art style, 
 dramatic lighting, high detail, 4k quality"
```

---

## 🎨 Estilos populares:

| Estilo | Prompt Keywords |
|--------|----------------|
| Fotorealista | `photorealistic, professional photography, high detail` |
| Arte Digital | `digital art, artstation, concept art` |
| Pintura | `oil painting, brush strokes, artistic` |
| Anime/Manga | `anime style, manga, studio ghibli` |
| Cyberpunk | `cyberpunk, neon lights, futuristic` |
| Fantasía | `fantasy art, magical, epic` |
| Pixar/3D | `pixar style, 3d render, cartoon` |
| Steampunk | `steampunk, victorian, brass and copper` |

---

## 🔧 Solución de Problemas

### Error: "Out of memory" (GPU)
```bash
# Reduce resolución
width=512, height=512

# Reduce steps
num_inference_steps=20
```

### API no responde
```bash
# Verifica conexión a internet
ping image.pollinations.ai

# Intenta de nuevo en unos minutos
```

### Dependencias faltantes
```bash
# Instalar todo lo necesario
pip install diffusers transformers accelerate torch pillow requests
```

---

## 📊 Comparación de Opciones

| Característica | Simple (API) | Stable Diffusion Local |
|---------------|--------------|------------------------|
| Velocidad | ⭐⭐⭐ (10-30s) | ⭐⭐ (30-60s) |
| Calidad | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GPU Requerida | ❌ No | ✅ Sí (4+ GB) |
| Costo | 💰 Gratis | 💰 Gratis (electricidad) |
| Control | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Setup | ⭐⭐⭐⭐⭐ Fácil | ⭐⭐⭐ Medio |

---

## 🎯 Próximos Pasos

### Para generación básica:
```bash
python generate_images_simple.py
```

### Para producción/calidad:
1. Instalar dependencias pesadas
2. Descargar modelo la primera vez
3. Usar `generate_images.py`

### Para integrar en tu app:
```python
from generate_images_simple import SimpleImageGenerator

generator = SimpleImageGenerator()
images = generator.generate("your prompt here")
generator.save_images(images)
```

---

## 🌟 Ejemplos de Uso

### Generar avatar de personaje:
```
"portrait of a female warrior, fantasy armor, 
 detailed face, epic lighting, digital art"
```

### Generar paisaje:
```
"beautiful mountain landscape at golden hour, 
 lake reflection, photorealistic, 8k"
```

### Generar producto/diseño:
```
"modern minimalist chair design, 
 white background, product photography"
```

---

## 📚 Recursos Adicionales

- **Lexica.art**: Explora prompts de Stable Diffusion
- **PromptHero**: Biblioteca de prompts
- **Civitai**: Modelos custom de Stable Diffusion
- **Hugging Face**: Modelos y demos

---

¿Listo para generar tu primera imagen? 🎨
```bash
python generate_images_simple.py
```
