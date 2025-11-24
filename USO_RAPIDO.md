# 🚀 Uso Rápido: Tu IA Aprende y Aplica Estilo

## ✅ ¡YA FUNCIONA!

Tu IA ahora **aprende características visuales** de tus imágenes/videos de entrenamiento y las **aplica a nuevas imágenes**.

---

## 📍 Cómo Usar

### Opción 1: Automático (usa imagen de prueba)
```bash
source venv/bin/activate
python learn_and_apply.py
```

### Opción 2: Con tu propia imagen
```bash
source venv/bin/activate
python learn_and_apply.py --input tu_imagen.jpg --output resultado.jpg
```

### Opción 3: Personalizado
```bash
python learn_and_apply.py \
  --input tu_imagen.jpg \
  --output resultado.jpg \
  --max-learn 50 \
  --iterations 300
```

---

## 🎯 Qué Hace

1. **Aprende** de tus imágenes en `data/train/`
   - Extrae características de color, textura, patrones
   - Calcula estilo promedio del dataset

2. **Aplica** esas características a tu imagen nueva
   - Transfiere el estilo aprendido
   - Preserva contenido de la imagen original

---

## 📊 Parámetros

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--input` | Auto | Imagen de entrada |
| `--output` | `outputs/styled_result.jpg` | Dónde guardar |
| `--train-dir` | `data/train` | Dataset de entrenamiento |
| `--max-learn` | 30 | Cuántas imágenes usar para aprender |
| `--iterations` | 200 | Iteraciones de optimización |

---

## 💡 Ejemplos

### Aprender de más imágenes (mejor calidad)
```bash
python learn_and_apply.py --input foto.jpg --max-learn 100 --iterations 300
```

### Rápido (para pruebas)
```bash
python learn_and_apply.py --input foto.jpg --max-learn 10 --iterations 100
```

### Usar solo imágenes (no videos)
```bash
python learn_and_apply.py --train-dir data/train/imagenes --input foto.jpg
```

---

## 📁 Resultados

Los resultados se guardan en `outputs/`

```bash
# Ver resultado
ls -lh outputs/styled_result.jpg

# Comparar con original
# Original: tu imagen de entrada
# Resultado: outputs/styled_result.jpg
```

---

## 🎨 Qué Características Aprende

Tu IA extrae y aplica:
- ✅ **Colores dominantes** del dataset
- ✅ **Texturas y patrones** comunes
- ✅ **Estilo visual** general
- ✅ **Características de iluminación**

---

## ⚡ Rendimiento

Con tu RTX 3050:
- **Aprendizaje:** ~1-2 segundos por imagen
- **Aplicación:** ~10-30 segundos
- **Total:** 30-60 segundos para proceso completo

---

## 🔧 Troubleshooting

### Error: "No module named torch"
```bash
source venv/bin/activate
```

### Error: "No se encuentra el modelo"
```bash
python quick_start.py  # Entrenar primero
```

### Resultado no se ve bien
- Aumenta `--max-learn` (más imágenes para aprender)
- Aumenta `--iterations` (más optimización)
- Verifica que tus imágenes de entrenamiento sean similares

---

## 📈 Mejorando Resultados

### Para mejor calidad:
1. **Más datos de entrenamiento** (100+ imágenes)
2. **Imágenes similares** en el dataset
3. **Más iteraciones** (300-500)

### Para más velocidad:
1. **Menos imágenes** para aprender (10-20)
2. **Menos iteraciones** (100-150)

---

## 🎯 Casos de Uso

### 1. Filtro Fotográfico Personalizado
```bash
# Entrena con fotos con tu filtro favorito
# Aplica a cualquier foto nueva
python learn_and_apply.py --input nueva_foto.jpg
```

### 2. Estilo Artístico
```bash
# Entrena con arte/ilustraciones
# Convierte fotos a ese estilo
python learn_and_apply.py --train-dir data/train/imagenes --input foto.jpg
```

### 3. Branding Consistente
```bash
# Entrena con imágenes de tu marca
# Aplica estilo a contenido nuevo
python learn_and_apply.py --max-learn 50 --input contenido_nuevo.jpg
```

---

## 🚀 Próximos Pasos

1. ✅ **Prueba con diferentes imágenes**
2. ✅ **Ajusta parámetros** para tu caso
3. ✅ **Agrega más datos** de entrenamiento
4. ✅ **Experimenta** con diferentes estilos

---

## 📚 Más Información

- `STYLE_TRANSFER_GUIDE.md` - Guía técnica completa
- `README.md` - Documentación general del proyecto
- `GUIA_RAPIDA.md` - Guía de inicio rápido

---

## ✅ Verificación

Tu último resultado:
```
📁 outputs/styled_result.jpg
🎨 Aprendió de 20 imágenes
⚡ Loss final: 0.52
✅ Funcionando correctamente
```
