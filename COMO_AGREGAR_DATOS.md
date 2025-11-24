# 📚 Guía: Cómo Agregar Más Datos para Entrenar

## 🎯 Opción 1: Agregar a las clases existentes (Recomendado)

Si quieres que el modelo aprenda a distinguir mejor entre las mismas categorías:

```bash
# Simplemente agrega más archivos a las carpetas existentes:
data/
  train/
    imagenes/
      ├── [imágenes existentes...]
      ├── nueva_imagen1.jpg  ← AGREGAR AQUÍ
      ├── nueva_imagen2.jpg
      └── nueva_imagen3.jpg
    videos/
      ├── [videos existentes...]
      ├── nuevo_video1.mp4   ← AGREGAR AQUÍ
      └── nuevo_video2.mp4
```

**Luego entrena de nuevo:**
```bash
python train.py --epochs 30
```

---

## 🎯 Opción 2: Agregar nuevas categorías/clases

Si quieres enseñarle nuevas categorías (ej: "personas", "animales", "naturaleza"):

```bash
# Crea nuevas carpetas para cada clase:
data/
  train/
    imagenes/
    videos/
    personas/        ← NUEVA CLASE
      persona1.jpg
      persona2.jpg
    animales/        ← NUEVA CLASE
      gato1.jpg
      perro1.jpg
    naturaleza/      ← NUEVA CLASE
      paisaje1.jpg
```

**Importante**: También crea las mismas carpetas en `val/`:
```bash
mkdir -p data/val/personas data/val/animales data/val/naturaleza
```

---

## 🚀 Flujo recomendado:

### Paso 1: Organiza tus nuevos datos
```bash
# Opción A: Si están en otra carpeta, usa el organizador
python organize_data.py /ruta/a/nuevos/datos

# Opción B: Copia manualmente
cp /ruta/a/nuevas/imagenes/* data/train/imagenes/
cp /ruta/a/nuevos/videos/* data/train/videos/
```

### Paso 2: Crea el split train/val
```bash
# El organizador hace esto automáticamente (80% train, 20% val)
# O manualmente mueve 20% a val/
```

### Paso 3: Re-entrena desde el checkpoint anterior
```bash
# Continuar desde donde quedaste (transfer learning)
python train.py --epochs 30 --batch-size 16

# O entrenar desde cero con todos los datos
python train.py --epochs 50 --no-pretrained
```

---

## 💡 Tips importantes:

### ✅ **DO (Hacer)**
- ✓ Agregar datos variados (diferentes ángulos, iluminación, contextos)
- ✓ Mantener balance entre clases (similar cantidad en cada carpeta)
- ✓ Verificar que los archivos no estén corruptos
- ✓ Usar el script organizador para automatizar
- ✓ Guardar un 20% para validación

### ❌ **DON'T (No hacer)**
- ✗ Mezclar clases (ej: poner videos en carpeta "imagenes")
- ✗ Dejar una clase con muy pocos ejemplos (< 10)
- ✗ Agregar solo a train/ sin actualizar val/
- ✗ Duplicar exactamente las mismas imágenes

---

## 🔧 Script rápido para agregar datos:

```bash
# 1. Ver estado actual
python diagnostico.py

# 2. Agregar nuevos datos desde carpeta externa
python organize_data.py /path/to/new/data

# 3. Verificar que se agregaron correctamente
python diagnostico.py

# 4. Re-entrenar
python train.py --epochs 30
```

---

## 📊 Ejemplo práctico:

### Situación actual:
```
Train: 31 archivos (15 imágenes, 16 videos)
Val: 8 archivos (4 imágenes, 4 videos)
```

### Después de agregar 100 nuevas imágenes:
```
Train: ~111 archivos (95 imágenes, 16 videos)
Val: ~28 archivos (24 imágenes, 4 videos)
```

### El modelo mejorará porque:
- Más ejemplos = mejor generalización
- Menos overfitting
- Mayor accuracy en datos nuevos

---

## 🎯 ¿Cuántos datos necesitas?

| Complejidad | Mínimo recomendado por clase | Ideal |
|-------------|------------------------------|-------|
| Simple (2 clases muy diferentes) | 50-100 | 500+ |
| Media (5-10 clases) | 100-500 | 1000+ |
| Compleja (tipo Meta AI) | 1000+ | 100,000+ |

---

## 🚀 ¿Listo para agregar más datos?

1. **¿Dónde están tus nuevos datos?** → Dime la ruta
2. **¿Son nuevas clases o más ejemplos?** → Para organizar correctamente
3. **Ejecuto el organizador** → Automático

¿Quieres que te ayude a agregar los datos ahora?
