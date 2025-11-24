# 🚀 Guía de Inicio Rápido - Virtual Try-On

## ¡Tu IA ya puede aprender de catálogos y aplicar estilos!

---

## 📝 Resumen en 30 segundos

**¿Qué hace?**
Tu IA ahora puede:
1. ✅ Aprender de catálogos de imágenes/videos (ej: ropa)
2. ✅ Aplicar lo aprendido a nuevas fotos (ej: modelos)
3. ✅ Cambiar/transformar elementos según el catálogo

**¿Cómo?**
```bash
python virtual_tryon.py
```
Y sigue el menú interactivo. ¡Así de simple!

---

## 🎯 Ejemplo Práctico: Catálogo de Camisas

### 1️⃣ Organiza tu catálogo
```
catalog/
  camisas/
    camisa1.jpg
    camisa2.jpg
    camisa3.jpg
    ... (mínimo 5-10 imágenes)
```

### 2️⃣ Instala dependencias (primera vez)
```bash
pip install -r requirements.txt
```

### 3️⃣ Entrena el modelo base (primera vez)
```bash
python quick_start.py
```
⏱️ Esto toma 10-30 minutos dependiendo de tu hardware.

### 4️⃣ ¡Usa Virtual Try-On!

**Opción A: Modo Interactivo (Recomendado)**
```bash
python virtual_tryon.py
```
1. Selecciona "1. Aprender de catálogo"
2. Ingresa: `catalog/camisas`
3. Selecciona "2. Aplicar estilo a imagen"
4. Ingresa: `foto_modelo.jpg`
5. ¡Listo! Tu imagen estará en `outputs/virtual_tryon/`

**Opción B: Línea de Comandos**
```bash
# Aprender del catálogo
python virtual_tryon.py --learn catalog/camisas --category camisas

# Aplicar a imagen
python virtual_tryon.py --apply foto_modelo.jpg --category camisas --output resultado.jpg
```

---

## 📚 ¿Primera Vez?

### Si no tienes un modelo entrenado:
```bash
# Organiza datos de entrenamiento
python organize_data.py /ruta/a/tus/imagenes

# Entrena (toma tiempo, pero solo se hace una vez)
python quick_start.py
```

### Si ya tienes un modelo entrenado:
```bash
# Directo a Virtual Try-On
python virtual_tryon.py
```

---

## 💡 Tips Rápidos

### ✅ Para Mejores Resultados:
- Usa 10-50 imágenes por categoría en tu catálogo
- Imágenes claras y bien iluminadas
- Variedad en colores y estilos

### 🎨 Múltiples Categorías:
```bash
# Aprende de varias categorías
python virtual_tryon.py --learn catalog/camisas --category camisas
python virtual_tryon.py --learn catalog/pantalones --category pantalones
python virtual_tryon.py --learn catalog/vestidos --category vestidos

# Aplica diferentes estilos a la misma imagen
python virtual_tryon.py --apply modelo.jpg --category camisas --output modelo_camisa.jpg
python virtual_tryon.py --apply modelo.jpg --category pantalones --output modelo_pantalon.jpg
```

---

## 🔍 Verificar Instalación

```bash
python test_virtual_tryon.py
```

Deberías ver:
```
✅ TODOS LOS TESTS PASARON
✅ ESTRUCTURA COMPLETA
```

---

## 📖 Documentación Completa

- **VIRTUAL_TRYON.md** - Guía completa con ejemplos
- **IMPLEMENTACION_RESUMEN.md** - Detalles técnicos
- **ejemplos_virtual_tryon.py** - Ejemplos interactivos

---

## ❓ Problemas Comunes

### "No se encontró modelo entrenado"
```bash
# Solución: Entrena primero
python quick_start.py
```

### "No module named 'torch'"
```bash
# Solución: Instala dependencias
pip install -r requirements.txt
```

### "Estilo 'X' no disponible"
```bash
# Solución: Aprende el estilo primero
python virtual_tryon.py --learn catalog/X --category X
```

---

## 🎉 ¡Eso es Todo!

En 3 comandos:
```bash
pip install -r requirements.txt    # 1. Instalar
python quick_start.py              # 2. Entrenar
python virtual_tryon.py            # 3. Usar
```

**¿Dudas?** Lee `VIRTUAL_TRYON.md` para más detalles.

**¡Disfruta tu nuevo sistema de Virtual Try-On!** 👔✨
