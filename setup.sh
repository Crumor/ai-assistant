#!/bin/bash

# Script de configuración inicial para AI Assistant

echo "🚀 Configurando entorno para AI Assistant..."

# Crear entorno virtual
echo "📦 Creando entorno virtual..."
python3 -m venv venv

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

# Crear estructura de directorios
echo "📁 Creando estructura de directorios..."
mkdir -p data/{raw,processed,train,val,test}
mkdir -p models/{checkpoints,pretrained}
mkdir -p notebooks
mkdir -p src/{data,models,training,inference,utils}
mkdir -p tests
mkdir -p logs
mkdir -p outputs

# Crear archivos __init__.py
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/inference/__init__.py
touch src/utils/__init__.py

echo "✅ Configuración completada!"
echo ""
echo "Para activar el entorno virtual, ejecuta:"
echo "source venv/bin/activate"
