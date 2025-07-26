# BurnIA-script

Este proyecto utiliza YOLOv7 para detectar quemaduras en imágenes médicas. A continuación, se explican los pasos para descargar, instalar y ejecutar el modelo.

## 1. Clonar el repositorio

```bash
git clone https://github.com/ivan7barragan/BurnIA-script
cd BurnIA-script
```

## 2. Instalar dependencias

Se recomienda usar un entorno virtual:

```bash
python3 -m venv burn-env
source burn-env/bin/activate
pip install -r requirements.txt
```

> Asegúrate de tener instalado Python 3.7 o superior y `pip`.

## 3. Modelos preentrenados

Puedes descargar los modelos `.pt` necesarios desde la sección de Releases:

- [`DataSetBurnIA.pt`](https://github.com/ivan7barragan/BurnIA-script/releases/download/v1.0.0/DataSetBurnIA.pt): Modelo entrenado para clasificación de quemaduras.
- [`traced_model.pt`](https://github.com/ivan7barragan/BurnIA-script/releases/download/v1.0.0/traced_model.pt): Versión optimizada para inferencia con TorchScript.

> Coloca estos archivos en la raíz del repositorio antes de ejecutar `detect.py`.

## 4. Ejecutar inferencia en una imagen

```bash
python3 detect.py --weights ./DataSetBurnIA.pt --source inference/images/3rd_degree_2.jpg --no-trace 2>/dev/null
```

## 5. Ver resultados

Las imágenes con las detecciones serán guardadas en:

```bash
./runs/detect/
```

Puedes abrirlas directamente desde un visor de imágenes o ejecutar:

```bash
xdg-open ./runs/detect/  # En Linux
```
