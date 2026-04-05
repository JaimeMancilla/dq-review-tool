# Cluster Reviewer

Herramienta local para revisar y corregir clusters de product matching.

---

## Instalación (una sola vez)

### 1. Instalar Python
Si no tienes Python instalado, descárgalo desde https://python.org (versión 3.9 o superior).

Verifica que esté instalado:
```
python --version
```

### 2. Abrir la carpeta en terminal
- **Mac**: abre Terminal, escribe `cd ` (con espacio), arrastra la carpeta `cluster_reviewer` al terminal y presiona Enter.
- **Windows**: abre la carpeta en el Explorador, haz clic en la barra de direcciones, escribe `cmd` y presiona Enter.

### 3. Instalar dependencias
```
pip install -r requirements.txt
```

---

## Uso diario

### 1. Iniciar la app
```
python app.py
```

Verás en el terminal:
```
==================================================
  Cluster Reviewer iniciado
  Abrir en browser: http://localhost:5000
==================================================
```

### 2. Abrir en el navegador
Ve a: **http://localhost:5000**

### 3. Flujo de revisión

**Paso 1 — Cargar archivo**
- Arrastra tu archivo Excel (.xlsx) o CSV al área de carga.
- La app identifica automáticamente el ancla de cada cluster (donde `item_index == new_cluster`).
- Marca revisión 1/0 comparando nombre y dirección vs el ancla.
- Puedes cambiar cualquier 1→0 o 0→1 haciendo clic en el botón circular de cada fila.

**Paso 2 — Revisar subgrupos incorrectos**
- Para cada cluster con incorrectos (rev=0), los agrupa en subgrupos por similitud entre ellos.
- Cada subgrupo tiene una **query SQL lista** para copiar y correr en tu BD PostgreSQL.

**Paso 3 — Subir CSV de BD**
- Corre la query en tu BD, exporta como CSV.
- Sube ese CSV en el subgrupo correspondiente.
- La app muestra los clusters encontrados. **Tú marcas cuáles son "cluster limpio"**.
- Haz clic en "Asignar clusters limpios marcados".
  - Si hay 1 cluster limpio → se asigna ese.
  - Si hay 2+ → se asigna como `("cluster1","cluster2")`.
  - Si no hay ninguno limpio → usa el botón "Crear nuevo cluster".

**Paso 4 — Descargar**
- Clic en "⬇ Descargar revisado" en la barra superior.
- Descarga el Excel original con las columnas `revision` y `correccion` añadidas.

---

## Lógica de identificación del ancla

El ancla de un `new_cluster` es el miembro cuyo `item_index` es igual al valor de `new_cluster`.  
Si el archivo no tiene columna `new_cluster`, se usa `cluster_index`.

---

## Lógica de nuevo cluster

Cuando no hay cluster limpio en la BD para un subgrupo:
- Se toma el `item_index` del miembro con menor valor del subgrupo.
- Se usa como nombre del nuevo cluster (ya que `item_index = app_id + "_" + store_id`).

---

## Detener la app
En el terminal donde corre: presiona `Ctrl + C`.
