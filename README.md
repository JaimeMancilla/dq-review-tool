# DQ Review Tool

Herramienta web local para revisión y corrección de calidad de datos en clusters de **product matching** — stores de delivery (restaurant y retail) y platos (dishes).


---

## Stack técnico

| Componente | Tecnología |
|-----------|------------|
| Backend | Python 3.9+ / Flask |
| BD producción | PostgreSQL (`pimvault.gcp.gregario.com`) |
| Memoria interna | SQLite (`memory.db`) |
| Similitud semántica | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Sesiones | `flask-session` (filesystem) |
| Frontend | JavaScript vanilla |

---

## Instalación (una sola vez)

### 1. Clonar el repositorio
```bash
git clone https://github.com/JaimeMancilla/stores-review.git
cd stores-review
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar credenciales PostgreSQL
Crea un archivo `.env` en la raíz del proyecto:
```
PG_HOST=pimvault.gcp.gregario.com
PG_PORT=5432
PG_DB=pimvault
PG_USER=tu_usuario
PG_PASS=tu_password
```

### 4. Descargar modelo semántico (una sola vez)
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
```
Una vez descargado, la app lo usa desde caché local sin conexión a internet.

---

## Uso diario

### Iniciar la app
```bash
python app.py
```
Abrir en el navegador: **http://localhost:5000**

### Detener la app
`Ctrl + C` en el terminal.

---

## Módulos

### 🏪 Stores (Restaurant / Retail)
Revisión de clusters de locales. Al seleccionar este módulo se elige el segmento:
- **Restaurant** → tabla `sales_opportunity.dim_maestra`
- **Retail** → tabla `sales_opportunity.dim_maestra_retail`

**Flujo:**
1. Cargar archivo `.xlsx` de revisión
2. Marcar stores incorrectos (checkbox)
3. Generar subgrupos (agrupados por similitud de marca)
4. Ejecutar query en BD para cada subgrupo
5. Identificar el escenario de corrección (ver `RULES.md`)
6. Asignar corrección
7. Descargar archivo revisado o hacer click en **"⬆ Actualizar BD"**

### 🍽️ Dishes
Revisión de grupos de platos por similitud semántica.

**Estados:**
- `1` = Correcto
- `0` = Incorrecto
- `2` = Incompleto (fusionar con otro grupo)

### 📊 Dashboard
Panel de progreso accesible desde el menú principal. Muestra:
- **Dishes**: archivos revisados, grupos ok/error/incompletos
- **Restaurant / Retail**: progreso vs BD, tiendas migradas, nuevos clusters, selector de país

---

## Los 4 escenarios de corrección

Ver documentación completa en [`RULES.md`](RULES.md).

| Escenario | Descripción |
|-----------|-------------|
| **A** | Store incorrecto → reasignar a cluster limpio existente |
| **B** | 2+ clusters separados que son el mismo local → fusionar según reglas |
| **C** | Store externo (no en el archivo) → anotar con "+" y asignar cluster |
| **D** | No existe cluster correcto → crear nuevo |

---

## Actualizar BD

El botón **"⬆ Actualizar BD"** (módulo Stores) muestra preview de los INSERTs e inserta en `sales_opportunity.ctrl_restaurant_homologation` con `homologation_type = 'manual'` y `data_quality = true`.

---

## Memoria interna (`memory.db`)

| Tabla | Contenido |
|-------|-----------|
| `session_state` | Estado de revisión por archivo (se restaura al recargar) |
| `corrections` | Correcciones asignadas (sugerencias futuras) |
| `external_corrections` | Correcciones de stores externos (Escenario C) |
| `reviewed_clusters` | Registro de clusters revisados |
| `reviewed_files` | Resumen por archivo (total, ok, error, fecha) |
| `store_pairs` | Pares etiquetados de stores para futuro fine-tuning |
| `dish_pairs` | Pares etiquetados de platos para futuro fine-tuning |

---

## Estructura del proyecto

```
stores-review/
├── app.py              # Backend Flask
├── templates/
│   └── index.html      # Frontend (HTML + JS + CSS)
├── requirements.txt
├── RULES.md            # Reglas de negocio y escenarios
├── .env                # Credenciales PG (no en repo)
├── memory.db           # SQLite local (no en repo)
└── flask_sessions/     # Sesiones servidor (no en repo)
```

---

## Licencia

Uso interno — Gregario © 2024-2026
