# Reglas de negocio — Product Matching & Data Quality

## Contexto

Un **cluster** agrupa las instancias del mismo local físico en distintas apps de delivery (Uber Eats, Rappi, Didi, Pedidos Ya, etc.). El proceso de clusterización es automático, pero genera errores que se corrigen manualmente con esta herramienta.

## Conceptos clave

| Campo | Descripción |
|-------|-------------|
| `cluster_index` | Identificador del cluster. Formato: `appId_storeId` del store ancla |
| `item_index` | Identificador de un store. Formato: `appId_storeId` |
| `store_id` | ID del store en su app de origen |
| `scraper_source` | App de origen (Uber Eats, Rappi, Didi, etc.) |
| `new_cluster` | Cluster al que pertenece actualmente el store |
| `store_ancla` | El store cuyo `item_index == cluster_index` |

### Regla base de creación de cluster
> El `cluster_index` siempre es igual al `item_index` del store ancla.

---

## Los 4 escenarios de corrección

### Escenario A — Reasignar a cluster limpio (existente)
**Situación:** Un store está en un cluster incorrecto. En BD existe un cluster limpio al que debería pertenecer.

**Flujo:**
1. Marcar el store como incorrecto (checkbox)
2. Generar subgrupos → Ejecutar query BD
3. Identificar el cluster limpio en los resultados
4. Marcar ese cluster como "limpio" → click en "★ Usar limpios"
5. La corrección se asigna automáticamente

**INSERT resultante:**
```sql
old_cluster = cluster_index_original_del_store
new_cluster = cluster_index_limpio
homologation_type = 'manual'
data_quality = true
```

---

### Escenario B — Fusionar clusters separados
**Situación:** La query BD devuelve 2+ clusters que representan el **mismo local físico** pero tienen distintos `cluster_index`. Deben unificarse.

**Reglas para determinar qué cluster absorbe (nuevo `cluster_index`):**
1. El que tenga **más miembros** absorbe al de menos
2. Si tienen igual cantidad → el que tenga el `appId` (primer segmento del `cluster_index`) **numéricamente menor**
3. Si hay que crear cluster nuevo → el store ancla será el que tenga el `item_index` numéricamente menor (primer segmento)

**Flujo:**
1. La app detecta automáticamente que hay 2+ clusters en los resultados BD
2. Muestra badge ⚠️ con el cluster sugerido según las reglas
3. Click en "✓ Aplicar fusión" asigna la corrección a todos los stores del subgrupo

**INSERT resultante:**
```sql
-- Para cada store de los clusters absorbidos:
old_cluster = cluster_index_absorbido
new_cluster = cluster_index_dominante  -- el que gana según las reglas
```

---

### Escenario C — Anotar store externo
**Situación:** Al revisar los resultados BD, se detecta un store que no estaba en el archivo de revisión original pero que también necesita corrección (está en un cluster sucio, debería estar en el limpio).

**Flujo:**
1. Identificar el store externo en los resultados BD
2. Click en "+" en la fila del store
3. Se abre el modal con la lista de clusters disponibles en la consulta
4. Seleccionar el cluster correcto (o "Crear nuevo")
5. La corrección se guarda en `external_corrections` y aparece en el INSERT

**INSERT resultante:**
```sql
old_cluster = cluster_index_donde_está_actualmente
new_cluster = cluster_index_correcto
scraper_source = scraper_source_del_store  -- viene de la tabla BD
```

---

### Escenario D — Crear nuevo cluster
**Situación:** No existe en BD ningún cluster que corresponda a los stores incorrectos. Hay que crear uno nuevo.

**Regla para el nuevo `cluster_index`:**
> El store ancla del nuevo cluster es el que tenga el `item_index` numéricamente menor (comparando el primer segmento, el `appId`).
> El nuevo `cluster_index` = `item_index` de ese store ancla.

**Flujo:**
1. Marcar stores incorrectos → Generar subgrupos → Ejecutar query BD
2. Verificar que no hay cluster limpio disponible
3. Click en "+ Nuevo" en el subgrupo
4. La app calcula automáticamente el nuevo `cluster_index`

**INSERT resultante:**
```sql
old_cluster = cluster_index_original
new_cluster = item_index_del_store_ancla_menor  -- nuevo cluster
```

---

## Tabla de destino en BD

```
sales_opportunity.ctrl_restaurant_homologation
```

| Campo | Valor | Descripción |
|-------|-------|-------------|
| `old_cluster` | `cluster_index` original | Cluster donde estaba el store |
| `new_cluster` | Corrección asignada | Cluster correcto |
| `store_id` | ID del store | Viene del archivo o de la query BD |
| `scraper_source` | App de origen | Uber Eats, Rappi, etc. |
| `country` | País | Detectado automáticamente del archivo |
| `load_date` | `NOW()` | Fecha de la corrección |
| `homologation_type` | `'manual'` | Siempre manual para correcciones DQ |
| `store_type` | `'restaurant'` / `'retail'` | Según el tipo elegido en la app |
| `data_quality` | `true` | Indica que la corrección fue hecha por el equipo DQ |

> **Nota:** La tabla se actualiza con INSERTs, no UPDATEs. Cada corrección es un registro histórico. Si un store fue corregido más de una vez, todos los registros quedan.

---

## Flujo general de revisión

```
Cargar archivo
    ↓
Marcar stores incorrectos
    ↓
Generar subgrupos (agrupados por similitud de marca)
    ↓
Ejecutar query en BD para cada subgrupo
    ↓
Identificar escenario (A / B / C / D)
    ↓
Asignar corrección
    ↓
[Opcional] Anotar stores externos (Escenario C)
    ↓
Descargar archivo revisado  ←→  Actualizar BD
    ↓
Marcar como revisado (registra en memoria interna)
```

---

## Stack técnico

| Componente | Tecnología |
|-----------|------------|
| Backend | Python / Flask |
| BD producción | PostgreSQL (`pimvault.gcp.gregario.com`) |
| Memoria interna | SQLite (`memory.db`) |
| Similitud semántica | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Frontend | JavaScript vanilla |

### Tablas en `memory.db`

| Tabla | Contenido |
|-------|-----------|
| `session_state` | Estado de revisión por archivo (se restaura al recargar) |
| `corrections` | Correcciones asignadas (memoria para sugerencias futuras) |
| `external_corrections` | Correcciones de stores externos (Escenario C) |
| `reviewed_clusters` | Registro de clusters que pasaron por revisión |
| `reviewed_files` | Resumen por archivo (total, ok, error, fecha) |
| `store_pairs` | Pares etiquetados de stores para futuro fine-tuning |
| `dish_pairs` | Pares etiquetados de platos para futuro fine-tuning |

---

*Última actualización: Abril 2026*
