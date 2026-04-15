"""
scraper.py — Módulo de scraping de lugares para DQ Matching Tool
Fuentes:
  1. Overpass API (OpenStreetMap) → malls, strip centers, centros comerciales
  2. Wikidata SPARQL               → enriquecimiento de nombres canónicos (opcional)
  3. dim_maestra (PostgreSQL)      → detección de dark kitchens por concentración de marcas
"""

import re, time, sqlite3, unicodedata
from pathlib import Path

# ── Constantes ────────────────────────────────────────────────────────────────

OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
WIKIDATA_URL  = "https://query.wikidata.org/sparql"
DB_FILE       = Path("memory.db")

# Mapeo de tipo UI → tags OSM
PLACE_TYPE_OSM = {
    "mall":         [('shop', 'mall'), ('shop', 'supermarket')],
    "strip_center": [('shop', 'strip_mall'), ('landuse', 'retail')],
    "market":       [('amenity', 'marketplace'), ('shop', 'market')],
    "commercial":   [('landuse', 'commercial'), ('building', 'commercial')],
}

# Nombres de bounding boxes por país (para Overpass)
COUNTRY_BBOX = {
    "cl": (-56.0, -75.7, -17.5, -66.0),   # Chile     (S, W, N, E)
    "mx": (14.5,  -118.5, 32.7, -86.7),   # México
    "co": (-4.2,  -79.0,  12.5, -66.9),   # Colombia
    "pe": (-18.4, -81.4,  -0.0, -68.7),   # Perú
    "ar": (-55.1, -73.6,  -21.8, -53.6),  # Argentina
    "br": (-33.8, -73.9,   5.3, -28.8),   # Brasil
    "ec": (-5.0,  -81.0,   1.5, -75.2),   # Ecuador
    "bo": (-22.9, -69.7,  -9.7, -57.5),   # Bolivia
    "uy": (-34.9, -58.5, -30.1, -53.2),   # Uruguay
    "py": (-27.6, -62.6, -19.3, -54.3),   # Paraguay
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()

def _make_session():
    """Crea una sesión requests con headers realistas."""
    import requests
    s = requests.Session()
    s.headers.update({
        "User-Agent":    "DQMatchingTool/1.0 (internal data quality tool)",
        "Accept":        "application/json",
        "Accept-Language": "es-419,es;q=0.9",
    })
    return s

# ── SQLite: tabla places ──────────────────────────────────────────────────────

def init_places_db():
    """Crea la tabla places si no existe. Migra si ya hay versión anterior."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS places (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            place_type       TEXT NOT NULL,
            place_name       TEXT NOT NULL,
            place_address    TEXT DEFAULT '',
            commune          TEXT DEFAULT '',
            region           TEXT DEFAULT '',
            country          TEXT NOT NULL,
            latitude         REAL,
            longitude        REAL,
            osm_id           TEXT,
            source           TEXT DEFAULT 'overpass',
            raw_tags         TEXT DEFAULT '{}',
            scraped_at       REAL
        )
    """)
    # Índice para evitar duplicados OSM
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_places_osm
        ON places(osm_id) WHERE osm_id IS NOT NULL
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_places_country_type
        ON places(country, place_type)
    """)
    conn.commit()
    conn.close()

def places_upsert(rows):
    """
    Inserta o actualiza registros en places.
    rows: lista de dicts con claves del schema.
    Retorna (inserted, updated, skipped).
    """
    conn   = sqlite3.connect(DB_FILE)
    ins = upd = skip = 0
    for r in rows:
        osm_id = r.get("osm_id")
        # Si tiene osm_id, intentar actualizar si ya existe
        if osm_id:
            existing = conn.execute(
                "SELECT id FROM places WHERE osm_id=?", (osm_id,)
            ).fetchone()
            if existing:
                conn.execute("""
                    UPDATE places SET
                        place_name=?, place_address=?, commune=?, region=?,
                        latitude=?, longitude=?, raw_tags=?, scraped_at=?
                    WHERE osm_id=?
                """, (
                    r.get("place_name",""), r.get("place_address",""),
                    r.get("commune",""),    r.get("region",""),
                    r.get("latitude"),      r.get("longitude"),
                    r.get("raw_tags","{}"), time.time(),
                    osm_id
                ))
                upd += 1
                continue

        try:
            conn.execute("""
                INSERT INTO places
                  (place_type, place_name, place_address, commune, region,
                   country, latitude, longitude, osm_id, source, raw_tags, scraped_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                r.get("place_type",""),    r.get("place_name",""),
                r.get("place_address",""), r.get("commune",""),
                r.get("region",""),        r.get("country",""),
                r.get("latitude"),         r.get("longitude"),
                osm_id,                    r.get("source","overpass"),
                r.get("raw_tags","{}"),    time.time(),
            ))
            ins += 1
        except sqlite3.IntegrityError:
            skip += 1

    conn.commit()
    conn.close()
    return ins, upd, skip

def places_list(country=None, place_type=None, query=None, limit=500):
    """Consulta places con filtros opcionales."""
    conn   = sqlite3.connect(DB_FILE)
    where  = ["1=1"]
    params = []
    if country:
        where.append("country=?"); params.append(country.lower())
    if place_type:
        where.append("place_type=?"); params.append(place_type)
    if query:
        like = f"%{query}%"
        where.append("(place_name LIKE ? OR commune LIKE ? OR place_address LIKE ?)")
        params.extend([like, like, like])
    sql = f"""
        SELECT id, place_type, place_name, place_address, commune, region,
               country, latitude, longitude, osm_id, source, scraped_at
        FROM places
        WHERE {' AND '.join(where)}
        ORDER BY country, place_type, place_name
        LIMIT ?
    """
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    cols = ["id","place_type","place_name","place_address","commune","region",
            "country","latitude","longitude","osm_id","source","scraped_at"]
    return [dict(zip(cols, r)) for r in rows]

def places_delete(place_id):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM places WHERE id=?", (place_id,))
    conn.commit(); conn.close()

def places_stats():
    """Resumen por país y tipo."""
    conn  = sqlite3.connect(DB_FILE)
    rows  = conn.execute("""
        SELECT country, place_type, COUNT(*) as cnt
        FROM places
        GROUP BY country, place_type
        ORDER BY country, cnt DESC
    """).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    conn.close()
    breakdown = {}
    for country, ptype, cnt in rows:
        if country not in breakdown:
            breakdown[country] = {}
        breakdown[country][ptype] = cnt
    return {"total": total, "breakdown": breakdown}

# ── Overpass: scraping de malls y centros comerciales ────────────────────────

def _overpass_query(osm_tag_key, osm_tag_val, bbox, timeout=60):
    """
    Ejecuta una consulta Overpass para nodos y ways con un tag específico
    dentro de un bbox (S, W, N, E).
    Retorna lista de elementos OSM con tags.
    """
    import requests, json

    s, w, n, e = bbox
    # Consulta: nodos + ways (no relations, son más pesadas)
    query = f"""
    [out:json][timeout:{timeout}];
    (
      node["{osm_tag_key}"="{osm_tag_val}"]({s},{w},{n},{e});
      way["{osm_tag_key}"="{osm_tag_val}"]({s},{w},{n},{e});
    );
    out center tags;
    """
    sess = _make_session()
    try:
        resp = sess.post(OVERPASS_URL, data={"data": query}, timeout=timeout + 10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("elements", [])
    except Exception as e:
        return []

def _extract_place(elem, place_type, country):
    """Transforma un elemento OSM en un dict normalizado para places."""
    import json
    tags = elem.get("tags", {})

    # Nombre: priorizar nombre en español
    name = (tags.get("name:es") or tags.get("name") or
            tags.get("brand") or tags.get("official_name") or "").strip()
    if not name:
        return None

    # Coordenadas: ways tienen 'center', nodes tienen lat/lon directos
    lat = elem.get("lat") or (elem.get("center") or {}).get("lat")
    lon = elem.get("lon") or (elem.get("center") or {}).get("lon")

    # Dirección
    addr_parts = []
    for key in ["addr:street", "addr:housenumber"]:
        v = tags.get(key, "").strip()
        if v: addr_parts.append(v)
    address = " ".join(addr_parts)

    commune = (tags.get("addr:city") or tags.get("addr:suburb") or
               tags.get("addr:quarter") or "").strip()
    region  = (tags.get("addr:state") or tags.get("addr:province") or
               tags.get("addr:region") or "").strip()

    return {
        "place_type":    place_type,
        "place_name":    name,
        "place_address": address,
        "commune":       commune,
        "region":        region,
        "country":       country,
        "latitude":      float(lat) if lat else None,
        "longitude":     float(lon) if lon else None,
        "osm_id":        f"{elem.get('type','x')[0]}{elem.get('id','')}",
        "source":        "overpass",
        "raw_tags":      json.dumps(tags, ensure_ascii=False),
    }

def scrape_overpass(place_type, country, city=None, progress_cb=None):
    """
    Scraping principal desde Overpass.
    - place_type: clave de PLACE_TYPE_OSM ('mall', 'strip_center', etc.)
    - country:    código ISO 2 en minúsculas
    - city:       nombre de ciudad para acotar bbox (usa Nominatim si se provee)
    - progress_cb: callable(msg) para streaming de progreso

    Retorna dict {inserted, updated, skipped, total_found, errors}
    """
    tag_pairs = PLACE_TYPE_OSM.get(place_type)
    if not tag_pairs:
        return {"error": f"place_type desconocido: {place_type}"}

    # Resolver bbox
    if city:
        bbox = _city_bbox(city, country)
        if bbox is None:
            return {"error": f"No se pudo geolocalizar '{city}' en {country}"}
    else:
        bbox = COUNTRY_BBOX.get(country)
        if bbox is None:
            return {"error": f"País '{country}' no soportado aún"}

    if progress_cb: progress_cb(f"Bbox: {bbox}")

    all_elements = []
    for tag_key, tag_val in tag_pairs:
        if progress_cb: progress_cb(f"Consultando OSM: [{tag_key}={tag_val}]…")
        elems = _overpass_query(tag_key, tag_val, bbox)
        if progress_cb: progress_cb(f"  → {len(elems)} elementos")
        all_elements.extend(elems)

    # Deduplicar por osm_id
    seen = set()
    unique = []
    for e in all_elements:
        eid = f"{e.get('type','x')[0]}{e.get('id','')}"
        if eid not in seen:
            seen.add(eid)
            unique.append(e)

    # Convertir a dicts normalizados
    records = []
    for e in unique:
        r = _extract_place(e, place_type, country.lower())
        if r: records.append(r)

    if progress_cb: progress_cb(f"Procesados: {len(records)} lugares válidos")

    ins, upd, skip = places_upsert(records)
    return {"inserted": ins, "updated": upd, "skipped": skip,
            "total_found": len(records)}

def _city_bbox(city_name, country, padding=0.2):
    """
    Usa Nominatim para obtener el bbox de una ciudad.
    Retorna (S, W, N, E) o None.
    """
    import requests
    country_codes = {
        "cl":"CL","mx":"MX","co":"CO","pe":"PE","ar":"AR",
        "br":"BR","ec":"EC","bo":"BO","uy":"UY","py":"PY",
    }
    cc = country_codes.get(country.lower(), "")
    params = {
        "q":             city_name,
        "format":        "json",
        "limit":         1,
        "countrycodes":  cc,
        "addressdetails": 0,
    }
    sess = _make_session()
    sess.headers["User-Agent"] = "DQMatchingTool/1.0 (internal)"
    try:
        resp = sess.get("https://nominatim.openstreetmap.org/search",
                        params=params, timeout=10)
        data = resp.json()
        if not data: return None
        bb = data[0].get("boundingbox")  # [S, N, W, E]
        if not bb or len(bb) < 4: return None
        s, n, w, e = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        # Agregar padding para capturar suburbios
        return (s - padding, w - padding, n + padding, e + padding)
    except Exception:
        return None

# ── Dark kitchens desde dim_maestra ──────────────────────────────────────────

def scrape_dark_kitchens_db(pg_query_fn, country, min_brands=4,
                             min_stores=5, progress_cb=None):
    """
    Detecta posibles dark kitchens minando dim_maestra.
    Lógica: agrupar stores por cluster_address normalizada; si en una misma
    dirección aparecen >= min_brands marcas distintas (brand_words distintos)
    con >= min_stores stores en total → probable dark kitchen.

    pg_query_fn: la función pg_query de app.py
    Retorna dict {inserted, candidates_found, errors}
    """
    if progress_cb: progress_cb("Consultando dim_maestra para dark kitchens…")

    # Traemos stores agrupadas por cluster_address
    sql = f"""
        SELECT
            cluster_address,
            cluster_latitude,
            cluster_longitude,
            COUNT(DISTINCT item_index)                     AS total_stores,
            COUNT(DISTINCT cluster_index)                  AS total_clusters,
            array_agg(DISTINCT cluster_name ORDER BY cluster_name) AS names
        FROM sales_opportunity.dim_maestra
        WHERE country = '{country}'
          AND cluster_address IS NOT NULL
          AND cluster_address != ''
          AND cluster_latitude IS NOT NULL
        GROUP BY cluster_address, cluster_latitude, cluster_longitude
        HAVING COUNT(DISTINCT cluster_index) >= {min_brands}
           AND COUNT(DISTINCT item_index)    >= {min_stores}
        ORDER BY total_clusters DESC
        LIMIT 500
    """
    rows, err = pg_query_fn(sql, timeout_ms=30000)
    if err:
        return {"error": err, "inserted": 0, "candidates_found": 0}

    if not rows:
        return {"inserted": 0, "candidates_found": 0}

    if progress_cb:
        progress_cb(f"Candidatos encontrados: {len(rows)}")

    # Construir registros para places
    records = []
    for r in rows:
        names   = r.get("names") or []
        # Nombre representativo: el más frecuente / el primero
        rep_name = names[0] if names else r.get("cluster_address", "")
        address  = r.get("cluster_address", "")

        # Intentar extraer comuna de la dirección (heurística simple)
        commune = _guess_commune(address, country)

        records.append({
            "place_type":    "dark_kitchen",
            "place_name":    f"DK: {rep_name}",
            "place_address": address,
            "commune":       commune,
            "region":        "",
            "country":       country.lower(),
            "latitude":      _safe_float(r.get("cluster_latitude")),
            "longitude":     _safe_float(r.get("cluster_longitude")),
            "osm_id":        None,   # no tiene OSM id
            "source":        "internal_db",
            "raw_tags":      f'{{"total_clusters":{r.get("total_clusters",0)},'
                             f'"total_stores":{r.get("total_stores",0)},'
                             f'"sample_names":{str(names[:5])}}}',
        })

    ins, upd, skip = places_upsert(records)
    return {"inserted": ins, "updated": upd, "skipped": skip,
            "candidates_found": len(rows)}

def _safe_float(v):
    try: return float(v)
    except: return None

def _guess_commune(address, country):
    """
    Heurística para extraer comuna/ciudad de una dirección libre.
    Busca patrones comunes: 'Col. X', 'Alcaldía X', ', Ciudad'.
    """
    if not address: return ""
    # Último segmento tras coma suele ser ciudad en LATAM
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) >= 2:
        # Descartar código postal (sólo dígitos)
        for p in reversed(parts):
            if not re.match(r"^\d+$", p) and len(p) > 2:
                return p
    return ""

# ── Exportar a Excel ──────────────────────────────────────────────────────────

def export_places_excel(country=None, place_type=None, query=None):
    """
    Genera un Excel con los places filtrados.
    Retorna la ruta del archivo generado.
    """
    import pandas as pd
    import tempfile, os

    rows = places_list(country=country, place_type=place_type,
                       query=query, limit=10000)
    if not rows:
        return None, "Sin resultados con esos filtros"

    df = pd.DataFrame(rows)
    # Renombrar columnas para el Excel
    rename = {
        "id":            "ID",
        "place_type":    "Tipo",
        "place_name":    "Nombre",
        "place_address": "Dirección",
        "commune":       "Comuna",
        "region":        "Región",
        "country":       "País",
        "latitude":      "Latitud",
        "longitude":     "Longitud",
        "osm_id":        "OSM_ID",
        "source":        "Fuente",
        "scraped_at":    "Fecha_scraping",
    }
    df = df.rename(columns=rename)
    # Formatear timestamp
    if "Fecha_scraping" in df.columns:
        df["Fecha_scraping"] = pd.to_datetime(df["Fecha_scraping"], unit="s",
                                              errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

    # Drop osm_id y raw_tags en la vista Excel (son técnicos)
    for col in ["OSM_ID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False,
                                     prefix="places_export_")
    with pd.ExcelWriter(tmp.name, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Places")
        ws = writer.sheets["Places"]
        # Autofit columnas
        for col_cells in ws.columns:
            max_len = max((len(str(c.value or "")) for c in col_cells), default=8)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 60)

    return tmp.name, None

# Inicializar tabla al importar
init_places_db()
