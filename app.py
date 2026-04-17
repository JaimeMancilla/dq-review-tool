"""
Cluster Reviewer v4
- Conexión directa a PostgreSQL (credenciales en .env)
- Memoria interna SQLite (correcciones pasadas)
- Subgrupos reagrupables
- Feedback / aprendizaje activo

Correr: python app.py
Abrir:  http://localhost:5000
"""

import re, unicodedata, json, time, sqlite3, os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd

# Cargar .env si existe
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.secret_key = "cluster_reviewer_2024"

# Sesiones en filesystem (evita límite de 4KB de cookies)
try:
    from flask_session import Session
    app.config["SESSION_TYPE"]           = "filesystem"
    app.config["SESSION_FILE_DIR"]       = "flask_sessions"
    app.config["SESSION_PERMANENT"]      = False
    app.config["SESSION_USE_SIGNER"]     = True
    Path("flask_sessions").mkdir(exist_ok=True)
    Session(app)
except ImportError:
    pass  # fallback a cookie session

UPLOAD_FOLDER = Path("uploads"); UPLOAD_FOLDER.mkdir(exist_ok=True)
FEEDBACK_FILE = Path("feedback.json")
DB_FILE       = Path("memory.db")
MODEL_NAME = r"C:\models\paraphrase-multilingual"

# ── PostgreSQL ────────────────────────────────────────────────────────────────

def get_pg_config():
    return {
        "host":     os.getenv("PG_HOST",     ""),
        "database": os.getenv("PG_DATABASE", ""),
        "user":     os.getenv("PG_USER",     ""),
        "password": os.getenv("PG_PASSWORD", ""),
        "port":     int(os.getenv("PG_PORT", "5432")),
    }

def pg_is_configured():
    cfg = get_pg_config()
    return bool(cfg["host"] and cfg["database"] and cfg["user"])

def pg_query(sql, timeout_ms=15000):
    """Ejecuta una query y retorna filas como lista de dicts."""
    import psycopg2
    import psycopg2.extras
    cfg = get_pg_config()
    conn = psycopg2.connect(**cfg, connect_timeout=10,
                             options=f"-c statement_timeout={timeout_ms}")
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
        return rows, None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

# ── SQLite memoria interna ────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            item_index    TEXT NOT NULL,
            cluster_id    TEXT,
            app_name      TEXT,
            app_address   TEXT,
            anchor_name   TEXT,
            correction    TEXT,
            is_new        INTEGER DEFAULT 0,
            reviewed_at   REAL,
            file_name     TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_item ON corrections(item_index)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_appname ON corrections(app_name)")
    # Tabla para restauración completa de sesión por archivo
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_state (
            file_name    TEXT PRIMARY KEY,
            state_json   TEXT NOT NULL,
            saved_at     REAL
        )
    """)
    # Tabla de correcciones externas (stores no en el archivo pero identificados de paso)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS external_corrections (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            store_id         TEXT NOT NULL,
            item_index       TEXT NOT NULL,
            cluster_index    TEXT,
            cluster_name     TEXT,
            cluster_address  TEXT,
            app_name         TEXT,
            app_address      TEXT,
            scraper_source   TEXT,
            correction       TEXT NOT NULL,
            added_at         REAL,
            file_name        TEXT
        )
    """)
    # Migración: agregar scraper_source si no existe
    try:
        conn.execute("ALTER TABLE external_corrections ADD COLUMN scraper_source TEXT")
    except: pass
    # Tabla de progreso: clusters revisados
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviewed_clusters (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_index   TEXT NOT NULL,
            cluster_name    TEXT,
            had_errors      INTEGER DEFAULT 0,
            corrections_count INTEGER DEFAULT 0,
            reviewed_at     REAL,
            file_name       TEXT,
            review_type     TEXT DEFAULT 'stores_restaurant'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ci ON reviewed_clusters(cluster_index)")
    # Migración: agregar columna review_type si no existe
    try:
        conn.execute("ALTER TABLE reviewed_clusters ADD COLUMN review_type TEXT DEFAULT 'stores_restaurant'")
    except: pass
    # Pares etiquetados de stores para futuro fine-tuning
    conn.execute("""
        CREATE TABLE IF NOT EXISTS store_pairs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            app_name_a      TEXT,
            app_address_a   TEXT,
            cluster_index_a TEXT,
            app_name_b      TEXT,
            app_address_b   TEXT,
            cluster_index_b TEXT,
            label           INTEGER,  -- 1=mismo local, 0=distinto
            source          TEXT,     -- "correction" | "manual"
            added_at        REAL,
            file_name       TEXT
        )
    """)
    # Pares etiquetados de dishes para futuro fine-tuning
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dish_pairs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name_a      TEXT, desc_a TEXT,
            name_b      TEXT, desc_b TEXT,
            label       INTEGER,  -- 1=mismo plato, 0=distinto, 2=fusionar
            source      TEXT,     -- "fusion" | "revision"
            added_at    REAL,
            file_name   TEXT
        )
    """)
    # Tabla de archivos revisados (resumen por archivo)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviewed_files (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name    TEXT NOT NULL,
            review_type  TEXT NOT NULL,
            total        INTEGER DEFAULT 0,
            ok           INTEGER DEFAULT 0,
            bad          INTEGER DEFAULT 0,
            incomplete   INTEGER DEFAULT 0,
            reviewed_at  REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def mem_save(records):
    """Guarda correcciones en la memoria interna."""
    conn = sqlite3.connect(DB_FILE)
    for r in records:
        conn.execute("""
            INSERT INTO corrections
              (item_index, cluster_id, app_name, app_address, anchor_name,
               correction, is_new, reviewed_at, file_name)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            r.get("item_index",""), r.get("cluster_id",""),
            r.get("app_name",""),   r.get("app_address",""),
            r.get("anchor_name",""),r.get("correction",""),
            1 if r.get("is_new") else 0,
            time.time(), r.get("file_name","")
        ))
    conn.commit(); conn.close()

def mem_lookup_item(item_index):
    """Busca un item_index exacto en memoria."""
    conn = sqlite3.connect(DB_FILE)
    row = conn.execute(
        "SELECT * FROM corrections WHERE item_index=? ORDER BY reviewed_at DESC LIMIT 1",
        (item_index,)
    ).fetchone()
    conn.close()
    if row:
        cols = ["id","item_index","cluster_id","app_name","app_address",
                "anchor_name","correction","is_new","reviewed_at","file_name"]
        return dict(zip(cols, row))
    return None

def mem_lookup_similar(app_name, anchor_name, threshold=0.75):
    """Busca correcciones previas con nombre similar."""
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        "SELECT * FROM corrections WHERE correction != '' ORDER BY reviewed_at DESC LIMIT 500"
    ).fetchall()
    conn.close()
    cols = ["id","item_index","cluster_id","app_name","app_address",
            "anchor_name","correction","is_new","reviewed_at","file_name"]
    best, best_sim = None, 0
    for row in rows:
        r = dict(zip(cols, row))
        sim_name   = jaccard_sim(r["app_name"],   app_name)
        sim_anchor = jaccard_sim(r["anchor_name"], anchor_name)
        combined   = sim_name * 0.7 + sim_anchor * 0.3
        if combined > best_sim and combined >= threshold:
            best_sim = combined; best = r
    return best, best_sim

def save_session_state(filename, clusters, subgroups=None):
    """Guarda el estado completo: revisiones, correcciones y subgrupos."""
    state = []
    for cl in clusters:
        state.append({
            "cluster_id":  cl["cluster_id"],
            "corrections": cl.get("corrections", {}),
            "subgroups":   subgroups.get(cl["cluster_id"], []) if subgroups else [],
            "members":     [
                {"item_index": m["item_index"], "revision": m["revision"]}
                for m in cl["members"] if not m["is_anchor"]
            ]
        })
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT OR REPLACE INTO session_state (file_name, state_json, saved_at)
        VALUES (?, ?, ?)
    """, (filename, json.dumps(state, ensure_ascii=False), time.time()))
    conn.commit(); conn.close()

def load_session_state(filename):
    """Restaura el estado guardado para un archivo."""
    conn = sqlite3.connect(DB_FILE)
    row = conn.execute(
        "SELECT state_json, saved_at FROM session_state WHERE file_name=?",
        (filename,)
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row[0]), row[1]
    return None, None

def apply_session_state(clusters, state):
    """Aplica el estado guardado sobre los clusters recién cargados."""
    state_map = {s["cluster_id"]: s for s in state}
    subgroups_restored = {}
    for cl in clusters:
        saved = state_map.get(cl["cluster_id"])
        if not saved: continue
        rev_map = {m["item_index"]: m["revision"] for m in saved["members"]}
        for m in cl["members"]:
            if not m["is_anchor"] and m["item_index"] in rev_map:
                m["revision"] = rev_map[m["item_index"]]
        cl["corrections"] = saved.get("corrections", {})
        cl["ok_count"]  = sum(1 for m in cl["members"] if m["revision"]==1)
        cl["bad_count"] = sum(1 for m in cl["members"] if m["revision"]==0)
        if saved.get("subgroups"):
            subgroups_restored[cl["cluster_id"]] = saved["subgroups"]
    return clusters, subgroups_restored

def mem_stats():
    conn = sqlite3.connect(DB_FILE)
    total   = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
    unique  = conn.execute("SELECT COUNT(DISTINCT item_index) FROM corrections").fetchone()[0]
    recent  = conn.execute(
        "SELECT item_index, app_name, correction, reviewed_at FROM corrections ORDER BY reviewed_at DESC LIMIT 5"
    ).fetchall()
    conn.close()
    return {"total": total, "unique_items": unique, "recent": recent}

def mem_search(query, limit=50):
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("""
        SELECT DISTINCT item_index, app_name, app_address, correction, reviewed_at
        FROM corrections
        WHERE app_name LIKE ? OR item_index LIKE ? OR correction LIKE ?
        ORDER BY reviewed_at DESC LIMIT ?
    """, (f"%{query}%", f"%{query}%", f"%{query}%", limit)).fetchall()
    conn.close()
    cols = ["item_index","app_name","app_address","correction","reviewed_at"]
    return [dict(zip(cols, r)) for r in rows]

# ── modelo semántico ──────────────────────────────────────────────────────────

_model = None; _model_ready = False; _model_error = None

def get_model():
    global _model, _model_ready, _model_error
    if _model_ready or _model_error: return _model
    try:
        from sentence_transformers import SentenceTransformer
        import os
        # Forzar uso de caché sin contactar HuggingFace
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            _model = SentenceTransformer(MODEL_NAME)
        finally:
            # Restaurar para no afectar otras operaciones
            os.environ.pop("HF_HUB_OFFLINE", None)
        _model_ready = True
    except Exception as e:
        _model_error = str(e)
    return _model

def batch_semantic_sim(anchor_name, anchor_addr, members):
    m = get_model()
    if m is None:
        return [jaccard_sim(mem.get("app_name",""), anchor_name)*0.85 +
                jaccard_sim(mem.get("app_address",""), anchor_addr)*0.15
                for mem in members]
    from sentence_transformers import util
    ea_n = m.encode(anchor_name, convert_to_tensor=True, show_progress_bar=False)
    ea_a = m.encode(anchor_addr, convert_to_tensor=True, show_progress_bar=False)
    names = [mem.get("app_name","") for mem in members]
    addrs = [mem.get("app_address","") for mem in members]
    en = m.encode(names, convert_to_tensor=True, show_progress_bar=False)
    ea = m.encode(addrs, convert_to_tensor=True, show_progress_bar=False)
    return [round(float(util.cos_sim(ea_n,en[i]))*0.85 + float(util.cos_sim(ea_a,ea[i]))*0.15, 3)
            for i in range(len(members))]

# ── helpers ───────────────────────────────────────────────────────────────────

def norm(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()

def jaccard_sim(a, b):
    wa = set(w for w in norm(a).split() if len(w) > 2)
    wb = set(w for w in norm(b).split() if len(w) > 2)
    if not wa or not wb: return 0.0
    return len(wa & wb) / len(wa | wb)

def load_file(path):
    path = str(path)
    if path.endswith(".csv"):
        for enc in ["utf-8","latin-1"]:
            try: df = pd.read_csv(path, dtype=str, encoding=enc); break
            except: continue
    else:
        df = pd.read_excel(path, dtype=str)
    df = df.fillna(""); df.columns = [c.strip() for c in df.columns]
    return df

def effective_cluster(row):
    nc = row.get("new_cluster","")
    if isinstance(nc,str) and nc.strip(): return nc.strip()
    ci = row.get("cluster_index","")
    return ci.strip() if isinstance(ci,str) else ""

def detect_country(df):
    if "country" in df.columns:
        vals = df["country"].dropna().str.strip().str.lower()
        vals = vals[vals != ""]
        if len(vals): return vals.mode()[0]
    return "mx"

def get_threshold():
    fb = load_feedback()
    return round(min(1.0, max(0.5, 0.95 + fb.get("threshold_adj", 0.0))), 4)

# ── feedback ──────────────────────────────────────────────────────────────────

def load_feedback():
    if FEEDBACK_FILE.exists():
        try: return json.loads(FEEDBACK_FILE.read_text())
        except: pass
    return {"pairs":[], "threshold_adj":0.0, "stats":{"total":0,"overrides_to_1":0,"overrides_to_0":0}}

def save_feedback(fb): FEEDBACK_FILE.write_text(json.dumps(fb, ensure_ascii=False, indent=2))

def record_feedback(anchor_name, anchor_addr, member_name, member_addr, model_score, model_rev, human_rev):
    if model_rev == human_rev: return
    fb = load_feedback()
    fb["pairs"].append({"anchor_name":anchor_name,"anchor_addr":anchor_addr,
                         "member_name":member_name,"member_addr":member_addr,
                         "model_score":model_score,"model_rev":model_rev,
                         "human_rev":human_rev,"ts":time.time()})
    fb["stats"]["total"] += 1
    if human_rev==1: fb["stats"]["overrides_to_1"] += 1
    else:            fb["stats"]["overrides_to_0"] += 1
    recent = fb["pairs"][-30:]
    if len(recent) >= 5:
        fp = sum(1 for p in recent if p["model_rev"]==1 and p["human_rev"]==0)
        fn = sum(1 for p in recent if p["model_rev"]==0 and p["human_rev"]==1)
        adj = max(-0.10, min(0.10, (fp-fn)/len(recent)*0.10))
        fb["threshold_adj"] = round(adj, 4)
    save_feedback(fb)

# ── subgrupos ─────────────────────────────────────────────────────────────────

GEO_STOP = {
    # Colonias / barrios MX frecuentes
    "reforma","serdan","serdán","canek","manuel","moreno","loza","morelos",
    "monica","mónica","hidalgo","centro","pedregal","polanco","roma","condesa",
    "satelite","satélite","narvarte","coyoacan","coyoacán","xochimilco",
    "tepito","tlalpan","iztapalapa","ecatepec","naucalpan","tlalnepantla",
    "texcoco","chalco","ixtapaluca","nezahualcoyotl","nezahualcóyotl",
    "insurgentes","chapultepec","juarez","juárez","guerrero","doctores",
    "lindavista","vallejo","azcapotzalco","magdalena","contreras",
    "tlahuac","tláhuac","milpa","alta","alvaro","obregon","obregón",
    "benito","cuauhtemoc","cuauhtémoc","miguel","venustiano","carranza",
    "iztacalco","gustavo","madero","cuajimalpa","cuautitlan","cuautitlán",
    # Palabras geográficas genéricas
    "san","santa","santo","nuevo","nueva","gran","grande","real","real",
    "jardines","residencial","fracc","fraccionamiento","unidad","barrio",
    "colonia","col","pueblo","ciudad","municipio","delegacion","delegación",
    "local","plaza","paseo","calle","avenida","blvd","boulevard","calz","calzada",
    "sur","norte","oriente","poniente","este","oeste","inner","outer",
    "zona","sector","modulo","módulo","piso","nivel","torre","edificio",
    # Ciudades/estados MX
    "puebla","monterrey","guadalajara","tijuana","juarez","merida","mérida",
    "leon","león","toluca","queretaro","querétaro","chihuahua","hermosillo",
    "mexicali","aguascalientes","morelia","veracruz","oaxaca","tuxtla",
    "gutierrez","gutiérrez","villahermosa","tampico","culiacan","culiacán",
    "durango","zacatecas","saltillo","tepic","colima","campeche","chetumal",
    "pachuca","tlaxcala","cuernavaca","chilpancingo","xalapa","jalapa",
    "cdmx","cdmex","edomex","mx","mexico","méxico",
    # Colombia
    "bogota","bogotá","medellin","medellín","cali","barranquilla","cartagena",
    "cucuta","cúcuta","bucaramanga","pereira","manizales","armenia","ibague",
    "ibagué","neiva","villavicencio","pasto","monteria","montería","sincelejo",
    "valledupar","santa","marta","riohacha","quibdo","quibdó","popayan",
    "popayán","tunja","manizales","florencia","mocoa","leticia","inirida",
    "inírida","mitu","mitú","yopal","arauca","puerto","carreño","san","andres",
    # Brasil
    "sao","são","paulo","rio","janeiro","belo","horizonte","brasilia","brasília",
    "salvador","fortaleza","curitiba","manaus","recife","porto","alegre",
    "belem","belém","goiania","goiânia","florianopolis","florianópolis",
    "natal","maceio","maceió","joao","joão","pessoa","teresina","campo",
    "grande","aracaju","macapa","macapá","porto","velho","palmas","boa","vista",
    "rio","branco","vitoria","vitória","cuiaba","cuiabá",
}

def brand_words(name):
    """
    Extrae palabras de marca ignorando sufijos geográficos.
    Para cadenas de comida rápida, las sub-entidades (Postres, Desayunos, Pollos,
    Chicken, Turbo, Vegetal) se tratan como parte distintiva de la marca.
    """
    func = {"la","el","los","las","de","del","en","al","por","con","que","y","e","o"}
    generic = {"burger","pizza","tacos","taco","tortas","torta","wings",
               "alitas","sushi","coffee","cafe","grill","cocina","comida"}

    # Sub-entidades de comida rápida: son palabras de marca, NO ignorar
    FAST_FOOD_SUBS = {
        "postres","desayunos","pollos","pollo","chicken",
        "vegetal","mccafe","cafe","express","drive","thru","junior"
    }

    # Cadenas de comida rápida conocidas: su nombre es 1 sola palabra clave
    FAST_FOOD_CHAINS = {
        "mcdonald","mcdonalds","burgerkng","burgerking","kfc","subway",
        "dominos","pizzahut","wendys","popeyes","chickfila","tacobell"
    }

    words_raw = norm(name).split()

    # Detectar si el nombre contiene una cadena + sub-entidad
    chain_word = next((w for w in words_raw if any(c in w for c in FAST_FOOD_CHAINS)), None)
    sub_word   = next((w for w in words_raw if w in FAST_FOOD_SUBS), None)

    if chain_word and sub_word:
        # Retornar cadena + sub-entidad como las dos palabras clave
        return [chain_word, sub_word]

    # Flujo normal: palabras sin stopwords geo, longitud > 1
    words = [w for w in words_raw if len(w) > 1 and w not in GEO_STOP]

    # Eliminar artículos iniciales
    while words and words[0] in func:
        words = words[1:]

    result = words[:3]

    # Si solo queda 1 palabra genérica, incluir también palabra anterior
    if len(result) == 1 and result[0] in generic:
        idx = next((i for i,w in enumerate(words_raw) if w == result[0]), -1)
        if idx > 0:
            prev = [w for w in words_raw[max(0,idx-2):idx] if w not in GEO_STOP]
            result = prev + result

    return result[:3]

def build_subgroups(bad_members):
    if not bad_members: return []

    FAST_FOOD_SUBS = {
        "postres","desayunos","pollos","pollo","chicken",
        "vegetal","mccafe","cafe","express","drive","thru","junior"
    }

    def has_sub_entity(name):
        return any(w in FAST_FOOD_SUBS for w in norm(name).split())

    used = [False]*len(bad_members)
    subgroups = []
    for i, a in enumerate(bad_members):
        if used[i]: continue
        wa = set(brand_words(a.get("app_name","")))
        ak = set(extract_addr_keys(a.get("app_address","")))
        grp = [a]; used[i] = True
        # Marca genérica: 1 sola palabra → separar por dirección siempre
        generic_brand = len(wa) <= 1
        for j, b in enumerate(bad_members):
            if used[j] or i==j: continue
            wb = set(brand_words(b.get("app_name","")))
            if not wa or not wb: continue
            union = wa|wb; inter = wa&wb
            if not union or len(inter)/len(union) < 0.60: continue
            # Separar por dirección si: marca genérica O tiene sub-entidad
            if generic_brand or has_sub_entity(a.get("app_name","")) or has_sub_entity(b.get("app_name","")):
                bk = set(extract_addr_keys(b.get("app_address","")))
                if ak and bk and not ak & bk:
                    continue  # direcciones distintas → subgrupos separados
            grp.append(b); used[j] = True
        grp.sort(key=lambda m: m.get("item_index",""))
        subgroups.append({
            "rep_name":   grp[0].get("app_name",""),
            "rep_addr":   grp[0].get("app_address",""),
            "members":    [m.get("item_index","") for m in grp],
            "correction": "", "is_new": False,
        })
    return subgroups

def extract_addr_keys(addr):
    """
    Extrae claves de dirección: número de calle + primera palabra larga no geográfica.
    Ej: 'Calle 4 Sur 302 Local C, La Libertad' → ['302', 'libertad']
    """
    addr_geo_stop = {
        "calle","avenida","av","blvd","boulevard","carretera","carr",
        "sur","norte","oriente","poniente","ote","pte","col","colonia",
        "fracc","fraccionamiento","local","plaza","paseo","zona","barrio",
        "mexico","latam","cp","sin","nombre","entre",
    }
    words = re.split(r"[,\s]+", norm(addr))
    keys = []
    for w in words:
        wc = re.sub(r"[^\w]", "", w)
        if wc.isdigit() and 2 <= len(wc) <= 5:
            keys.append(wc)
        elif len(wc) > 4 and wc.isalpha() and wc not in addr_geo_stop:
            keys.append(wc)
        if len(keys) >= 2:
            break
    return keys

ACCENT_MAP = [
    ("a", "á"), ("e", "é"), ("i", "í"), ("o", "ó"), ("u", "ú"), ("n", "ñ"),
]

def accent_variants(word):
    """
    Genera variantes de una palabra con y sin tildes.
    'burreria' → ['burreria', 'burrería']
    'burrerıa'  → ['burrería', 'burreria']
    """
    variants = {word}
    for plain, accented in ACCENT_MAP:
        if plain in word:
            variants.add(word.replace(plain, accented, 1))
        if accented in word:
            variants.add(word.replace(accented, plain, 1))
    return list(variants)

def word_ilike(field, word):
    """Genera condición ilike robusta a tildes y Ñ.
    Para palabras con Ñ, corta antes de la Ñ para matchear tanto 'castano' como 'castaño'.
    Para otras palabras con tilde, genera variantes.
    """
    w_norm = norm(word)  # versión ASCII sin tildes ni Ñ
    if not w_norm:
        return f"{field} ilike '%{word}%'"

    # Si la palabra normalizada difiere del original por Ñ,
    # usar el prefijo antes de la Ñ (más robusto que variantes)
    if 'n' in w_norm and 'ñ' in word.lower():
        # Cortar antes de la Ñ → '%casta%' matchea 'castano' y 'castaño'
        idx = word.lower().index('ñ')
        prefix = norm(word[:idx])
        if len(prefix) >= 3:
            return f"{field} ilike '%{prefix}%'"

    # Para otras palabras: variantes de tilde
    variants = accent_variants(w_norm)
    if len(variants) == 1:
        return f"{field} ilike '%{w_norm}%'"
    conditions = " or ".join(f"{field} ilike '%{v}%'" for v in sorted(variants))
    return f"({conditions})"

def addr_ilike_safe(field, word):
    """
    Busca una palabra robusta a tildes usando OR de segmentos de 4 chars
    que empiezan en consonante. Para cualquier posición de tilde, al menos
    un segmento matcheará.
    Ej: 'amunategui' → ('%muna%' OR '%nate%' OR '%tegu%')
        matchea 'Amunátegui', 'amunategui', 'amunAtegui', etc.
    """
    w = norm(word)
    if not w: return f"{field} ilike '%{word}%'"
    if w.isdigit() or len(w) <= 4:
        return f"{field} ilike '%{w}%'"

    VOWELS = set('aeiou')
    segments = []
    for i in range(len(w) - 3):
        if w[i] not in VOWELS:
            seg = w[i:i+4]
            if seg not in segments:
                segments.append(seg)

    if not segments:
        # Sin consonantes (muy raro) — usar la palabra completa normalizada
        return f"{field} ilike '%{w}%'"

    if len(segments) == 1:
        return f"{field} ilike '%{segments[0]}%'"

    conds = " or ".join(f"{field} ilike '%{s}%'" for s in segments)
    return f"({conds})"

def get_pg_table():
    """Retorna la tabla correcta según el tipo de revisión de la sesión."""
    rt = session.get("review_type", "stores_restaurant")
    if rt == "stores_retail":
        return "sales_opportunity.dim_maestra_retail"
    return "sales_opportunity.dim_maestra"

def generate_unified_sql(bad_members, country="mx"):
    """
    Genera SQL usando:
    - brand_words(name): palabras de marca sin sufijos geográficos
    - extract_addr_keys(addr): número de calle + palabra distintiva de dirección
    - accent_variants: maneja palabras con/sin tilde automáticamente
    Para cadenas con sub-entidad (McDonald's Postres), busca por cadena Y sub-entidad
    con OR entre ellas para mayor flexibilidad.
    """
    if not bad_members: return ""
    seen, blocks = set(), []

    FAST_FOOD_SUBS = {
        "postres","desayunos","pollos","pollo","chicken",
        "vegetal","mccafe","cafe","express","drive","thru","junior"
    }

    for m in bad_members:
        name = m.get("app_name","")
        addr = m.get("app_address","")

        bwords = brand_words(name)
        if len(bwords) < 1:
            bwords = [w for w in norm(name).split() if len(w) > 3][:2]

        sql_bwords = [w for w in bwords if len(w) > 2] or bwords

        # Detectar si hay sub-entidad entre las brand_words
        sub = next((w for w in sql_bwords if w in FAST_FOOD_SUBS), None)
        chain_words = [w for w in sql_bwords if w not in FAST_FOOD_SUBS]

        if sub and chain_words:
            akeys = extract_addr_keys(addr)
            key = "_".join(chain_words) + "_" + sub + "_" + "_".join(akeys)
            if key in seen: continue
            seen.add(key)
            # Usar nombre original para detectar Ñ correctamente
            chain_cond = " and ".join(word_ilike("app_name", w) for w in chain_words)
            sub_cond   = word_ilike("app_name", sub)
            name_cond  = f"({chain_cond}\n     and {sub_cond})"
        else:
            key = "_".join(bwords)
            if key in seen or not bwords: continue
            seen.add(key)
            # Pasar nombre original para detectar Ñ; si hay una sola bword usar el nombre completo
            if len(sql_bwords) == 1:
                name_cond = word_ilike("app_name", name)
            else:
                name_cond = " and ".join(word_ilike("app_name", w) for w in sql_bwords)

        # Condición de dirección
        akeys = extract_addr_keys(addr)
        if akeys:
            addr_cond = " and ".join(addr_ilike_safe("app_address", w) for w in akeys)
            blocks.append(f"    ({name_cond}\n     and {addr_cond})")
        else:
            # Sin dirección útil: agregar solo condición de nombre (sin deduplicar más)
            name_key = "_".join(sql_bwords)
            if name_key not in seen:
                seen.add(name_key)
                blocks.append(f"    ({name_cond})")

    if not blocks: return ""

    # Limitar a 8 bloques máximo para evitar queries demasiado largas
    if len(blocks) > 8:
        blocks = blocks[:8]

    comment = "; ".join(m.get("app_name","")[:35] for m in bad_members[:5])
    if len(bad_members) > 5: comment += "..."

    return (
        f"-- {len(bad_members)} store(s): {comment}\n"
        f"select\n  cluster_index, store_id, cluster_name, main_chain,\n"
        f"  app_name, app_address, app_longitude, app_latitude\n"
        f"from {get_pg_table()}\n"
        f"where country = '{country}'\n"
        f"  and (\n" + "\n    or\n".join(blocks) + "\n  )\n"
        f"order by cluster_index;"
    )


# ── build clusters ────────────────────────────────────────────────────────────

def build_clusters(df, filename=""):
    grouped = {}
    for _, row in df.iterrows():
        r = {k:("" if pd.isna(v) else str(v)) for k,v in row.items()}
        cf = effective_cluster(r)
        if cf: grouped.setdefault(cf,[]).append(r)

    clusters = []; threshold = get_threshold()
    fb = load_feedback()

    for cf, members in grouped.items():
        anchor = next((m for m in members if m.get("item_index","").strip()==cf), None)
        if anchor is None:
            anchor = sorted(members, key=lambda m: m.get("item_index",""))[0]
        rest = sorted([m for m in members if m is not anchor],
                      key=lambda m: norm(m.get("app_name","")))
        anchor_name = anchor.get("app_name","")
        anchor_addr = anchor.get("app_address","")
        scores = batch_semantic_sim(anchor_name, anchor_addr, rest)

        members_out = [{
            "item_index": anchor.get("item_index","").strip(),
            "new_cluster": anchor.get("new_cluster",cf),
            "cluster_index": anchor.get("cluster_index",""),
            "app_name": anchor_name, "app_address": anchor_addr,
            "store_id": anchor.get("store_id",""),
            "scraper_source": anchor.get("scraper_source",""),
            "is_anchor": True, "revision": 1, "score": None,
            "mem_hit": None,
            **{k: anchor.get(k,"") for k in ["cluster_name","main_chain","country",
               "cluster_latitude","cluster_longitude","app_latitude","app_longitude",
               "cluster_ciudad","cluster_estado","app_phone_1","app_phone_2","app_phone_3"]}
        }]

        for i, m in enumerate(rest):
            ii    = m.get("item_index","").strip()
            score = scores[i]
            rev   = 1 if score >= threshold else 0

            # Consultar memoria interna
            mem_hit = mem_lookup_item(ii)
            if not mem_hit:
                mem_hit, _ = mem_lookup_similar(m.get("app_name",""), anchor_name)

            members_out.append({
                "item_index": ii,
                "new_cluster": m.get("new_cluster",cf),
                "cluster_index": m.get("cluster_index",""),
                "app_name": m.get("app_name",""),
                "app_address": m.get("app_address",""),
                "store_id": m.get("store_id",""),
                "scraper_source": m.get("scraper_source",""),
                "is_anchor": False, "revision": rev, "score": score,
                "mem_hit": mem_hit,
                **{k: m.get(k,"") for k in ["cluster_name","main_chain","country",
                   "cluster_latitude","cluster_longitude","app_latitude","app_longitude",
                   "cluster_ciudad","cluster_estado","app_phone_1","app_phone_2","app_phone_3"]}
            })

        clusters.append({
            "cluster_id": cf, "anchor_name": anchor_name, "anchor_addr": anchor_addr,
            "count": len(members_out),
            "ok_count": sum(1 for m in members_out if m["revision"]==1),
            "bad_count": sum(1 for m in members_out if m["revision"]==0),
            "members": members_out, "sql": "", "bd_results": [],
            "corrections": {}, "threshold_used": threshold,
        })

    clusters.sort(key=lambda c: c["cluster_id"])
    return clusters, {"threshold": threshold, "fb_stats": fb["stats"],
                      "model_ready": _model_ready, "pg_ready": pg_is_configured()}


def ext_corr_add(record):
    """Agrega una corrección externa (store no en el archivo)."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO external_corrections
          (store_id, item_index, cluster_index, cluster_name, cluster_address,
           app_name, app_address, scraper_source, correction, added_at, file_name)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record.get("store_id",""),   record.get("item_index",""),
        record.get("cluster_index",""), record.get("cluster_name",""),
        record.get("cluster_address",""), record.get("app_name",""),
        record.get("app_address",""), record.get("scraper_source",""),
        record.get("correction",""),
        time.time(), record.get("file_name","")
    ))
    conn.commit(); conn.close()

def ext_corr_list():
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("""
        SELECT id, store_id, item_index, cluster_index, cluster_name,
               cluster_address, app_name, app_address, scraper_source,
               correction, added_at, file_name
        FROM external_corrections ORDER BY added_at DESC
    """).fetchall()
    conn.close()
    cols = ["id","store_id","item_index","cluster_index","cluster_name",
            "cluster_address","app_name","app_address","scraper_source","correction","added_at","file_name"]
    return [dict(zip(cols,r)) for r in rows]

def ext_corr_delete(rec_id):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM external_corrections WHERE id=?", (rec_id,))
    conn.commit(); conn.close()

def save_reviewed_clusters_dishes(groups, filename):
    """Guarda grupos de dishes en reviewed_clusters."""
    conn = sqlite3.connect(DB_FILE)
    for g in groups:
        conn.execute("""
            INSERT INTO reviewed_clusters
              (cluster_index, cluster_name, had_errors, corrections_count, reviewed_at, file_name)
            VALUES (?,?,?,?,?,?)
        """, (g["group_id"], "", 1 if g["revision"] in (0,2) else 0,
              0, time.time(), filename))
    conn.commit(); conn.close()

def save_reviewed_file(file_name, review_type, total, ok, bad, incomplete=0):
    """Guarda un resumen del archivo revisado al descargar."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO reviewed_files (file_name, review_type, total, ok, bad, incomplete, reviewed_at)
        VALUES (?,?,?,?,?,?,?)
    """, (file_name, review_type, total, ok, bad, incomplete, time.time()))
    conn.commit(); conn.close()

def get_reviewed_files(review_type=None):
    """Retorna historial de archivos revisados."""
    conn = sqlite3.connect(DB_FILE)
    if review_type:
        rows = conn.execute("""
            SELECT file_name, review_type, total, ok, bad, incomplete, reviewed_at
            FROM reviewed_files WHERE review_type=?
            ORDER BY reviewed_at DESC
        """, (review_type,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT file_name, review_type, total, ok, bad, incomplete, reviewed_at
            FROM reviewed_files ORDER BY reviewed_at DESC
        """).fetchall()
    conn.close()
    cols = ["file_name","review_type","total","ok","bad","incomplete","reviewed_at"]
    return [dict(zip(cols,r)) for r in rows]

def save_reviewed_clusters(clusters, filename, review_type=None):
    """Guarda qué clusters pasaron por revisión al exportar."""
    rt = review_type or session.get("review_type","stores_restaurant")
    conn = sqlite3.connect(DB_FILE)
    for cl in clusters:
        had_errors = cl.get("bad_count",0) > 0
        corr_count = len(cl.get("corrections",{}))
        conn.execute("""
            INSERT INTO reviewed_clusters
              (cluster_index, cluster_name, had_errors, corrections_count, reviewed_at, file_name, review_type)
            VALUES (?,?,?,?,?,?,?)
        """, (
            cl["cluster_id"],
            cl.get("anchor_name",""),
            1 if had_errors else 0,
            corr_count,
            time.time(),
            filename,
            rt
        ))
    conn.commit(); conn.close()

def progress_stats(table="sales_opportunity.dim_maestra", country="mx", review_type="stores_restaurant"):
    """Cruza la memoria con la BD para calcular progreso."""
    if not pg_is_configured():
        return None, "BD no configurada"

    # Filtro de country
    country_filter = f"WHERE country = '{country}'" if country != "all" else ""
    where_and = f"AND country = '{country}'" if country != "all" else ""

    # Clusters y tiendas totales en BD
    sql_total = f"""
        SELECT
            COUNT(DISTINCT cluster_index) as total_clusters,
            COUNT(*) as total_stores
        FROM {table}
        {country_filter}
    """
    rows, err = pg_query(sql_total, timeout_ms=30000)
    if err:
        if "does not exist" in str(err) or "doesn't exist" in str(err):
            return None, f"La tabla '{table}' no existe en la BD"
        return None, err
    total_clusters = rows[0]["total_clusters"] if rows else 0
    total_stores   = rows[0]["total_stores"]   if rows else 0

    # Desde memoria interna — filtrar por review_type
    conn = sqlite3.connect(DB_FILE)
    rt_filter = f"AND review_type = '{review_type}'"
    reviewed_clusters = conn.execute(f"""
        SELECT COUNT(DISTINCT cluster_index) FROM reviewed_clusters
        WHERE 1=1 {rt_filter}
    """).fetchone()[0]
    corrected_clusters = conn.execute(f"""
        SELECT COUNT(DISTINCT cluster_index) FROM reviewed_clusters
        WHERE corrections_count > 0 {rt_filter}
    """).fetchone()[0]
    # Tiendas corregidas desde corrections (filtradas por review_type via file_name)
    migrated = conn.execute("""
        SELECT COUNT(*) FROM corrections WHERE is_new = 0
    """).fetchone()[0]
    new_cluster = conn.execute("""
        SELECT COUNT(*) FROM corrections WHERE is_new = 1
    """).fetchone()[0]
    recent = conn.execute(f"""
        SELECT cluster_index, cluster_name, had_errors, corrections_count, reviewed_at, file_name
        FROM reviewed_clusters WHERE 1=1 {rt_filter}
        ORDER BY reviewed_at DESC LIMIT 20
    """).fetchall()
    conn.close()

    pct = round(reviewed_clusters / int(total_clusters) * 100, 1) if total_clusters else 0
    return {
        "table":              table,
        "country":            country,
        "total_clusters":     int(total_clusters),
        "total_stores":       int(total_stores),
        "reviewed":           reviewed_clusters,
        "corrected":          corrected_clusters,
        "migrated_stores":    migrated,
        "new_cluster_stores": new_cluster,
        "pending":            max(0, int(total_clusters) - reviewed_clusters),
        "pct":                pct,
        "recent": [
            {"cluster_index":r[0],"cluster_name":r[1],"had_errors":bool(r[2]),
             "corrections_count":r[3],"reviewed_at":r[4],"file_name":r[5]}
            for r in recent
        ]
    }, None

# ── dishes (revisión de platos) ───────────────────────────────────────────────

def pairwise_sim_semantic(names, descs):
    """Similitud semántica promedio entre todos los pares. Sin fallback Jaccard."""
    texts = [f"{n} {d}".strip() for n, d in zip(names, descs)]
    if len(texts) < 2:
        return None  # grupo de 1 → incompleto
    m = get_model()
    if not m:
        return None  # sin modelo → no evaluar
    from sentence_transformers import util
    embs = m.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    sims = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sims.append(float(util.cos_sim(embs[i], embs[j])))
    return round(sum(sims)/len(sims), 3)

def detect_error_categories(members):
    """
    Detecta categorías de error automáticamente.
    Retorna lista de categorías: price, name, description, no_data
    """
    categories = []

    # Error de precio: diferencia > 20% entre min y max
    prices = []
    for m in members:
        try:
            p = float(str(m.get("price","")).replace(",",".").strip())
            if p > 0: prices.append(p)
        except: pass
    if len(prices) >= 2:
        mn, mx = min(prices), max(prices)
        if mx > 0 and (mx - mn) / mx > 0.20:
            categories.append("price")

    # Error de nombre: palabra clave distintiva en uno que no está en otros
    names = [norm(m.get("product_name","")) for m in members]
    all_words = [set(w for w in n.split() if len(w) > 3) for n in names]
    for i, wi in enumerate(all_words):
        for j, wj in enumerate(all_words):
            if i == j: continue
            exclusive = wi - wj  # palabras de i que no están en j
            if exclusive:
                categories.append("name")
                break
        if "name" in categories: break

    # Error de descripción: baja similitud entre descripciones (si existen)
    descs = [m.get("product_description","").strip() for m in members]
    has_descs = [d for d in descs if d]
    if len(has_descs) >= 2:
        m_model = get_model()
        if m_model:
            from sentence_transformers import util
            embs = m_model.encode(has_descs, convert_to_tensor=True, show_progress_bar=False)
            sims = []
            for i in range(len(has_descs)):
                for j in range(i+1, len(has_descs)):
                    sims.append(float(util.cos_sim(embs[i], embs[j])))
            avg = sum(sims)/len(sims) if sims else 1.0
            if avg < 0.70:
                categories.append("description")
        elif not has_descs:
            categories.append("no_data")
    elif not has_descs:
        categories.append("no_data")

    return list(set(categories))

def build_dish_groups(df, filename="", threshold=0.75):
    """Construye grupos de platos a partir del archivo de revisión."""
    df = df.fillna("")
    groups = {}
    for _, row in df.iterrows():
        r = {k: str(v) for k, v in row.items()}
        gid = r.get("grupo_pn9_max", "").strip()
        if not gid: continue
        groups.setdefault(gid, []).append(r)

    result = []
    for gid, members in groups.items():
        names = [m.get("product_name","") for m in members]
        descs = [m.get("product_description","") for m in members]
        score = pairwise_sim_semantic(names, descs)

        # revision: 2=incompleto (1 solo miembro o sin modelo), 1=correcto, 0=incorrecto
        if score is None:
            revision = 2  # incompleto — grupo de 1 o sin modelo
        elif score >= threshold:
            revision = 1
        else:
            revision = 0

        # Categorías de error (solo para incorrectos)
        error_cats = detect_error_categories(members) if revision in (0, 2) else []

        result.append({
            "group_id":    gid,
            "score":       score,
            "revision":    revision,
            "count":       len(members),
            "error_cats":  error_cats,
            "fusion_with": "",  # grupo_pn9_max con el que fusionar (si revision==2)
            "members": [{
                "product_id":          m.get("product_id",""),
                "scraper_source":      m.get("scraper_source",""),
                "product_name":        m.get("product_name",""),
                "product_description": m.get("product_description",""),
                "price":               m.get("price",""),
                "cluster_index":       m.get("cluster_index",""),
            } for m in members]
        })

    result.sort(key=lambda g: g["group_id"])
    ok   = sum(1 for g in result if g["revision"]==1)
    bad  = sum(1 for g in result if g["revision"]==0)
    inc  = sum(1 for g in result if g["revision"]==2)
    return result, {
        "total_groups": len(result), "ok": ok, "bad": bad, "incomplete": inc,
        "threshold": threshold, "model_ready": _model_ready
    }



def save_store_pair(app_name_a, app_address_a, cluster_index_a,
                    app_name_b, app_address_b, cluster_index_b,
                    label=0, source="correction", file_name=""):
    """Guarda un par etiquetado de stores para futuro fine-tuning del algoritmo de clusterización.
    label=1: mismo local físico, label=0: locales distintos.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO store_pairs
          (app_name_a, app_address_a, cluster_index_a,
           app_name_b, app_address_b, cluster_index_b,
           label, source, added_at, file_name)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (app_name_a, app_address_a, cluster_index_a,
          app_name_b, app_address_b, cluster_index_b,
          label, source, time.time(), file_name))
    conn.commit(); conn.close()

def save_dish_pair(name_a, desc_a, name_b, desc_b, label, source="revision", file_name=""):
    """Guarda un par etiquetado de dishes para futuro fine-tuning."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO dish_pairs (name_a, desc_a, name_b, desc_b, label, source, added_at, file_name)
        VALUES (?,?,?,?,?,?,?,?)
    """, (name_a, desc_a, name_b, desc_b, label, source, time.time(), file_name))
    conn.commit(); conn.close()

def compute_group_similarity(g_target, g_other):
    """
    Calcula similitud entre dos grupos de dishes.
    Combina nombre (70%) y descripción (30%).
    """
    # Texto representativo de cada grupo (primer miembro)
    name_a = g_target["members"][0].get("product_name","") if g_target["members"] else ""
    desc_a = g_target["members"][0].get("product_description","") if g_target["members"] else ""
    name_b = g_other["members"][0].get("product_name","") if g_other["members"] else ""
    desc_b = g_other["members"][0].get("product_description","") if g_other["members"] else ""

    m = get_model()
    if m:
        from sentence_transformers import util
        texts = [name_a, name_b, desc_a or name_a, desc_b or name_b]
        embs  = m.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        sim_name = float(util.cos_sim(embs[0], embs[1]))
        sim_desc = float(util.cos_sim(embs[2], embs[3]))
    else:
        sim_name = jaccard_sim(name_a, name_b)
        sim_desc = jaccard_sim(desc_a or name_a, desc_b or name_b)

    return round(sim_name * 0.7 + sim_desc * 0.3, 3)

# ── rutas ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/session/restore")
def session_restore():
    """Devuelve el estado de la sesión activa si existe."""
    if session.get("clusters"):
        clusters = session.get("clusters",[])
        return jsonify({
            "type":     "stores",
            "filename": session.get("filename",""),
            "clusters": clusters,
            "country":  session.get("country","mx"),
            "stats": {
                "total_rows":    sum(len(c["members"]) for c in clusters),
                "total_clusters":len(clusters),
                "total_ok":      sum(c["ok_count"] for c in clusters),
                "total_bad":     sum(c["bad_count"] for c in clusters),
            }
        })
    elif session.get("dishes_groups"):
        groups = session.get("dishes_groups",[])
        ok  = sum(1 for g in groups if g["revision"]==1)
        bad = sum(1 for g in groups if g["revision"]==0)
        inc = sum(1 for g in groups if g["revision"]==2)
        return jsonify({
            "type":     "dishes",
            "filename": session.get("dishes_filename",""),
            "groups":   groups,
            "meta":     {"total_groups":len(groups),"ok":ok,"bad":bad,
                         "incomplete":inc,"threshold":0.75,"model_ready":_model_ready}
        })
    return jsonify({"type": None})

@app.route("/model_status")
def model_status():
    get_model()
    return jsonify({"ready":_model_ready,"error":_model_error,"model":MODEL_NAME})

@app.route("/pg_status")
def pg_status():
    if not pg_is_configured():
        return jsonify({"ok":False,"error":"No configurado (.env)"})
    try:
        import psycopg2
        cfg = get_pg_config()
        conn = psycopg2.connect(**cfg, connect_timeout=3)  # 3 segundos máximo
        conn.close()
        return jsonify({"ok":True})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)})

@app.route("/cluster_search", methods=["POST"])
def cluster_search():
    """Busca clusters en dim_maestra por nombre o dirección.
    Devuelve clusters únicos con sus miembros para trabajar sobre ellos
    independientemente del archivo en revisión.
    """
    if not pg_is_configured():
        return jsonify({"error": "BD no configurada"}), 400
    d = request.json or {}
    q = (d.get("query") or "").strip()
    country = (d.get("country") or session.get("country","mx")).strip()
    if not q or len(q) < 3:
        return jsonify({"error": "Query muy corta (mín 3 chars)"}), 400

    table = get_pg_table()
    # Normalizar query para búsqueda
    q_norm = re.sub(r'[^\w\s]', ' ', q.lower()).strip()
    words = [w for w in q_norm.split() if len(w) >= 3]
    if not words:
        return jsonify({"error": "Sin términos de búsqueda válidos"}), 400

    # Construir condición OR por palabra en nombre o dirección
    conditions = []
    for w in words[:4]:
        conditions.append(f"(app_name ilike '%{w}%' OR app_address ilike '%{w}%')")
    where = " AND ".join(conditions)

    sql = f"""
        SELECT DISTINCT cluster_index, cluster_name, cluster_address, country
        FROM {table}
        WHERE country = '{country}'
          AND ({where})
        ORDER BY cluster_name
        LIMIT 30
    """
    rows, err = pg_query(sql)
    if err:
        return jsonify({"error": err}), 400

    if not rows:
        return jsonify({"clusters": [], "count": 0})

    # Traer todos los miembros de los clusters encontrados
    ci_list = ", ".join(f"'{str(r.get('cluster_index',''))}'" for r in rows)
    members_sql = f"""
        SELECT cluster_index, store_id, item_index, app_name, app_address,
               app_latitude, app_longitude, scraper_source, main_chain
        FROM {table}
        WHERE country = '{country}'
          AND cluster_index IN ({ci_list})
        ORDER BY cluster_index, store_id
    """
    members_rows, err2 = pg_query(members_sql)
    if err2:
        members_rows = []

    # Agrupar miembros por cluster
    by_cluster = {}
    for r in members_rows:
        ci = str(r.get("cluster_index",""))
        m = {k: str(v) if v is not None else "" for k, v in r.items()}
        m["is_anchor"] = (str(r.get("item_index","")) == ci)
        by_cluster.setdefault(ci, []).append(m)

    clusters = []
    for r in rows:
        ci = str(r.get("cluster_index",""))
        clusters.append({
            "cluster_id":      ci,
            "cluster_index":   ci,
            "anchor_name":     str(r.get("cluster_name","")),
            "anchor_address":  str(r.get("cluster_address","")),
            "members":         by_cluster.get(ci, []),
            "ok_count":        0,
            "bad_count":       len(by_cluster.get(ci, [])),
        })

    return jsonify({"clusters": clusters, "count": len(clusters)})

@app.route("/pg_query", methods=["POST"])
def run_pg_query():
    """
    Ejecuta la SQL de búsqueda, obtiene los cluster_index que matchean,
    luego trae TODOS los miembros de esos clusters para poder evaluar si están limpios.
    """
    data = request.json
    sql  = data.get("sql","").strip()
    if not sql: return jsonify({"error":"SQL vacía"}), 400
    if not pg_is_configured():
        return jsonify({"error":"BD no configurada. Revisa el archivo .env"}), 400

    # 1. Query original — encontrar clusters que matchean
    rows, err = pg_query(sql)
    if err: return jsonify({"error": err}), 400

    # Obtener cluster_indexes únicos encontrados
    matched_clusters = list({str(r.get("cluster_index","")) for r in rows if r.get("cluster_index")})

    if not matched_clusters:
        return jsonify({"clusters_found": [], "row_count": 0})

    # 2. Segunda query — traer TODOS los miembros de esos clusters
    ci_list = ", ".join(f"'{ci}'" for ci in matched_clusters)
    # Detectar country de la primera query
    country = "mx"
    for line in sql.lower().split("\n"):
        if "country" in line and "=" in line:
            m = re.search(r"country\s*=\s*'(\w+)'", line)
            if m: country = m.group(1); break

    full_sql = f"""select
  cluster_index, store_id, item_index, cluster_name, main_chain,
  app_name, app_address, app_longitude, app_latitude, scraper_source
from {get_pg_table()}
where country = '{country}'
  and cluster_index in ({ci_list})
order by cluster_index, store_id"""

    all_rows, err2 = pg_query(full_sql)
    if err2:
        # Fallback: usar los resultados originales si la segunda query falla
        all_rows = rows

    # Agrupar por cluster_index mostrando todos los miembros
    by_cluster = {}
    for row in all_rows:
        ci = str(row.get("cluster_index",""))
        m = {k: str(v) if v is not None else "" for k, v in row.items()}
        m["is_anchor"] = (str(row.get("item_index","")) == ci)
        by_cluster.setdefault(ci, []).append(m)

    clusters_found = [
        {
            "cluster_index": ci,
            "members": mems,
            "is_clean": False,
            "matched": ci in set(matched_clusters),  # indicar cuáles matchearon la query
        }
        for ci, mems in by_cluster.items()
    ]

    return jsonify({"clusters_found": clusters_found, "row_count": len(all_rows)})

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f: return jsonify({"error":"Sin archivo"}), 400
    ext  = Path(f.filename).suffix.lower()
    path = UPLOAD_FOLDER / f"review{ext}"
    f.save(path)
    try: df = load_file(path)
    except Exception as e: return jsonify({"error":str(e)}), 400

    review_type = request.form.get("review_type", "stores_restaurant")
    session["review_type"] = review_type

    clusters, meta = build_clusters(df, filename=f.filename)
    country = detect_country(df)

    # Intentar restaurar estado previo para este archivo
    saved_state, saved_at = load_session_state(f.filename)
    restored = False
    subgroups_restored = {}
    if saved_state:
        clusters, subgroups_restored = apply_session_state(clusters, saved_state)
        restored = True
        meta["restored"] = True
        meta["saved_at"] = saved_at

    # Marcar clusters ya revisados previamente
    reviewed_set = set()
    try:
        conn = sqlite3.connect(DB_FILE)
        rows = conn.execute("SELECT DISTINCT cluster_index FROM reviewed_clusters").fetchall()
        reviewed_set = {r[0] for r in rows}
        conn.close()
    except: pass
    for cl in clusters:
        cl["already_reviewed"] = cl["cluster_id"] in reviewed_set

    session["review_path"] = str(path)
    session["clusters"]    = clusters
    session["country"]     = country
    session["filename"]    = f.filename

    # Si el archivo ya fue marcado como revisado antes, restaurar ese estado
    already_reviewed = False
    try:
        conn = sqlite3.connect(DB_FILE)
        row = conn.execute(
            "SELECT COUNT(*) FROM reviewed_files WHERE file_name=?", (f.filename,)
        ).fetchone()
        conn.close()
        already_reviewed = row and row[0] > 0
    except: pass
    session["is_reviewed"] = already_reviewed
    session.modified = True

    return jsonify({"clusters":clusters, "meta":meta, "country":country,
                    "filename": session.get("filename",""),
                    "restored": restored,
                    "already_reviewed": already_reviewed,
                    "review_type": review_type,
                    "subgroups_restored": subgroups_restored,
                    "stats":{"total_rows":len(df),"total_clusters":len(clusters),
                             "total_ok":sum(c["ok_count"] for c in clusters),
                             "total_bad":sum(c["bad_count"] for c in clusters)}})

@app.route("/update_revisions", methods=["POST"])
def update_revisions():
    data=request.json; cid=data["cluster_id"]; revisions=data["revisions"]
    record_fb=data.get("record_feedback",True)
    clusters=session.get("clusters",[])
    sql,bad=("",0)
    for cl in clusters:
        if cl["cluster_id"]!=cid: continue
        anchor=next((m for m in cl["members"] if m["is_anchor"]),None)
        for m in cl["members"]:
            ii=m["item_index"]
            if ii not in revisions: continue
            new_rev=revisions[ii]
            if record_fb and not m["is_anchor"] and new_rev!=m["revision"]:
                record_feedback(anchor["app_name"] if anchor else "",
                                anchor["app_address"] if anchor else "",
                                m["app_name"],m["app_address"],
                                m.get("score"),m["revision"],new_rev)
            m["revision"]=new_rev
        cl["ok_count"]=sum(1 for m in cl["members"] if m["revision"]==1)
        cl["bad_count"]=sum(1 for m in cl["members"] if m["revision"]==0)
        bads=[m for m in cl["members"] if m["revision"]==0]
        cl["sql"]=generate_unified_sql(bads)
        sql,bad=cl["sql"],cl["bad_count"]
        break
    session["clusters"]=clusters; session.modified=True
    # Guardar estado en disco en tiempo real
    save_session_state(session.get("filename",""), clusters)
    fb=load_feedback()
    return jsonify({"sql":sql,"bad_count":bad,"threshold":get_threshold(),"fb_stats":fb["stats"]})

@app.route("/get_subgroups", methods=["POST"])
def get_subgroups():
    data=request.json; cid=data["cluster_id"]
    members_data=data["members_data"]; country=data.get("country") or session.get("country","mx")
    subgroups=build_subgroups(members_data)
    for sg in subgroups:
        sg_members=[m for m in members_data if m["item_index"] in sg["members"]]
        sg["sql"]=generate_unified_sql(sg_members,country=country)
    return jsonify({"subgroups":subgroups,"cluster_id":cid})

@app.route("/save_subgroups", methods=["POST"])
def save_subgroups():
    """Guarda el estado actual de subgrupos en session_state."""
    data     = request.json
    subgroups_map = data.get("subgroups", {})  # {cluster_id: [subgrupos]}
    clusters = session.get("clusters", [])
    filename = session.get("filename", "")
    if filename and clusters:
        save_session_state(filename, clusters, subgroups=subgroups_map)
    return jsonify({"ok": True})

@app.route("/set_clean", methods=["POST"])
def set_clean():
    data=request.json; cid=data["cluster_id"]
    clusters=session.get("clusters",[])
    for cl in clusters:
        if cl["cluster_id"]!=cid: continue
        for bdc in cl.get("bd_results",[]):
            if bdc["cluster_index"]==data["bd_cluster_index"]: bdc["is_clean"]=data["is_clean"]
        break
    session["clusters"]=clusters; session.modified=True
    return jsonify({"ok":True})

@app.route("/set_correction", methods=["POST"])
def set_correction():
    data=request.json; cid=data["cluster_id"]
    items=data["item_indexes"]; corr=data["correction"]
    clusters=session.get("clusters",[])
    filename=session.get("filename","")
    for cl in clusters:
        if cl["cluster_id"]!=cid: continue
        anchor=next((m for m in cl["members"] if m["is_anchor"]),None)
        for ii in items:
            if corr:
                cl["corrections"][ii]=corr
                # Guardar par etiquetado: store incorrecto vs anchor del cluster
                member=next((m for m in cl["members"] if m["item_index"]==ii),None)
                if member and anchor and corr:
                    # Par label=0: este store NO es el mismo local que el anchor
                    save_store_pair(
                        member.get("app_name",""), member.get("app_address",""), cid,
                        anchor.get("app_name",""),  anchor.get("app_address",""),  cid,
                        label=0, source="correction", file_name=filename
                    )
            else:
                cl["corrections"].pop(ii,None)
        break
    session["clusters"]=clusters; session.modified=True
    save_session_state(session.get("filename",""), clusters)
    return jsonify({"ok":True})

@app.route("/memory/search")
def memory_search():
    q=request.args.get("q","").strip()
    if not q: return jsonify({"results":[]})
    return jsonify({"results": mem_search(q)})

@app.route("/memory/stats")
def memory_stats_route():
    s=mem_stats()
    return jsonify({"total":s["total"],"unique_items":s["unique_items"],
                    "recent":[{"item_index":r[0],"app_name":r[1],
                               "correction":r[2],"reviewed_at":r[3]} for r in s["recent"]]})

@app.route("/feedback_stats")
def feedback_stats():
    fb=load_feedback()
    return jsonify({"stats":fb["stats"],"threshold":get_threshold(),
                    "threshold_adj":fb.get("threshold_adj",0.0),"pairs_count":len(fb["pairs"])})

@app.route("/reset_feedback", methods=["POST"])
def reset_feedback():
    if FEEDBACK_FILE.exists(): FEEDBACK_FILE.unlink()
    return jsonify({"ok":True})

@app.route("/export")
def export():
    review_path=session.get("review_path")
    clusters=session.get("clusters",[])
    filename=session.get("filename","archivo")
    if not review_path: return jsonify({"error":"Sin sesión"}),400
    df=load_file(review_path)
    rev_map,corr_map={},{}
    mem_records=[]
    for cl in clusters:
        anchor=next((m for m in cl["members"] if m["is_anchor"]),None)
        for m in cl["members"]:
            ii=m["item_index"]
            rev_map[ii]=m["revision"]
            corr=cl["corrections"].get(ii,"")
            corr_map[ii]=corr
            # Guardar en memoria si tiene corrección
            if corr:
                mem_records.append({
                    "item_index":ii,"cluster_id":cl["cluster_id"],
                    "app_name":m["app_name"],"app_address":m["app_address"],
                    "anchor_name":anchor["app_name"] if anchor else "",
                    "correction":corr,"is_new":"NUEVO:" in corr,
                    "file_name":filename,
                })
    if mem_records: mem_save(mem_records)
    # Guardar progreso de revisión
    save_reviewed_clusters(clusters, filename)
    # Guardar resumen del archivo
    ok  = sum(c["ok_count"]  for c in clusters)
    bad = sum(c["bad_count"] for c in clusters)
    save_reviewed_file(filename, session.get("review_type","stores_restaurant"), len(clusters), ok, bad)

    df["item_index"]=df["item_index"].astype(str).str.strip()
    df["revision"]=df["item_index"].map(rev_map).fillna(1).astype(int)
    df["correccion"]=df["item_index"].map(corr_map).fillna("")

    # Hoja 2: correcciones externas acumuladas
    ext_records = ext_corr_list()
    df_ext = pd.DataFrame(ext_records if ext_records else [], columns=[
        "id","store_id","item_index","cluster_index","cluster_name",
        "cluster_address","app_name","app_address","correction","added_at","file_name"
    ])
    # Columnas ordenadas como pidió el usuario
    ext_cols = ["cluster_index","cluster_name","cluster_address",
                "app_name","app_address","item_index","correction"]
    df_ext_out = df_ext[ext_cols] if not df_ext.empty else pd.DataFrame(columns=ext_cols)
    df_ext_out.insert(ext_cols.index("correction")+1, "revision", 0)

    out = UPLOAD_FOLDER/"revisado_final.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Revisión", index=False)
        if not df_ext_out.empty:
            df_ext_out.to_excel(writer, sheet_name="Correcciones externas", index=False)

    # Nombre dinámico: nombre_archivo_revisado_YYYYMMDD_HHMMSS.xlsx
    stem = Path(filename).stem
    ts   = time.strftime("%Y%m%d_%H%M%S")
    download_name = f"{stem}_revisado_{ts}.xlsx"
    return send_file(out, as_attachment=True, download_name=download_name,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.route("/dishes/upload", methods=["POST"])
def dishes_upload():
    f = request.files.get("file")
    if not f: return jsonify({"error":"Sin archivo"}), 400
    ext  = Path(f.filename).suffix.lower()
    path = UPLOAD_FOLDER / f"dishes{ext}"
    f.save(path)
    try: df = load_file(path)
    except Exception as e: return jsonify({"error":str(e)}), 400
    groups, meta = build_dish_groups(df, filename=f.filename)
    session["dishes_path"]     = str(path)
    session["dishes_groups"]   = groups
    session["dishes_filename"] = f.filename
    session["review_type"]     = "dishes"
    session["is_reviewed"]     = False
    # Intentar restaurar estado previo
    saved_state, saved_at = load_session_state("dishes_"+f.filename)
    restored = False
    if saved_state:
        rev_map = {s["group_id"]: s for s in saved_state}
        for g in groups:
            saved = rev_map.get(g["group_id"])
            if saved:
                g["revision"]    = saved["revision"]
                g["fusion_with"] = saved.get("fusion_with","")
        restored = True
        meta["saved_at"] = saved_at
    session["dishes_groups"] = groups; session.modified = True
    # Si el archivo ya fue marcado como revisado antes, restaurar ese estado
    already_reviewed = False
    try:
        conn = sqlite3.connect(DB_FILE)
        row = conn.execute(
            "SELECT COUNT(*) FROM reviewed_files WHERE file_name=?", (f.filename,)
        ).fetchone()
        conn.close()
        already_reviewed = row and row[0] > 0
    except: pass
    session["is_reviewed"] = already_reviewed
    session.modified = True
    return jsonify({"groups": groups, "meta": meta,
                    "filename": f.filename, "restored": restored,
                    "already_reviewed": already_reviewed})

@app.route("/dishes/update", methods=["POST"])
def dishes_update():
    data     = request.json
    group_id = data["group_id"]
    revision = data["revision"]
    fusion   = data.get("fusion_with","")
    groups   = session.get("dishes_groups", [])
    for g in groups:
        if g["group_id"] == group_id:
            g["revision"]    = revision
            g["fusion_with"] = fusion
            # Si marca como 0, recalcular categorías de error
            if revision in (0, 2):
                g["error_cats"] = detect_error_categories(g["members"])
            else:
                g["error_cats"] = []
            break
            break
    session["dishes_groups"] = groups; session.modified = True
    state = [{"group_id": g["group_id"], "revision": g["revision"],
              "fusion_with": g.get("fusion_with","")} for g in groups]
    filename = session.get("dishes_filename","")
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT OR REPLACE INTO session_state (file_name, state_json, saved_at)
        VALUES (?,?,?)
    """, ("dishes_"+filename, json.dumps(state), time.time()))
    conn.commit(); conn.close()
    ok  = sum(1 for g in groups if g["revision"]==1)
    bad = sum(1 for g in groups if g["revision"]==0)
    inc = sum(1 for g in groups if g["revision"]==2)
    return jsonify({"ok": ok, "bad": bad, "incomplete": inc})

@app.route("/dishes/set_fusion", methods=["POST"])
def dishes_set_fusion():
    """Marca dos grupos para fusionar (revision=2) con el menor ID como destino."""
    data   = request.json
    gid_a  = data["group_a"]
    gid_b  = data["group_b"]
    groups = session.get("dishes_groups",[])
    # El destino es el menor group_id
    try:
        dest = str(min(int(gid_a), int(gid_b)))
    except:
        dest = min(gid_a, gid_b)
    for g in groups:
        if g["group_id"] in (gid_a, gid_b):
            g["revision"]    = 2
            g["fusion_with"] = dest
            g["error_cats"]  = []
    session["dishes_groups"] = groups; session.modified = True
    # Guardar par etiquetado label=2 (mismo plato, deben fusionarse)
    ga = next((g for g in groups if g["group_id"]==gid_a), None)
    gb = next((g for g in groups if g["group_id"]==gid_b), None)
    if ga and gb and ga.get("members") and gb.get("members"):
        save_dish_pair(
            ga["members"][0].get("product_name",""),
            ga["members"][0].get("product_description",""),
            gb["members"][0].get("product_name",""),
            gb["members"][0].get("product_description",""),
            label=2, source="fusion",
            file_name=session.get("dishes_filename","")
        )
    # Guardar estado
    state = [{"group_id": g["group_id"], "revision": g["revision"],
              "fusion_with": g.get("fusion_with","")} for g in groups]
    filename = session.get("dishes_filename","")
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""INSERT OR REPLACE INTO session_state (file_name, state_json, saved_at)
        VALUES (?,?,?)""", ("dishes_"+filename, json.dumps(state), time.time()))
    conn.commit(); conn.close()
    return jsonify({"ok": True, "dest": dest})

@app.route("/dishes/export")
def dishes_export():
    dishes_path = session.get("dishes_path")
    groups      = session.get("dishes_groups", [])
    filename    = session.get("dishes_filename","archivo")
    if not dishes_path: return jsonify({"error":"Sin sesión"}), 400
    df = load_file(dishes_path)
    rev_map    = {g["group_id"]: g["revision"]           for g in groups}
    fusion_map = {g["group_id"]: g.get("fusion_with","") for g in groups}
    cats_map   = {g["group_id"]: ",".join(g.get("error_cats",[])) for g in groups}
    df["grupo_pn9_max"] = df["grupo_pn9_max"].astype(str).str.strip()
    df["revision"]      = df["grupo_pn9_max"].map(rev_map).fillna(1).astype(int)
    df["fusion"]        = df["grupo_pn9_max"].map(fusion_map).fillna("")
    df["error_cats"]    = df["grupo_pn9_max"].map(cats_map).fillna("")
    # Guardar en memoria
    conn = sqlite3.connect(DB_FILE)
    for g in groups:
        conn.execute("""
            INSERT INTO reviewed_clusters
              (cluster_index, cluster_name, had_errors, corrections_count, reviewed_at, file_name)
            VALUES (?,?,?,?,?,?)
        """, (g["group_id"], "", 1 if g["revision"] in (0,2) else 0,
              0, time.time(), filename))
    conn.commit(); conn.close()
    # Guardar resumen del archivo
    ok  = sum(1 for g in groups if g["revision"]==1)
    bad = sum(1 for g in groups if g["revision"]==0)
    inc = sum(1 for g in groups if g["revision"]==2)
    save_reviewed_file(filename, "dishes", len(groups), ok, bad, inc)
    out = UPLOAD_FOLDER/"dishes_revisado.xlsx"
    df.to_excel(out, index=False)
    stem = Path(filename).stem
    ts   = time.strftime("%Y%m%d_%H%M%S")
    download_name = f"{stem}_revisado_{ts}.xlsx"
    return send_file(out, as_attachment=True,
                     download_name=download_name,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/mark_as_reviewed", methods=["POST"])
def mark_as_reviewed():
    """Registra el archivo como revisado en memoria sin descargar."""
    review_type = session.get("review_type","stores_restaurant")
    pairs_added = 0
    if review_type == "dishes":
        groups   = session.get("dishes_groups",[])
        filename = session.get("dishes_filename","")
        ok  = sum(1 for g in groups if g["revision"]==1)
        bad = sum(1 for g in groups if g["revision"]==0)
        inc = sum(1 for g in groups if g["revision"]==2)
        save_reviewed_file(filename, "dishes", len(groups), ok, bad, inc)
        save_reviewed_clusters_dishes(groups, filename)
        session["is_reviewed"] = True
        session.modified = True
    else:
        clusters = session.get("clusters",[])
        filename = session.get("filename","")
        ok  = sum(c["ok_count"]  for c in clusters)
        bad = sum(c["bad_count"] for c in clusters)
        save_reviewed_clusters(clusters, filename)
        save_reviewed_file(filename, review_type, len(clusters), ok, bad)
        # Registrar feedback final
        fb = load_feedback()
        for cl in clusters:
            anchor = next((m for m in cl["members"] if m["is_anchor"]), None)
            if not anchor: continue
            for m in cl["members"]:
                if m["is_anchor"] or m.get("score") is None: continue
                model_rev = 1 if m["score"] >= cl.get("threshold_used", 0.80) else 0
                human_rev = m["revision"]
                if model_rev != human_rev:
                    save_store_pair(
                        anchor["app_name"], anchor["app_address"], cl["cluster_id"],
                        m["app_name"], m["app_address"], cl["cluster_id"],
                        label=human_rev, source="reviewed", file_name=filename
                    )
                    fb["pairs"].append({
                        "anchor_name": anchor["app_name"],
                        "anchor_addr": anchor["app_address"],
                        "member_name": m["app_name"],
                        "member_addr": m["app_address"],
                        "model_score": m["score"],
                        "model_rev":   model_rev,
                        "human_rev":   human_rev,
                        "ts":          time.time()
                    })
                    fb["stats"]["total"] += 1
                    if human_rev == 1: fb["stats"]["overrides_to_1"] += 1
                    else:              fb["stats"]["overrides_to_0"] += 1
                    pairs_added += 1
        recent = fb["pairs"][-30:]
        if len(recent) >= 5:
            fp  = sum(1 for p in recent if p["model_rev"]==1 and p["human_rev"]==0)
            fn  = sum(1 for p in recent if p["model_rev"]==0 and p["human_rev"]==1)
            adj = max(-0.10, min(0.10, (fp-fn)/len(recent)*0.10))
            fb["threshold_adj"] = round(adj, 4)
        save_feedback(fb)
        session["is_reviewed"] = True
        session.modified = True
    return jsonify({"ok": True, "filename": filename, "pairs_added": pairs_added})

@app.route("/reset_session", methods=["POST"])
def reset_session():
    """Limpia la sesión activa para empezar una nueva revisión."""
    for key in ["clusters","review_path","filename","country",
                "dishes_groups","dishes_path","dishes_filename"]:
        session.pop(key, None)
    session.modified = True
    return jsonify({"ok": True})

@app.route("/upload_bd", methods=["POST"])
def upload_bd():
    f          = request.files.get("file")
    cluster_id = request.form.get("cluster_id","")
    if not f: return jsonify({"error":"Sin archivo"}), 400
    ext  = Path(f.filename).suffix.lower()
    path = UPLOAD_FOLDER / f"bd_{re.sub(r'[^a-z0-9]','_',cluster_id.lower())}{ext}"
    f.save(path)
    try: df = load_file(path)
    except Exception as e: return jsonify({"error":str(e)}), 400
    from flask import jsonify as _j
    rows = [{k:("" if pd.isna(v) else str(v)) for k,v in row.items()} for _,row in df.iterrows()]
    by_cluster = {}
    for row in rows:
        ci = row.get("cluster_index","")
        by_cluster.setdefault(ci,[]).append(row)
    clusters_found = [{"cluster_index":ci,"members":mems,"is_clean":False} for ci,mems in by_cluster.items()]
    return jsonify({"cluster_id":cluster_id,"clusters_found":clusters_found})


@app.route("/ext_correction/add", methods=["POST"])
def ext_correction_add():
    data = request.json
    data["file_name"] = session.get("filename","")
    # Si scraper_source está vacío, buscarlo en la BD por store_id
    if not data.get("scraper_source","").strip() and data.get("store_id","").strip():
        store_id = data["store_id"].strip()
        table = get_pg_table()
        sql = f"SELECT scraper_source FROM {table} WHERE store_id = '{store_id}' LIMIT 1"
        rows, err = pg_query(sql)
        if not err and rows:
            data["scraper_source"] = rows[0].get("scraper_source","")
    ext_corr_add(data)
    # Guardar par etiquetado: store externo es distinto al cluster donde estaba
    save_store_pair(
        data.get("app_name",""), data.get("app_address",""), data.get("cluster_index",""),
        "", "", data.get("correction",""),
        label=0, source="ext_correction", file_name=data["file_name"]
    )
    return jsonify({"ok": True, "count": len(ext_corr_list())})

@app.route("/ext_correction/list")
def ext_correction_list():
    return jsonify({"records": ext_corr_list()})

@app.route("/ext_correction/delete", methods=["POST"])
def ext_correction_delete():
    ext_corr_delete(request.json.get("id"))
    return jsonify({"ok": True})

def progress_stats_dishes():
    """Progreso de revisión de dishes — solo desde memoria interna."""
    files = get_reviewed_files("dishes")
    total_files = len(files)
    total_groups = sum(f["total"] for f in files)
    total_ok     = sum(f["ok"]    for f in files)
    total_bad    = sum(f["bad"]   for f in files)
    total_inc    = sum(f["incomplete"] for f in files)
    return {
        "mode":          "dishes",
        "total_files":   total_files,
        "total_groups":  total_groups,
        "total_ok":      total_ok,
        "total_bad":     total_bad,
        "total_incomplete": total_inc,
        "files":         files
    }

@app.route("/reviewed_files/clean", methods=["POST"])
def reviewed_files_clean():
    """Elimina duplicados en reviewed_files, dejando solo el más reciente por archivo."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        DELETE FROM reviewed_files
        WHERE id NOT IN (
            SELECT MAX(id) FROM reviewed_files GROUP BY file_name, review_type
        )
    """)
    conn.commit()
    remaining = conn.execute("SELECT COUNT(*) FROM reviewed_files").fetchone()[0]
    conn.close()
    return jsonify({"ok": True, "remaining": remaining})

@app.route("/reviewed_files")
def reviewed_files_route():
    review_type = request.args.get("type")
    return jsonify({"files": get_reviewed_files(review_type)})

@app.route("/progress")
def progress():
    mode        = request.args.get("mode", "stores")
    table       = request.args.get("table", "sales_opportunity.dim_maestra")
    country     = request.args.get("country", "mx")
    review_type = request.args.get("review_type", "")

    if mode == "dishes":
        return jsonify(progress_stats_dishes())

    # Determinar tabla y review_type
    if "retail" in table:
        review_type = review_type or "stores_retail"
    else:
        review_type = review_type or "stores_restaurant"

    stats, err = progress_stats(table=table, country=country, review_type=review_type)
    if err: return jsonify({"error": err}), 400
    stats["files"] = get_reviewed_files(review_type)
    return jsonify(stats)

@app.route("/memory/reset", methods=["POST"])
def memory_reset():
    conn=sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM corrections"); conn.commit(); conn.close()
    return jsonify({"ok":True})


@app.route("/dishes/suggestions", methods=["POST"])
def dishes_suggestions():
    """Retorna los 3 grupos más similares al grupo dado."""
    data     = request.json
    group_id = data["group_id"]
    groups   = session.get("dishes_groups", [])
    target   = next((g for g in groups if g["group_id"]==group_id), None)
    if not target: return jsonify({"suggestions":[]})

    others = [g for g in groups if g["group_id"] != group_id]
    scored = []
    for g in others:
        sim = compute_group_similarity(target, g)
        scored.append({"group_id": g["group_id"], "sim": sim,
                       "members": g["members"], "count": g["count"]})
    scored.sort(key=lambda x: x["sim"], reverse=True)
    return jsonify({"suggestions": scored[:3]})


@app.route("/store_pairs/stats")
def store_pairs_stats():
    """Estadísticas de pares etiquetados de stores."""
    conn = sqlite3.connect(DB_FILE)
    total    = conn.execute("SELECT COUNT(*) FROM store_pairs").fetchone()[0]
    by_src   = conn.execute("""
        SELECT source, COUNT(*) FROM store_pairs GROUP BY source
    """).fetchall()
    conn.close()
    return jsonify({
        "total": total,
        "by_source": {r[0]: r[1] for r in by_src}
    })

@app.route("/save_pair", methods=["POST"])
def save_pair():
    """Guarda un par etiquetado de stores para feedback/fine-tuning.
    Llamado automáticamente al fusionar (label=1) o separar (label=0) subgrupos.
    """
    d = request.json or {}
    file_name = d.get("file_name") or session.get("filename", "")
    save_store_pair(
        app_name_a    = d.get("name_a",""),
        app_address_a = d.get("addr_a",""),
        cluster_index_a = d.get("ci_a",""),
        app_name_b    = d.get("name_b",""),
        app_address_b = d.get("addr_b",""),
        cluster_index_b = d.get("ci_b",""),
        label         = int(d.get("label", 1)),
        source        = d.get("source", "subgroup_drag"),
        file_name     = file_name
    )
    return jsonify({"ok": True})


def clean_correction(corr):
    """Elimina prefijo NUEVO: si existe (artefacto de versiones anteriores)."""
    if isinstance(corr, str) and corr.startswith("NUEVO:"):
        return corr[6:]
    return corr

@app.route("/bd_update/preview", methods=["POST"])
def bd_update_preview():
    """Genera preview enriquecido de los INSERTs que se harían en ctrl_restaurant_homologation."""
    if not session.get("is_reviewed"):
        return jsonify({"error": "Debes marcar el archivo como revisado antes de actualizar la BD"}), 403

    clusters   = session.get("clusters", [])
    country    = session.get("country", "mx")
    rt         = session.get("review_type", "stores_restaurant")
    store_type = "retail" if rt == "stores_retail" else "restaurant"
    table      = get_pg_table()
    rows = []

    # Correcciones de subgrupos
    for cl in clusters:
        for ii, corr in cl.get("corrections", {}).items():
            if not corr: continue
            corr = clean_correction(corr)
            member = next((m for m in cl["members"] if m["item_index"] == ii), None)
            if not member: continue
            rows.append({
                "store_id":       member.get("store_id", ""),
                "app_name":       member.get("app_name", ""),
                "app_address":    member.get("app_address", ""),
                "scraper_source": member.get("scraper_source", ""),
                "old_cluster":    member.get("cluster_index", cl["cluster_id"]),
                "new_cluster":    corr,
                "country":        country,
                "store_type":     store_type,
                "source":         "file",
            })

    # Correcciones externas (Anotar) — solo del archivo actual
    current_file = session.get("filename", "")
    for rec in ext_corr_list():
        if not rec.get("correction"): continue
        if rec.get("file_name") and rec["file_name"] != current_file: continue
        rows.append({
            "store_id":       rec.get("store_id", ""),
            "app_name":       rec.get("app_name", ""),
            "app_address":    rec.get("app_address", ""),
            "scraper_source": rec.get("scraper_source", ""),
            "old_cluster":    rec.get("cluster_index", ""),
            "new_cluster":    clean_correction(rec.get("correction", "")),
            "country":        country,
            "store_type":     store_type,
            "source":         "external",
        })

    if not rows:
        return jsonify({"rows": [], "total": 0, "store_type": store_type, "country": country})

    # Enriquecer con contexto de PostgreSQL
    # Traer cluster_name + miembros de old y new clusters únicos
    old_clusters = list({r["old_cluster"] for r in rows if r["old_cluster"]})
    new_clusters = list({r["new_cluster"] for r in rows if r["new_cluster"] and not r["new_cluster"].startswith("NUEVO:")})
    all_clusters = list(set(old_clusters + new_clusters))

    cluster_ctx = {}  # cluster_index → {name, address, members, prev_corrections}
    if all_clusters and pg_is_configured():
        placeholders = ",".join(f"'{c}'" for c in all_clusters)
        sql_ctx = f"""
            SELECT cluster_index, cluster_name, cluster_address,
                   store_id, app_name, app_address, scraper_source, item_index
            FROM {table}
            WHERE cluster_index IN ({placeholders})
            AND country = '{country}'
            ORDER BY cluster_index, item_index
        """
        ctx_rows, err = pg_query(sql_ctx, timeout_ms=15000)
        if not err and ctx_rows:
            for r in ctx_rows:
                ci = r["cluster_index"]
                if ci not in cluster_ctx:
                    cluster_ctx[ci] = {
                        "name":    r["cluster_name"] or "",
                        "address": r["cluster_address"] or "",
                        "members": []
                    }
                cluster_ctx[ci]["members"].append({
                    "store_id":    r["store_id"],
                    "app_name":    r["app_name"],
                    "app_address": r["app_address"],
                    "is_anchor":   r["item_index"] == ci
                })

        # Traer correcciones previas aplicadas a estos clusters
        prev_sql = f"""
            SELECT old_cluster, new_cluster, store_id, load_date
            FROM sales_opportunity.ctrl_restaurant_homologation
            WHERE (old_cluster IN ({placeholders}) OR new_cluster IN ({placeholders}))
            ORDER BY load_date DESC
            LIMIT 100
        """
        prev_rows, _ = pg_query(prev_sql, timeout_ms=10000)
        prev_by_cluster = {}
        if prev_rows:
            for r in prev_rows:
                for ci in [r["old_cluster"], r["new_cluster"]]:
                    if ci in all_clusters:
                        prev_by_cluster.setdefault(ci, []).append({
                            "store_id":    r["store_id"],
                            "old_cluster": r["old_cluster"],
                            "new_cluster": r["new_cluster"],
                            "load_date":   str(r["load_date"])[:10] if r["load_date"] else ""
                        })
        for ci in cluster_ctx:
            cluster_ctx[ci]["prev_corrections"] = prev_by_cluster.get(ci, [])

    # Adjuntar contexto a cada row
    for r in rows:
        r["old_ctx"] = cluster_ctx.get(r["old_cluster"], {})
        r["new_ctx"] = cluster_ctx.get(r["new_cluster"], {})

    return jsonify({"rows": rows, "total": len(rows),
                    "store_type": store_type, "country": country})


@app.route("/bd_update/execute", methods=["POST"])
def bd_update_execute():
    """Ejecuta los INSERTs en ctrl_restaurant_homologation."""
    if not pg_is_configured():
        return jsonify({"error": "BD no configurada"}), 400

    rows     = request.json.get("rows", [])
    inserted = 0
    errors   = []

    conn_str = (f"host={os.environ.get('PG_HOST')} "
                f"port={os.environ.get('PG_PORT','5432')} "
                f"dbname={os.environ.get('PG_DATABASE')} "
                f"user={os.environ.get('PG_USER')} "
                f"password={os.environ.get('PG_PASSWORD')}")
    try:
        import psycopg2
        conn = psycopg2.connect(conn_str)
        cur  = conn.cursor()
        for r in rows:
            try:
                cur.execute("""
                    INSERT INTO sales_opportunity.ctrl_restaurant_homologation
                      (old_cluster, new_cluster, store_id, scraper_source,
                       country, load_date, homologation_type, store_type, data_quality)
                    VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s)
                """, (
                    r["old_cluster"], r["new_cluster"],
                    r["store_id"],    r.get("scraper_source",""),
                    r["country"],     "manual",
                    r["store_type"],  True
                ))
                inserted += 1
            except Exception as e:
                errors.append(str(e))
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"ok": True, "inserted": inserted, "errors": errors})


@app.route("/audit_cluster", methods=["POST"])
def audit_cluster():
    """
    Dado un cluster_index, trae todos sus miembros desde dim_maestra,
    calcula scores vs el ancla, y busca stores dispersas (similares al ancla
    en otros clusters).
    """
    if not pg_is_configured():
        return jsonify({"error": "BD no configurada"}), 400

    data          = request.json
    cluster_index = data.get("cluster_index", "").strip()
    country       = data.get("country", session.get("country", "mx"))
    search_name   = data.get("search_name", "").strip()  # store del subgrupo origen
    search_addr   = data.get("search_addr", "").strip()
    table         = get_pg_table()

    if not cluster_index:
        return jsonify({"error": "cluster_index requerido"}), 400

    # 1. Traer todos los miembros del cluster
    sql_members = f"""
        SELECT cluster_index, cluster_name, cluster_address,
               store_id, app_name, app_address, scraper_source, item_index,
               cluster_latitude, cluster_longitude
        FROM {table}
        WHERE cluster_index = '{cluster_index}'
        AND country = '{country}'
        ORDER BY item_index
    """
    members_raw, err = pg_query(sql_members, timeout_ms=10000)
    if err: return jsonify({"error": err}), 400
    if not members_raw: return jsonify({"error": f"Cluster {cluster_index} no encontrado"}), 404

    # Identificar ancla (item_index == cluster_index)
    anchor = next((m for m in members_raw if m["item_index"] == cluster_index), None)
    if not anchor:
        anchor = members_raw[0]  # fallback: primer miembro

    anchor_name = anchor.get("app_name", "")
    anchor_addr = anchor.get("app_address", "")

    # 2. Calcular scores de cada miembro vs el ancla
    member_names  = [m.get("app_name","")    for m in members_raw]
    member_addrs  = [m.get("app_address","") for m in members_raw]
    scores = batch_semantic_sim(anchor_name, anchor_addr,
                                [{"app_name": n, "app_address": a}
                                 for n, a in zip(member_names, member_addrs)])

    members = []
    for i, m in enumerate(members_raw):
        is_anchor = m["item_index"] == cluster_index
        members.append({
            "item_index":    m["item_index"],
            "store_id":      m["store_id"],
            "app_name":      m["app_name"],
            "app_address":   m["app_address"],
            "scraper_source":m.get("scraper_source",""),
            "is_anchor":     is_anchor,
            "score":         1.0 if is_anchor else round(scores[i], 3),
        })

    # 3. Buscar dispersos
    # Si viene search_name/addr del subgrupo, usarlos — encuentran el cluster correcto
    # aunque sea distinto al ancla del cluster BD
    disp_name = search_name if search_name else anchor_name
    disp_addr = search_addr if search_addr else anchor_addr

    disp_bwords = brand_words(disp_name)
    disp_akeys  = extract_addr_keys(disp_addr)

    dispersos = []
    if disp_bwords:
        name_cond = " and ".join(word_ilike("app_name", w) for w in disp_bwords[:2])
        addr_parts = []
        if disp_akeys:
            addr_parts = [addr_ilike_safe("app_address", w) for w in disp_akeys[:2]]

        where_parts = [f"({name_cond})"]
        if addr_parts:
            where_parts.append("(" + " and ".join(addr_parts) + ")")
        where_clause = " and ".join(where_parts)

        sql_dispersos = f"""
            SELECT cluster_index, cluster_name, cluster_address,
                   store_id, app_name, app_address, scraper_source, item_index
            FROM {table}
            WHERE country = '{country}'
            AND cluster_index != '{cluster_index}'
            AND {where_clause}
            ORDER BY cluster_index
            LIMIT 50
        """
        disp_raw, err2 = pg_query(sql_dispersos, timeout_ms=15000)
        if not err2 and disp_raw:
            # Agrupar por cluster
            disp_by_cluster = {}
            for r in disp_raw:
                ci = r["cluster_index"]
                if ci not in disp_by_cluster:
                    disp_by_cluster[ci] = {
                        "cluster_index": ci,
                        "cluster_name":  r.get("cluster_name",""),
                        "cluster_address":r.get("cluster_address",""),
                        "members": []
                    }
                is_anchor_disp = r["item_index"] == ci
                disp_by_cluster[ci]["members"].append({
                    "store_id":      r["store_id"],
                    "app_name":      r["app_name"],
                    "app_address":   r["app_address"],
                    "scraper_source":r.get("scraper_source",""),
                    "item_index":    r["item_index"],
                    "is_anchor":     is_anchor_disp,
                })
            dispersos = list(disp_by_cluster.values())

    # 4. Correcciones previas en este cluster
    prev_sql = f"""
        SELECT store_id, old_cluster, new_cluster, load_date
        FROM sales_opportunity.ctrl_restaurant_homologation
        WHERE old_cluster = '{cluster_index}' OR new_cluster = '{cluster_index}'
        ORDER BY load_date DESC LIMIT 20
    """
    prev_rows, _ = pg_query(prev_sql, timeout_ms=10000)
    prev_corrections = []
    if prev_rows:
        prev_corrections = [{"store_id": r["store_id"],
                             "old_cluster": r["old_cluster"],
                             "new_cluster": r["new_cluster"],
                             "load_date":   str(r["load_date"])[:10] if r["load_date"] else ""}
                            for r in prev_rows]

    return jsonify({
        "cluster_index":    cluster_index,
        "cluster_name":     anchor.get("cluster_name",""),
        "cluster_address":  anchor.get("cluster_address",""),
        "anchor":           anchor_name,
        "anchor_addr":      anchor_addr,
        "members":          members,
        "dispersos":        dispersos,
        "prev_corrections": prev_corrections,
        "threshold":        get_threshold(),
    })

# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO PLACES — scraping de malls, dark kitchens, centros comerciales
# ═══════════════════════════════════════════════════════════════════════════════

# Import lazy: scraper sólo se carga si se usa este módulo
def _get_scraper():
    import scraper as _s
    return _s


@app.route("/places/stats")
def places_stats():
    s = _get_scraper()
    return jsonify(s.places_stats())


@app.route("/places/list")
def places_list():
    s       = _get_scraper()
    country = request.args.get("country", "")
    ptype   = request.args.get("place_type", "")
    query   = request.args.get("q", "")
    limit   = int(request.args.get("limit", 500))
    rows    = s.places_list(
        country    = country or None,
        place_type = ptype   or None,
        query      = query   or None,
        limit      = limit,
    )
    return jsonify({"rows": rows, "total": len(rows)})


@app.route("/places/scrape", methods=["POST"])
def places_scrape():
    """
    Lanza un scraping sincrónico.
    Body JSON:
      {
        "source":     "overpass" | "internal_db",
        "place_type": "mall" | "strip_center" | "market" | "commercial" | "dark_kitchen",
        "country":    "cl" | "mx" | "co" | ...,
        "city":       "Santiago"   (opcional, sólo para overpass)
      }
    Retorna progreso y resultado cuando termina.
    """
    s    = _get_scraper()
    data = request.json or {}

    source     = data.get("source", "overpass")
    place_type = data.get("place_type", "mall")
    country    = data.get("country", "cl").lower()
    city       = data.get("city", "").strip() or None

    messages = []
    def log(msg): messages.append(msg)

    if source == "internal_db":
        if not pg_is_configured():
            return jsonify({"error": "BD no configurada — agrega credenciales en .env"}), 400
        result = s.scrape_dark_kitchens_db(
            pg_query_fn  = pg_query,
            country      = country,
            min_brands   = int(data.get("min_brands", 4)),
            min_stores   = int(data.get("min_stores", 5)),
            progress_cb  = log,
        )
    else:
        result = s.scrape_overpass(
            place_type  = place_type,
            country     = country,
            city        = city,
            progress_cb = log,
        )

    result["log"] = messages
    return jsonify(result)


@app.route("/places/delete/<int:place_id>", methods=["DELETE"])
def places_delete(place_id):
    s = _get_scraper()
    s.places_delete(place_id)
    return jsonify({"ok": True, "deleted": place_id})


@app.route("/places/export")
def places_export():
    """Genera y descarga el Excel con filtros opcionales."""
    s       = _get_scraper()
    country = request.args.get("country", "") or None
    ptype   = request.args.get("place_type", "") or None
    query   = request.args.get("q", "") or None

    path, err = s.export_places_excel(country=country, place_type=ptype, query=query)
    if err:
        return jsonify({"error": err}), 400

    from flask import send_file as _sf
    import os
    filename = f"places_{country or 'all'}_{ptype or 'all'}.xlsx"
    return _sf(path, as_attachment=True, download_name=filename,
               mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
@app.route("/places/verify/<int:place_id>", methods=["POST"])
def places_verify(place_id):
    verified = request.json.get("verified", 1)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE places SET verified=? WHERE id=?", (verified, place_id))
    conn.commit(); conn.close()
    return jsonify({"ok": True, "id": place_id, "verified": verified})

if __name__=="__main__":
    print("\n"+"="*54)
    print("  DQ Matching Tool")
    print(f"  PostgreSQL: {'✓ configurado' if pg_is_configured() else '✗ falta .env'}")
    print(f"  Memoria:    {DB_FILE}")
    print("  http://localhost:5000")
    print("="*54+"\n")
    import threading
    threading.Thread(target=get_model,daemon=True).start()
    app.run(debug=False,port=5000)
