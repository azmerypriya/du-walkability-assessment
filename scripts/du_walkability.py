#!/usr/bin/env python3
"""
DU Campus Pedestrian Walkability Analysis
==========================================
Computes 6 walkability metrics from:
  - Footpath network  : line.kmz   (KML LineStrings)
  - Points of Interest: LOCATION_POINT_categorized.xlsx
  - Campus boundary   : DU-boundary-export.geojson

Metrics computed
----------------
  M1  Pedestrian Accessibility Ratio (PAR)
  M2  Network Connectivity Index
  M3  Shortest-Path Distance Matrix & Detour Index
  M4  Network Density
  M5  Functional-Zone Path Directness
  M6  Walkable Catchment Area  (5-min / 10-min Isochrones)

Author  : Azmery Priya
Purpose : Master's Thesis — Pedestrian Walkability, University of Dhaka Campus
"""

import xml.etree.ElementTree as ET
import json
import math
import zipfile
import io
import warnings
import numpy as np
import networkx as nx
import openpyxl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import cm
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, shape
from shapely.ops import unary_union, nearest_points, substring

warnings.filterwarnings("ignore")

# ===========================================================
# CONFIGURATION
# ===========================================================

# File paths
KMZ_PATH = "./data/line.kmz"
POI_PATH = "./data/LOCATION_POINT_categorized.xlsx"
BOUNDARY_PATH = "./data/DU-boundary-export.geojson"
OUTPUT_DIR = "./data/output"

# Parameters
SNAP_TOLERANCE_DEG = 0.00009  # ~10 m grid for vertex merging
PAR_THRESHOLD_M = 50.0  # metres: max snap distance for "served" POI
WALK_SPEED_MPS = 1.4  # m/s  (WHO standard pedestrian speed)
ISO_5MIN_M = WALK_SPEED_MPS * 5 * 60  # 420 m
ISO_10MIN_M = WALK_SPEED_MPS * 10 * 60  # 840 m
PATH_BUFFER_DEG = 0.00008  # ~8 m path buffer for isochrone area polygon

# Reference latitude for degree → metre conversion
LAT_REF = 23.73
M_PER_DEG_LAT = 111_000
M_PER_DEG_LON = 111_000 * math.cos(math.radians(LAT_REF))

# Visualisation
CAT_COLOR = {"academic": "#2196F3", "residential": "#4CAF50", "service": "#FF9800"}
CAT_MARKER = {"academic": "s", "residential": "o", "service": "^"}


# ===========================================================
# SECTION 1 — DATA LOADING
# ===========================================================


def load_footpaths(kmz_path):
    """
    Extract polyline coordinate lists from a KMZ archive.

    A KMZ file is a ZIP containing 'doc.kml'. Each <Placemark> element
    holds a <LineString><coordinates> block with space-separated
    'lon,lat[,elev]' tuples.  Elevation is ignored.

    Returns
    -------
    segments : list of list of (lon, lat) tuples
    """
    with zipfile.ZipFile(kmz_path) as z:
        kml_bytes = z.read("doc.kml")

    tree = ET.parse(io.BytesIO(kml_bytes))
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    segments = []
    for pm in root.findall(".//kml:Placemark", ns):
        raw = pm.find(".//kml:coordinates", ns).text.strip()
        coords = []
        for token in raw.split():
            parts = token.split(",")
            coords.append((float(parts[0]), float(parts[1])))  # lon, lat
        segments.append(coords)

    return segments


def load_pois(xlsx_path):
    """
    Load POI records from the categorised spreadsheet.

    Columns expected: OBJECTID, SHAPE, Name, x (lon), Y (lat), Type

    Returns
    -------
    pois : list of dicts  {id, name, lon, lat, type}
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    pois = []
    for row in rows[1:]:  # skip header
        if row[2] is None:
            continue
        pois.append(
            {
                "id": int(row[0]),
                "name": str(row[2]).strip(),
                "lon": float(row[3]),
                "lat": float(row[4]),
                "type": str(row[5]).lower().strip(),
            }
        )
    return pois


def load_campus_boundary(geojson_path):
    """
    Extract the University of Dhaka MultiPolygon from an OSM GeoJSON export.

    The GeoJSON may contain other features (medical universities, buildings).
    We select the feature where name:en == 'University of Dhaka'.

    Returns
    -------
    campus_geom : shapely MultiPolygon (in WGS-84 degrees)
    area_m2     : float  (approximate area in square metres)
    """
    with open(geojson_path) as f:
        gj = json.load(f)

    for feat in gj["features"]:
        if feat["properties"].get("name:en") == "University of Dhaka":
            geom = shape(feat["geometry"])
            area_m2 = geom.area * M_PER_DEG_LAT * M_PER_DEG_LON
            return geom, area_m2

    raise ValueError("University of Dhaka boundary not found in GeoJSON.")


# ===========================================================
# SECTION 2 — NETWORK GRAPH CONSTRUCTION
# ===========================================================


def build_network_graph(segments):
    """
    Build a weighted undirected graph from footpath polyline segments.

    Algorithm
    ---------
    1. For each segment, snap every vertex to a coarse rectangular grid
       (SNAP_TOLERANCE_DEG ≈ 10 m) so vertices within ~10 m of each other
       are merged into a single node.
    2. Add a weighted edge between each consecutive snapped vertex pair.
       Edge weight = haversine distance in metres.
    3. If two segments share an endpoint (within snap tolerance) the same
       node key is produced, automatically creating an intersection node.

    Parameters
    ----------
    segments : list of list of (lon, lat)

    Returns
    -------
    G               : networkx.Graph
    total_length_m  : float   (sum of all edge weights)
    line_geoms      : list of shapely LineString
    """
    G = nx.Graph()
    total_length = 0.0
    line_geoms = []

    def snap(lon, lat):
        t = SNAP_TOLERANCE_DEG
        return (round(lon / t) * t, round(lat / t) * t)

    for seg in segments:
        line_geoms.append(LineString(seg))
        snapped = [snap(lon, lat) for lon, lat in seg]

        for i in range(len(snapped) - 1):
            u, v = snapped[i], snapped[i + 1]
            if u == v:
                continue
            w = haversine(u[0], u[1], v[0], v[1])
            total_length += w
            if G.has_edge(u, v):
                G[u][v]["weight"] = min(G[u][v]["weight"], w)
            else:
                G.add_edge(u, v, weight=w)

    return G, total_length, line_geoms


# ===========================================================
# SECTION 3 — SNAP POIs TO NETWORK
# ===========================================================


def snap_all_pois(pois, segments, G):
    """
    Project each POI onto the nearest point of the footpath network,
    then insert a virtual node at that projection and connect the POI.

    Algorithm (per POI)
    -------------------
    1. Iterate over every sub-segment (consecutive vertex pair) in all
       polylines and compute the perpendicular nearest point using
       Shapely's nearest_points().
    2. Record the sub-segment with the globally smallest distance.
    3. Compute the snap point's distance from the upstream vertex (u)
       to find the split fractions.
    4. Remove edge (u, v) from G and replace it with (u, snap_node)
       and (snap_node, v), preserving total weight.
    5. Add the POI node and connect it to snap_node.

    Parameters
    ----------
    pois     : list of POI dicts
    segments : list of coordinate lists
    G        : networkx.Graph (modified in-place)

    Returns
    -------
    poi_nodes     : list of hashable node keys (one per POI)
    snap_dists_m  : list of floats (snap distances in metres)
    snap_coords   : list of (lon, lat) for each snap point
    """
    FINE_TOL = SNAP_TOLERANCE_DEG / 10.0  # finer grid for snap nodes

    def snap_fine(lon, lat):
        t = FINE_TOL
        return (round(lon / t) * t, round(lat / t) * t)

    def snap_coarse(lon, lat):
        t = SNAP_TOLERANCE_DEG
        return (round(lon / t) * t, round(lat / t) * t)

    poi_nodes = []
    snap_dists = []
    snap_coords_ = []

    for poi in pois:
        poi_pt = Point(poi["lon"], poi["lat"])
        poi_node = ("poi", poi["id"])

        min_dist = float("inf")
        best_snap_pt = None
        best_u = None
        best_v = None
        best_d_from_u = 0.0

        for seg in segments:
            coarse = [snap_coarse(lon, lat) for lon, lat in seg]
            for i in range(len(seg) - 1):
                u_coord = coarse[i]
                v_coord = coarse[i + 1]
                if u_coord == v_coord:
                    continue
                edge_seg = LineString([u_coord, v_coord])
                _, snap_pt = nearest_points(poi_pt, edge_seg)
                d = haversine(poi["lon"], poi["lat"], snap_pt.x, snap_pt.y)
                if d < min_dist:
                    min_dist = d
                    best_snap_pt = snap_pt
                    best_u = u_coord
                    best_v = v_coord
                    best_d_from_u = haversine(
                        u_coord[0], u_coord[1], snap_pt.x, snap_pt.y
                    )

        # ── Insert snap node into graph ──────────────────────────────────
        snap_key = snap_fine(best_snap_pt.x, best_snap_pt.y)

        if G.has_edge(best_u, best_v):
            orig_w = G[best_u][best_v]["weight"]
            G.remove_edge(best_u, best_v)
            w_u_snap = best_d_from_u
            w_snap_v = max(0.0, orig_w - best_d_from_u)
            if snap_key != best_u and w_u_snap > 0:
                G.add_edge(best_u, snap_key, weight=w_u_snap)
            if snap_key != best_v and w_snap_v > 0:
                G.add_edge(snap_key, best_v, weight=w_snap_v)

        G.add_node(
            poi_node,
            lon=poi["lon"],
            lat=poi["lat"],
            name=poi["name"],
            poi_type=poi["type"],
        )
        G.add_edge(poi_node, snap_key, weight=min_dist)

        poi_nodes.append(poi_node)
        snap_dists.append(min_dist)
        snap_coords_.append((best_snap_pt.x, best_snap_pt.y))

    return poi_nodes, snap_dists, snap_coords_


# ===========================================================
# SECTION 4 — METRIC COMPUTATIONS
# ===========================================================

# ── M1 : Pedestrian Accessibility Ratio ────────────────────


def metric_PAR(snap_dists_m, threshold_m=PAR_THRESHOLD_M):
    """
    PAR = number of POIs within threshold_m of the path network
          ─────────────────────────────────────────────────────
                       total number of POIs

    A POI is considered 'served' if its nearest path is ≤ threshold_m away.
    """
    served = [d for d in snap_dists_m if d <= threshold_m]
    par = len(served) / len(snap_dists_m)
    return {
        "PAR": round(par, 4),
        "served": len(served),
        "total": len(snap_dists_m),
        "threshold_m": threshold_m,
        "snap_dists_m": snap_dists_m,
    }


# ── M2 : Network Connectivity Index ────────────────────────


def metric_connectivity(G, poi_nodes):
    """
    Connectivity Index = number of POI pairs with an existing network path
                         ──────────────────────────────────────────────────
                                 total number of POI pairs

    Uses networkx.has_path() for each pair; time complexity O(n² V).
    """
    n = len(poi_nodes)
    total_pairs = n * (n - 1) // 2
    connected = 0
    pair_matrix = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            if nx.has_path(G, poi_nodes[i], poi_nodes[j]):
                connected += 1
                pair_matrix[i, j] = pair_matrix[j, i] = True

    return {
        "CI": round(connected / total_pairs, 4),
        "connected_pairs": connected,
        "total_pairs": total_pairs,
        "pair_matrix": pair_matrix,
    }


# ── M3 : Shortest-Path Distance Matrix & Detour Index ──────


def metric_distance_matrix(G, pois, poi_nodes):
    """
    For every ordered POI pair (i, j):

      network_distance[i,j]  = Dijkstra shortest path weight (metres)
      euclidean_dist[i,j]    = haversine straight-line distance (metres)
      detour_index[i,j]      = network_distance / euclidean_dist

    Detour Index interpretation
    ---------------------------
      1.0       perfect straight path
      1.1–1.3   minor detour (acceptable)
      1.3–1.6   moderate detour
      > 1.6     significant detour / connectivity gap
    """
    n = len(pois)
    net_matrix = np.full((n, n), np.nan)
    euc_matrix = np.zeros((n, n))
    det_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            if i == j:
                net_matrix[i, j] = 0.0
                euc_matrix[i, j] = 0.0
                det_matrix[i, j] = 1.0
                continue

            ed = haversine(
                pois[i]["lon"], pois[i]["lat"], pois[j]["lon"], pois[j]["lat"]
            )
            euc_matrix[i, j] = ed

            try:
                nd = nx.shortest_path_length(
                    G, poi_nodes[i], poi_nodes[j], weight="weight"
                )
                net_matrix[i, j] = nd
                if ed > 0:
                    det_matrix[i, j] = nd / ed
            except nx.NetworkXNoPath:
                pass  # NaN → disconnected pair

    valid_net = net_matrix[~np.isnan(net_matrix) & (net_matrix > 0)]
    valid_det = det_matrix[~np.isnan(det_matrix) & (det_matrix > 1)]

    return {
        "net_matrix": net_matrix,
        "euc_matrix": euc_matrix,
        "detour_matrix": det_matrix,
        "mean_net_dist_m": (
            round(float(np.mean(valid_net)), 1) if len(valid_net) else np.nan
        ),
        "median_net_dist_m": (
            round(float(np.median(valid_net)), 1) if len(valid_net) else np.nan
        ),
        "mean_detour": (
            round(float(np.mean(valid_det)), 4) if len(valid_det) else np.nan
        ),
        "max_detour": round(float(np.nanmax(det_matrix)), 4),
    }


# ── M4 : Network Density ───────────────────────────────────


def metric_network_density(total_length_m, campus_area_m2):
    """
    Network Density = total footpath length (m)
                      ──────────────────────────
                      campus area (km²)

    International benchmarks for university campuses:
      Low     < 5,000 m/km²
      Medium  5,000–10,000 m/km²
      High    > 10,000 m/km²
    """
    campus_km2 = campus_area_m2 / 1e6
    density = total_length_m / campus_km2
    return {
        "total_length_m": round(total_length_m, 1),
        "campus_area_m2": round(campus_area_m2, 0),
        "campus_area_km2": round(campus_km2, 4),
        "density_m_per_km2": round(density, 1),
    }


# ── M5 : Functional-Zone Path Directness ───────────────────


def metric_zone_directness(pois, detour_matrix):
    """
    Group all POI pairs by their functional zone combination and compute
    mean / median detour index per group.

    Zone pairs: academic↔academic, academic↔residential,
                academic↔service,  residential↔residential,
                residential↔service, service↔service
    """
    cats = ["academic", "residential", "service"]
    idx = {c: [i for i, p in enumerate(pois) if p["type"] == c] for c in cats}
    results = {}

    for ci, c1 in enumerate(cats):
        for c2 in cats[ci:]:
            values = []
            for i in idx[c1]:
                for j in idx[c2]:
                    if i != j and not np.isnan(detour_matrix[i, j]):
                        values.append(detour_matrix[i, j])
            key = f"{c1}↔{c2}"
            if values:
                results[key] = {
                    "mean": round(np.mean(values), 4),
                    "median": round(np.median(values), 4),
                    "std": round(np.std(values), 4),
                    "n_pairs": len(values),
                }
            else:
                results[key] = {
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "n_pairs": 0,
                }
    return results


# ── M6 : Walkable Catchment Area / Isochrone Coverage ──────


def metric_isochrone_coverage(
    G, poi_nodes, campus_geom, thresholds_m=(ISO_5MIN_M, ISO_10MIN_M)
):
    """
    For each POI, run Dijkstra up to max(thresholds).
    Reconstruct the reachable path geometry (partial edges included).
    Buffer paths by PATH_BUFFER_DEG and union all buffers.
    Clip to campus boundary and compute coverage percentage.

    Partial edge handling
    ---------------------
    For edge (u → v) with weight w:
      - If dist[u] + w ≤ threshold  → full edge included
      - If dist[u] < threshold < dist[u]+w → fraction
          f = (threshold − dist[u]) / w  of the edge from u is included
    """
    max_thresh = max(thresholds_m)
    iso_buffers = {t: [] for t in thresholds_m}

    for poi_node in poi_nodes:
        try:
            dist_map = nx.single_source_dijkstra_path_length(
                G, poi_node, cutoff=max_thresh, weight="weight"
            )
        except (nx.NodeNotFound, nx.NetworkXError):
            continue

        for t in thresholds_m:
            reach_lines = []
            for u, v, data in G.edges(data=True):
                w = data.get("weight", 0)
                d_u = dist_map.get(u, float("inf"))
                d_v = dist_map.get(v, float("inf"))

                # Only process edges where at least one endpoint is reachable
                if d_u > t and d_v > t:
                    continue

                # Build segment geometry — only for coordinate-pair nodes
                # POI nodes have form ('poi', id) so we check both elements are float-like
                def is_coord_node(n):
                    return (
                        isinstance(n, tuple)
                        and len(n) == 2
                        and isinstance(n[0], float)
                        and isinstance(n[1], float)
                    )

                if not (is_coord_node(u) and is_coord_node(v)):
                    continue
                seg = LineString([u, v])

                # Determine which direction is closer to source
                if d_u <= d_v:
                    near, far, d_near = u, v, d_u
                else:
                    near, far, d_near = v, u, d_v
                    seg = LineString([v, u])  # flip to near→far direction

                if d_near + w <= t:
                    reach_lines.append(seg)  # full edge
                elif d_near < t:
                    frac = (t - d_near) / w if w > 0 else 1.0
                    frac = min(1.0, max(0.0, frac))
                    try:
                        partial = substring(seg, 0, frac, normalized=True)
                        reach_lines.append(partial)
                    except Exception:
                        reach_lines.append(seg)

            if reach_lines:
                merged = unary_union(reach_lines)
                buffered = merged.buffer(PATH_BUFFER_DEG)
                iso_buffers[t].append(buffered)

    results = {}
    for t in thresholds_m:
        if iso_buffers[t]:
            union_all = unary_union(iso_buffers[t])
            clipped = union_all.intersection(campus_geom)
            pct = (clipped.area / campus_geom.area) * 100
            results[t] = {
                "coverage_pct": round(pct, 2),
                "union_polygon": union_all,
                "clipped_polygon": clipped,
            }
        else:
            results[t] = {
                "coverage_pct": 0.0,
                "union_polygon": None,
                "clipped_polygon": None,
            }
    return results


# ===========================================================
# UTILITY
# ===========================================================


def haversine(lon1, lat1, lon2, lat2):
    """Haversine great-circle distance in metres."""
    R = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(max(0, min(1, a))))


# ===========================================================
# SECTION 5 — VISUALISATIONS
# ===========================================================


def plot_overview(
    pois, segments, campus_geom, snap_coords, snap_dists, poi_nodes, out_path
):
    """Base map: campus boundary + footpaths + categorised POIs."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#f0ede8")
    fig.patch.set_facecolor("white")

    # Campus boundary
    if campus_geom.geom_type == "MultiPolygon":
        for poly in campus_geom.geoms:
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=0.15, fc="#a8d5a2", ec="#3a7d44", lw=1.5)
    else:
        xs, ys = campus_geom.exterior.xy
        ax.fill(xs, ys, alpha=0.15, fc="#a8d5a2", ec="#3a7d44", lw=1.5)

    # Footpaths
    for seg in segments:
        lons, lats = zip(*seg)
        ax.plot(lons, lats, color="#795548", lw=2, alpha=0.8, solid_capstyle="round")

    # Snap lines (POI → network)
    for poi, sc, sd in zip(pois, snap_coords, snap_dists):
        color = "#e53935" if sd > PAR_THRESHOLD_M else "#43a047"
        ax.plot(
            [poi["lon"], sc[0]],
            [poi["lat"], sc[1]],
            color=color,
            lw=0.8,
            ls="--",
            alpha=0.6,
        )

    # POIs
    for poi in pois:
        ax.scatter(
            poi["lon"],
            poi["lat"],
            c=CAT_COLOR[poi["type"]],
            marker=CAT_MARKER[poi["type"]],
            s=80,
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(
            poi["name"],
            (poi["lon"], poi["lat"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=5.5,
            color="#212121",
            zorder=6,
        )

    # Legend
    handles = [
        mpatches.Patch(color=CAT_COLOR[c], label=c.title())
        for c in ["academic", "residential", "service"]
    ]
    handles += [
        plt.Line2D([0], [0], color="#795548", lw=2, label="Footpath"),
        plt.Line2D([0], [0], color="#3a7d44", lw=1.5, ls="-", label="Campus boundary"),
        plt.Line2D(
            [0], [0], color="#43a047", lw=0.8, ls="--", label="Snap ≤50 m (served)"
        ),
        plt.Line2D(
            [0], [0], color="#e53935", lw=0.8, ls="--", label="Snap >50 m (unserved)"
        ),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)

    ax.set_title(
        "University of Dhaka — Footpath Network and Points of Interest",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_distance_heatmap(
    matrix, pois, title, out_path, cmap="YlOrRd", fmt=".0f", unit=""
):
    """Generic heatmap for distance or detour matrix."""
    n = len(pois)
    labels = [p["name"][:22] for p in pois]

    fig, ax = plt.subplots(figsize=(18, 16))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.set_yticklabels(labels, fontsize=5.5)

    thresh = masked.mean()
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val) and i != j:
                color = "white" if val > thresh else "black"
                ax.text(
                    j,
                    i,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    fontsize=3.5,
                    color=color,
                )

    plt.colorbar(im, ax=ax, label=unit, fraction=0.03, pad=0.02)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_top_detour_bars(
    matrix,
    pois,
    title,
    out_path,
    threshold=1.3,
    top_n=15,
    cmap="RdYlGn_r",
):
    """Ranked bar chart of the most severe detour pairs above a threshold."""
    candidates = []
    n = len(pois)

    for i in range(n):
        for j in range(i + 1, n):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            if val >= threshold:
                a_name = pois[i]["name"]
                b_name = pois[j]["name"]
                label = f"{a_name[:20]} — {b_name[:20]}"
                candidates.append((val, label, i, j))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:top_n]

    if not candidates:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No detour pairs above the selected threshold",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
        return

    vals = [c[0] for c in candidates]
    labels = [c[1] for c in candidates]

    cmap_v = plt.cm.get_cmap(cmap)
    norm = Normalize(vmin=min(vals), vmax=max(vals))
    colors = [cmap_v(norm(v)) for v in vals]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(
        threshold,
        color="#e65100",
        ls="--",
        lw=1.2,
        label=f"Threshold ({threshold:.1f})",
    )
    ax.set_xlabel("Detour Index", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    for bar, v in zip(bars, vals):
        ax.text(
            v + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def filter_campus_geom_for_pois(campus_geom, pois):
    """Return only campus polygons that contain at least one POI."""
    if campus_geom is None or campus_geom.is_empty or not pois:
        return None

    poi_points = [Point(p["lon"], p["lat"]) for p in pois]

    def contains_poi(poly):
        return any(poly.covers(pt) for pt in poi_points)

    if campus_geom.geom_type == "Polygon":
        return campus_geom if contains_poi(campus_geom) else None

    if campus_geom.geom_type == "MultiPolygon":
        matched = [poly for poly in campus_geom.geoms if contains_poi(poly)]
        if not matched:
            return None
        if len(matched) == 1:
            return matched[0]
        return MultiPolygon(matched)

    return campus_geom


def plot_isochrone_map(pois, segments, campus_geom, iso_results, out_path):
    """Isochrone overlay on campus map."""
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69), dpi=180)
    titles = {
        ISO_5MIN_M: "5-Minute Isochrone (420 m)",
        ISO_10MIN_M: "10-Minute Isochrone (840 m)",
    }
    colors = {ISO_5MIN_M: "#1565C0", ISO_10MIN_M: "#B71C1C"}
    visible_campus_geom = filter_campus_geom_for_pois(campus_geom, pois)

    for ax, t in zip(axes, [ISO_5MIN_M, ISO_10MIN_M]):
        ax.set_facecolor("#e8e8e8")

        # Campus boundary
        if visible_campus_geom is not None:
            if visible_campus_geom.geom_type == "MultiPolygon":
                for poly in visible_campus_geom.geoms:
                    xs, ys = poly.exterior.xy
                    ax.fill(xs, ys, alpha=0.1, fc="#a8d5a2", ec="#3a7d44", lw=1.2)
            elif visible_campus_geom.geom_type == "Polygon":
                xs, ys = visible_campus_geom.exterior.xy
                ax.fill(xs, ys, alpha=0.1, fc="#a8d5a2", ec="#3a7d44", lw=1.2)

        # Isochrone polygon
        poly = iso_results[t].get("clipped_polygon")
        if poly and not poly.is_empty:

            def plot_poly(p):
                if p.geom_type == "Polygon":
                    xs, ys = p.exterior.xy
                    ax.fill(xs, ys, alpha=0.35, fc=colors[t], ec=colors[t], lw=0.5)
                elif p.geom_type == "MultiPolygon":
                    for sub in p.geoms:
                        plot_poly(sub)

            plot_poly(poly)

        # Footpaths
        for seg in segments:
            lons, lats = zip(*seg)
            ax.plot(lons, lats, color="#795548", lw=1.5, alpha=0.9)

        # POIs
        for poi in pois:
            ax.scatter(
                poi["lon"],
                poi["lat"],
                c=CAT_COLOR[poi["type"]],
                marker=CAT_MARKER[poi["type"]],
                s=50,
                zorder=5,
                edgecolors="white",
                linewidths=0.5,
            )

        cov = iso_results[t]["coverage_pct"]
        ax.set_title(
            f"{titles[t]}\nCampus Coverage: {cov:.1f}%", fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Walkable Catchment Area Analysis — University of Dhaka",
        fontsize=14,
        fontweight="bold",
        y=0.998,
    )

    base_handles = [
        plt.Line2D(
            [0],
            [0],
            linestyle="None",
            marker=CAT_MARKER[c],
            markerfacecolor=CAT_COLOR[c],
            markeredgecolor="white",
            markersize=7,
            label=c.title(),
        )
        for c in ["academic", "residential", "service"]
    ]
    base_handles.append(
        mpatches.Patch(
            facecolor="#a8d5a2",
            edgecolor="#3a7d44",
            alpha=0.15,
            label="Campus area",
        )
    )

    for ax, t in zip(axes, [ISO_5MIN_M, ISO_10MIN_M]):
        handles = base_handles + [
            mpatches.Patch(
                color=colors[t],
                alpha=0.4,
                label="5-min reach area" if t == ISO_5MIN_M else "10-min reach area",
            )
        ]
        ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)

    plt.subplots_adjust(hspace=0.18, top=0.94, left=0.09, right=0.98, bottom=0.06)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_zone_directness(zone_results, out_path):
    """Bar chart comparing mean detour index by zone pair."""
    labels = list(zone_results.keys())
    means = [zone_results[k]["mean"] for k in labels]
    stds = [zone_results[k]["std"] for k in labels]
    ns = [zone_results[k]["n_pairs"] for k in labels]

    # Filter out NaN
    valid = [
        (l, m, s, n) for l, m, s, n in zip(labels, means, stds, ns) if not np.isnan(m)
    ]
    if not valid:
        return
    labels, means, stds, ns = zip(*valid)

    cmap_v = plt.cm.get_cmap("RdYlGn_r")
    m_arr = np.array(means)
    norm = Normalize(vmin=m_arr.min(), vmax=m_arr.max())
    bar_colors = [cmap_v(norm(m)) for m in m_arr]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        capsize=5,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
        error_kw=dict(elinewidth=1, ecolor="#555"),
    )

    ax.axhline(1.0, color="#1B5E20", ls="--", lw=1.2, label="Ideal (1.0)")
    for bar, m, n in zip(bars, means, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{m:.3f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Mean Detour Index", fontsize=10)
    ax.set_xlabel("Functional Zone Pair", fontsize=10)
    ax.set_title(
        "Path Directness by Functional Zone\n(Detour Index ≥ 1.0; lower is better)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(means) + 12)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_par_bars(pois, snap_dists, out_path):
    """Ranked bar chart of POI snap distances with threshold line."""
    paired = sorted(zip(snap_dists, pois), key=lambda x: x[0])
    dists = [p[0] for p in paired]
    names = [p[1]["name"][:28] for p in paired]
    colors = ["#43a047" if d <= PAR_THRESHOLD_M else "#e53935" for d in dists]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(
        range(len(names)), dists, color=colors, edgecolor="white", height=0.7
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.axvline(
        PAR_THRESHOLD_M,
        color="#e65100",
        lw=1.5,
        ls="--",
        label=f"PAR threshold ({PAR_THRESHOLD_M} m)",
    )
    ax.set_xlabel("Distance to Nearest Footpath (metres)", fontsize=10)
    ax.set_title(
        "POI Snap Distances — Pedestrian Accessibility Assessment",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)

    for i, (bar, d) in enumerate(zip(bars, dists)):
        ax.text(d + 1, i, f"{d:.1f} m", va="center", fontsize=6.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_par_bars_a4_portrait(pois, snap_dists, out_path):
    """A4 portrait variant of plot_par_bars with larger Y-axis labels."""
    import textwrap

    paired = sorted(zip(snap_dists, pois), key=lambda x: x[0])
    dists = [p[0] for p in paired]
    raw_names = [p[1]["name"] for p in paired]
    # wrap long names so they use vertical space better
    names = [textwrap.fill(n, 30) for n in raw_names]

    colors = ["#43a047" if d <= PAR_THRESHOLD_M else "#e53935" for d in dists]

    # A4 portrait size in inches (width, height)
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    n = len(names)
    # compute reasonable bar height so bars don't become too thin for many POIs
    bar_height = max(0.45, min(0.9, 8.0 / max(1, n)))

    y = np.arange(n)
    bars = ax.barh(y, dists, color=colors, edgecolor="white", height=bar_height)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)  # larger font for readability
    ax.invert_yaxis()  # keep highest value at top

    # threshold line
    ax.axvline(
        PAR_THRESHOLD_M,
        color="#e65100",
        lw=1.5,
        ls="--",
        label=f"PAR threshold ({PAR_THRESHOLD_M} m)",
    )
    ax.set_xlabel("Distance to Nearest Footpath (metres)", fontsize=11)
    ax.set_title(
        "POI Snap Distances — Pedestrian Accessibility",
        fontsize=14,
        fontweight="bold",
    )

    # set x-limit with small margin
    maxd = max(dists) if dists else PAR_THRESHOLD_M
    ax.set_xlim(0, maxd * 1.12 + 5)

    # annotate values; place inside bar when there's room, otherwise outside
    for i, (bar, d) in enumerate(zip(bars, dists)):
        mid_y = bar.get_y() + bar.get_height() / 2
        inside_threshold = d >= (maxd * 0.12)
        if inside_threshold:
            ax.text(
                d - (maxd * 0.01),
                mid_y,
                f"{d:.1f} m",
                va="center",
                ha="right",
                fontsize=8,
                color="white",
            )
        else:
            ax.text(
                d + (maxd * 0.01) + 1,
                mid_y,
                f"{d:.1f} m",
                va="center",
                ha="left",
                fontsize=8,
                color="#212121",
            )

    ax.legend(fontsize=9)
    plt.subplots_adjust(left=0.32, right=0.96, top=0.95, bottom=0.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ===========================================================
# SECTION 6 — REPORT OUTPUT
# ===========================================================


def print_report(m1, m2, m3, m4, m5, m6, pois):
    sep = "=" * 65

    print(f"\n{sep}")
    print("  DU CAMPUS WALKABILITY ANALYSIS — RESULTS SUMMARY")
    print(sep)

    print(f"\n{'─'*65}")
    print("M1 — Pedestrian Accessibility Ratio (PAR)")
    print(f"{'─'*65}")
    print(f"  Threshold                 : {m1['threshold_m']} m")
    print(f"  Served POIs               : {m1['served']} / {m1['total']}")
    print(f"  PAR                       : {m1['PAR']:.4f}  ({m1['PAR']*100:.1f}%)")
    print()
    print("  POI snap distances (metres):")
    sorted_pois = sorted(zip(m1["snap_dists_m"], pois), key=lambda x: x[0])
    for d, p in sorted_pois:
        flag = "✓" if d <= m1["threshold_m"] else "✗"
        print(f"    {flag} {p['name']:<50s} {d:6.1f} m")

    print(f"\n{'─'*65}")
    print("M2 — Network Connectivity Index")
    print(f"{'─'*65}")
    print(
        f"  Connected pairs           : {m2['connected_pairs']} / {m2['total_pairs']}"
    )
    print(f"  Connectivity Index (CI)   : {m2['CI']:.4f}  ({m2['CI']*100:.1f}%)")

    print(f"\n{'─'*65}")
    print("M3 — Shortest-Path Distance Matrix & Detour Index")
    print(f"{'─'*65}")
    print(f"  Mean network distance     : {m3['mean_net_dist_m']} m")
    print(f"  Median network distance   : {m3['median_net_dist_m']} m")
    print(f"  Mean Detour Index         : {m3['mean_detour']:.4f}")
    print(f"  Maximum Detour Index      : {m3['max_detour']:.4f}")

    print(f"\n{'─'*65}")
    print("M4 — Network Density")
    print(f"{'─'*65}")
    print(
        f"  Total footpath length     : {m4['total_length_m']:.1f} m  "
        f"({m4['total_length_m']/1000:.3f} km)"
    )
    print(f"  Campus area               : {m4['campus_area_m2']/1e6:.4f} km²")
    print(f"  Network Density           : {m4['density_m_per_km2']:.1f} m / km²")
    nd = m4["density_m_per_km2"]
    tier = (
        "Low (<5,000)"
        if nd < 5000
        else "Medium (5,000–10,000)" if nd < 10000 else "High (>10,000)"
    )
    print(f"  Benchmark tier            : {tier}")

    print(f"\n{'─'*65}")
    print("M5 — Functional-Zone Path Directness")
    print(f"{'─'*65}")
    print(
        f"  {'Zone Pair':<30s} {'Mean DI':>8s}  {'Median':>8s}  {'Std':>6s}  {'N':>4s}"
    )
    print(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*4}")
    for key, vals in m5.items():
        if vals["n_pairs"] > 0:
            print(
                f"  {key:<30s} {vals['mean']:8.4f}  {vals['median']:8.4f}  "
                f"{vals['std']:6.4f}  {vals['n_pairs']:4d}"
            )

    print(f"\n{'─'*65}")
    print("M6 — Walkable Catchment Area (Isochrone Coverage)")
    print(f"{'─'*65}")
    for t, res in m6.items():
        min_ = t / 60
        print(
            f"  {min_:.0f}-min isochrone ({t:.0f} m)   : {res['coverage_pct']:.2f}% of campus area"
        )

    print(f"\n{sep}\n")


# ===========================================================
# MAIN PIPELINE
# ===========================================================


def main():
    print("\n[1/7] Loading data …")
    segments = load_footpaths(KMZ_PATH)
    pois = load_pois(POI_PATH)
    campus_geom, campus_area_m2 = load_campus_boundary(BOUNDARY_PATH)
    print(
        f"      {len(segments)} footpath segments | {len(pois)} POIs | campus ≈ {campus_area_m2/1e6:.4f} km²"
    )

    print("[2/7] Building network graph …")
    G, total_length_m, line_geoms = build_network_graph(segments)
    print(
        f"      Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | "
        f"Total path: {total_length_m:.1f} m"
    )

    print("[3/7] Snapping POIs to network …")
    poi_nodes, snap_dists, snap_coords = snap_all_pois(pois, segments, G)
    print(
        f"      Max snap: {max(snap_dists):.1f} m | Min snap: {min(snap_dists):.1f} m | "
        f"Mean: {sum(snap_dists)/len(snap_dists):.1f} m"
    )

    print("[4/7] Computing metrics …")
    m1 = metric_PAR(snap_dists)
    m2 = metric_connectivity(G, poi_nodes)
    m3 = metric_distance_matrix(G, pois, poi_nodes)
    m4 = metric_network_density(total_length_m, campus_area_m2)
    m5 = metric_zone_directness(pois, m3["detour_matrix"])
    print("      M1–M5 done")

    print("[5/7] Computing isochrones (this may take a moment) …")
    m6 = metric_isochrone_coverage(G, poi_nodes, campus_geom)
    print(
        f"      5-min coverage: {m6[ISO_5MIN_M]['coverage_pct']:.1f}% | "
        f"10-min coverage: {m6[ISO_10MIN_M]['coverage_pct']:.1f}%"
    )

    print("[6/7] Generating figures …")
    plot_overview(
        pois,
        segments,
        campus_geom,
        snap_coords,
        snap_dists,
        poi_nodes,
        f"{OUTPUT_DIR}/fig1_overview.png",
    )
    plot_par_bars_a4_portrait(
        pois, snap_dists, f"{OUTPUT_DIR}/fig2_par_a4_portrait.png"
    )
    plot_top_detour_bars(
        m3["detour_matrix"],
        pois,
        "Top Detour Pairs Above Threshold",
        f"{OUTPUT_DIR}/fig4_detour_v2.png",
        threshold=1.3,
        top_n=15,
        cmap="RdYlGn_r",
    )
    plot_zone_directness(m5, f"{OUTPUT_DIR}/fig5_zone_directness.png")
    plot_isochrone_map(
        pois, segments, campus_geom, m6, f"{OUTPUT_DIR}/fig6_isochrones.png"
    )

    print("[7/7] Results summary")
    print_report(m1, m2, m3, m4, m5, m6, pois)

    return (
        m1,
        m2,
        m3,
        m4,
        m5,
        m6,
        pois,
        poi_nodes,
        G,
        segments,
        campus_geom,
        snap_dists,
        snap_coords,
    )


if __name__ == "__main__":
    main()
