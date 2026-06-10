import requests
import geopandas as gpd
import zipfile


def load_admin_borders(GADM_URL) -> gpd.GeoDataFrame:
    """Download GADM level-1 province boundaries for the Netherlands."""
    print("  Downloading province borders …")
    r = requests.get(GADM_URL, timeout=120)
    r.raise_for_status()
    gdf = gpd.read_file(io.BytesIO(r.content))
    return gdf.to_crs(epsg=28992)


def download_geojson(url: str) -> gpd.GeoDataFrame:
    """Download a zipped GeoJSON from HDX and return as GeoDataFrame."""
    print(f"  Downloading {url.split('/')[-1]} …")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        geojson_name = next(n for n in z.namelist() if n.endswith(".geojson"))
        with z.open(geojson_name) as f:
            gdf = gpd.read_file(f)
    print(f"    → {len(gdf)} features loaded")
    return gdf


def split_name(name):
    return name.split(",")[0]
