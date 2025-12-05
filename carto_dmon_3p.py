import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union


def generate_zoning_map(gdf_bat, csv_path):
    print("--- Génération de la Carte de Zonage (Style Urbaniste) ---")

    # 1. Chargement
    df_res = pd.read_csv(csv_path)
    gdf_bat["rnb_id"] = gdf_bat["rnb_id"].astype(str)
    df_res["id_bat"] = df_res["id_bat"].astype(str)
    com_92 = gpd.read_file(
        "data/BDT_3-5_GPKG_LAMB93_D092-ED2025-06-15.gpkg", layer="commune"
    )
    com_92 = com_92.loc[com_92["code_insee_du_departement"] == "92"]
    # Fusion
    gdf_final = gdf_bat.merge(df_res, left_on="rnb_id", right_on="id_bat", how="inner")

    # Identifier le fond (le plus gros cluster)
    top_cluster = gdf_final["community"].value_counts().idxmax()

    # Séparation
    gdf_background = gdf_final[gdf_final["community"] == top_cluster]
    gdf_structure = gdf_final[gdf_final["community"] != top_cluster]

    # --- CRÉATION DES ZONES (Agglomération) ---
    print("Calcul des zones (Buffer & Union)... cela peut prendre quelques secondes.")
    zones = []
    # On itère sur chaque communauté détectée (sauf le fond)
    for community_id in gdf_structure["community"].unique():
        subset = gdf_structure[gdf_structure["community"] == community_id]

        merged_geom = subset.geometry.buffer(40).unary_union

        # Si c'est un MultiPolygon, on garde tout
        zones.append({"community": community_id, "geometry": merged_geom})

    gdf_zones = gpd.GeoDataFrame(zones, crs=gdf_bat.crs)

    # --- FIGURE : VUE GLOBALE ---
    fig, ax = plt.subplots(figsize=(15, 15))

    # 1. Le Fond Urbain (Gris très clair, presque blanc)
    # On le dessine en premier pour donner le contexte sans polluer
    gdf_background.plot(ax=ax, color="#eeeeee", edgecolor="none", zorder=1)

    # 2. Les Zones de Communautés (Transparence)
    # Cela "remplit" les vides entre les bâtiments des clusters
    gdf_zones.plot(
        ax=ax, column="community", cmap="tab20", alpha=0.3, edgecolor="none", zorder=2
    )

    # 3. Les Bâtiments des Communautés (Solide)
    # Pour que la structure bâtie reste nette par dessus la zone
    gdf_structure.plot(
        ax=ax,
        column="community",
        cmap="tab20",
        edgecolor="white",
        linewidth=0.1,
        zorder=3,
    )

    # 4. Le contour des communes
    com_92.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
        zorder=4,
    )
    ax.set_title(
        "Urban Morphology Identification \n(Colored areas correspond to neighbourhoods detected beyond the standard urban fabric)",
        fontsize=14,
    )
    ax.set_axis_off()

    plt.savefig("img/figure_3_zoning_style_en.png", dpi=300, bbox_inches="tight")
    print("Carte sauvegardée : figure_3_zoning_style.png")
    plt.show()


# --- EXECUTION ---


if "gdf_bat" in locals():
    generate_zoning_map(gdf_bat, "out/final_building_communities_dmon3p.csv")
else:
    # from src.utils import create_golden_datasets
    # gdf_bat, _, _, _, _ = create_golden_datasets()
    gdf_bat = gpd.read_file("data/gdf_bat.geojson")
    generate_zoning_map(gdf_bat, "out/final_building_communities_dmon3p.csv")
