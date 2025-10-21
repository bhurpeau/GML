# -*- coding: utf-8 -*-
#!/usr/bin/env python
import fiona 
WFS_ENDPOINT = "https://data.geopf.fr/wfs/ows" 
PLU_LAYER_NAME = "wfs_du:zone_urba" 


def telecharger_donnees_plu(insee_code):
    partition_value = f'DU_{insee_code}'
    cql_filter = f"partition='{partition_value}'"
    
    wfs_request_url = (
        f'WFS:{WFS_ENDPOINT}?service=WFS&request=GetFeature&typename={PLU_LAYER_NAME}&version=2.0.0'
        f'&outputFormat=json&cql_filter={cql_filter}'
    )
    
    print(f"Téléchargement WFS pour INSEE {insee_code} en cours...")
    
    try:
        # Utilisation d'un timeout pour les grands serveurs
        gdf_plu_zones = gpd.read_file(wfs_request_url) 
        
        if gdf_plu_zones.empty:
             print("Succès de la requête, mais 0 zones retournées. La commune est peut-être couverte par un PLUI non filtrable par INSEE.")
             return None
             
        print(f"Succès ! {len(gdf_plu_zones)} zones PLU téléchargées.")
        return gdf_plu_zones

    except Exception as e:
        print(f"Échec critique du WFS : {e}")
        return None