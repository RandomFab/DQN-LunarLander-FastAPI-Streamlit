import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def load_tfevents_to_df(path):
    # Charger l'accumulateur d'événements
    ea = EventAccumulator(path)
    ea.Reload()

    # Liste des tags disponibles (ex: rollout/ep_rew_mean)
    tags = ea.Tags()['scalars']
    
    data = {}
    for tag in tags:
        # Extraire les valeurs pour chaque tag
        events = ea.Scalars(tag)
        data[tag] = pd.DataFrame(events).drop(columns=['wall_time'])
        # Renommer la colonne 'value' par le nom du tag
        data[tag] = data[tag].rename(columns={'value': tag})

    # Fusionner les dataframes sur les 'step'
    df_final = None
    for tag, df in data.items():
        if df_final is None:
            df_final = df
        else:
            df_final = pd.merge(df_final, df, on='step', how='outer')
    
    return df_final

if __name__ == "__main__":
    # Chemin vers ton log spécifique
    log_path = "logs/dqn_LunarLander_v1/best_HP_1/events.out.tfevents.1771422769.LAPTOP-R6T5Q982.28996.8"
    
    if os.path.exists(log_path):
        df = load_tfevents_to_df(log_path)
        print("Aperçu des données (5 dernières lignes) :")
        print(df.tail())
        
        # Exemple pour afficher la récompense moyenne
        if 'rollout/ep_rew_mean' in df.columns:
            print("\nColonnes disponibles:", df.columns.tolist())
    else:
        print(f"Fichier non trouvé : {log_path}")
