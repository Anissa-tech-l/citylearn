"""
reference_agents.py
Agents de référence pour comparaison avec SAC sur CityLearn
- Random Agent (baseline aléatoire)
- Rule-Based Agent (heuristique simple)
"""

import numpy as np


class RandomAgent:
    """Agent baseline aléatoire"""
    
    def __init__(self, action_dim=5):
        self.action_dim = action_dim
        print(f"✅ Random Agent créé (action_dim={action_dim})")
    
    def select_action(self, observation):
        """Sélectionne une action aléatoire dans [0, 1]"""
        return np.random.uniform(0, 1, self.action_dim)


class RuleBasedAgent:
    """
    Agent basé sur des règles simples
    Stratégie: Décharger les batteries quand la demande est faible, 
    charger quand la demande est forte
    """
    
    def __init__(self):
        self.prev_action = None
        print("✅ Rule-Based Agent créé")
    
    def select_action(self, observation):
        """
        Sélectionne une action basée sur des heuristiques
        
        Observation typique CityLearn:
        - Consommation électrique
        - Températures
        - État des batteries
        - Prix de l'électricité
        """
        
        if isinstance(observation, dict):
            # Si c'est un dict, extraire les valeurs
            obs_array = np.array(list(observation.values()))
        else:
            obs_array = np.array(observation)
        
        # Normaliser l'observation
        obs_array = np.clip(obs_array, 0, 1)
        
        # Stratégie simple:
        # - Si prix élevé (dernière valeur souvent), décharger
        # - Si prix bas, charger
        # - Sinon, action neutre
        
        action_dim = 5  # Nombre de bâtiments ou actions
        action = np.zeros(action_dim if action_dim > 1 else 1)
        
        if len(obs_array) > 0:
            # Moyenne de l'observation
            mean_obs = np.mean(obs_array)
            
            # Si demande/prix élevé -> décharger (action basse)
            if mean_obs > 0.7:
                action = np.full_like(action, 0.2, dtype=float)
            # Si demande/prix bas -> charger (action haute)
            elif mean_obs < 0.3:
                action = np.full_like(action, 0.8, dtype=float)
            # Sinon action neutre
            else:
                action = np.full_like(action, 0.5, dtype=float)
        
        # Gérer le cas scalaire
        if action.size == 1:
            return float(action[0])
        
        self.prev_action = action.copy()
        return action


class ThresholdAgent:
    """
    Agent basé sur des seuils
    Utilisable comme baseline supplémentaire
    """
    
    def __init__(self, charge_threshold=0.3, discharge_threshold=0.7):
        self.charge_threshold = charge_threshold
        self.discharge_threshold = discharge_threshold
        print("✅ Threshold Agent créé")
    
    def select_action(self, observation):
        """Sélectionne une action basée sur des seuils"""
        
        if isinstance(observation, dict):
            obs_array = np.array(list(observation.values()), dtype=float)
        else:
            obs_array = np.array(observation, dtype=float)
        
        obs_array = np.clip(obs_array, 0, 1)
        
        # Appliquer logique seuils
        action = np.where(
            obs_array < self.charge_threshold,
            0.9,  # Charger beaucoup
            np.where(
                obs_array > self.discharge_threshold,
                0.1,  # Décharger beaucoup
                0.5   # Action neutre
            )
        )
        
        if action.size == 1:
            return float(action[0])
        
        return action


class SmartChargingAgent:
    """
    Agent de charge intelligente
    Stratégie: Charger aux heures creuses, décharger aux heures de pointe
    """
    
    def __init__(self):
        self.hour_cycle = 0
        print("✅ Smart Charging Agent créé")
    
    def select_action(self, observation):
        """
        Sélectionne une action basée sur les heures
        Simulation simple: cycle de 24h
        """
        
        if isinstance(observation, dict):
            obs_array = np.array(list(observation.values()), dtype=float)
        else:
            obs_array = np.array(observation, dtype=float)
        
        obs_array = np.clip(obs_array, 0, 1)
        
        # Heure simulée (0-23)
        hour = int((self.hour_cycle % 24))
        self.hour_cycle += 1
        
        # Heures creuses: 22h-6h -> charger
        # Heures de pointe: 12h-18h -> décharger
        # Autres: action neutre
        
        if 22 <= hour or hour < 6:
            # Heures creuses
            action = 0.8
        elif 12 <= hour < 18:
            # Heures de pointe
            action = 0.2
        else:
            # Heures neutres
            action = 0.5
        
        return np.full_like(obs_array, action, dtype=float)


class NoOpAgent:
    """
    Agent No-Op (ne rien faire)
    Action neutre à 0.5 tout le temps
    """
    
    def __init__(self):
        print("✅ No-Op Agent créé")
    
    def select_action(self, observation):
        """Retourne toujours une action neutre"""
        if isinstance(observation, dict):
            obs_size = len(observation)
        else:
            obs_size = len(observation) if hasattr(observation, '__len__') else 1
        
        return np.full(obs_size, 0.5, dtype=float)


# ============================================================
# Test des agents
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("  TEST DES AGENTS DE RÉFÉRENCE")
    print("="*70)
    
    # Observation test
    test_obs = np.random.uniform(0, 1, 10)
    
    agents = {
        'Random Agent': RandomAgent(action_dim=5),
        'Rule-Based Agent': RuleBasedAgent(),
        'Threshold Agent': ThresholdAgent(),
        'Smart Charging Agent': SmartChargingAgent(),
        'No-Op Agent': NoOpAgent()
    }
    
    print("\n" + "="*70)
    print("  TESTS D'ACTIONS")
    print("="*70 + "\n")
    
    for name, agent in agents.items():
        action = agent.select_action(test_obs)
        print(f"{name:25} | Action: {action}")
    
    print("\n✅ Tests complétés!")
