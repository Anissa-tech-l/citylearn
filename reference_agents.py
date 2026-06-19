"""
reference_agents.py
Agent de référence pour comparaison avec SAC sur CityLearn
- RBC (Rule-Based Control) - Baseline heuristique
"""

import numpy as np


class RuleBasedAgent:
    """
    Agent basé sur des règles simples (RBC)
    Stratégie: Décharger les batteries quand la demande est forte,
    charger quand la demande est faible
    
    C'est la baseline de référence pour comparer avec SAC
    """
    
    def __init__(self):
        self.prev_action = None
        print("✅ RBC Agent créé")
    
    def select_action(self, observation):
        """
        Sélectionne une action basée sur des heuristiques simples
        
        Observation typique CityLearn:
        - Consommation électrique
        - Températures intérieures/extérieures
        - État des batteries
        - Prix de l'électricité
        - Rayonnement solaire
        
        Stratégie RBC:
        - Prix/Demande élevée (obs > 0.7) → Décharger batteries (action = 0.2)
        - Prix/Demande basse (obs < 0.3) → Charger batteries (action = 0.8)
        - Prix/Demande moyen (0.3-0.7) → Action neutre (action = 0.5)
        """
        
        if isinstance(observation, dict):
            # Si c'est un dict, extraire les valeurs
            obs_array = np.array(list(observation.values()))
        else:
            obs_array = np.array(observation)
        
        # Normaliser l'observation
        obs_array = np.clip(obs_array, 0, 1)
        
        # Nombre d'actions (généralement nombre de bâtiments ou stocks)
        action_dim = len(obs_array) if obs_array.ndim == 1 else 1
        
        # Calculer moyenne de l'observation
        mean_obs = np.mean(obs_array) if len(obs_array) > 0 else 0.5
        
        # Appliquer heuristique basée sur seuils
        if mean_obs > 0.7:
            # Demande/Prix élevée → Décharger (action basse)
            action_value = 0.2
        elif mean_obs < 0.3:
            # Demande/Prix basse → Charger (action haute)
            action_value = 0.8
        else:
            # Demande/Prix moyenne → Action neutre
            action_value = 0.5
        
        # Créer action
        if action_dim > 1:
            action = np.full(action_dim, action_value, dtype=float)
        else:
            action = float(action_value)
        
        self.prev_action = action.copy() if isinstance(action, np.ndarray) else action
        return action


# ============================================================
# Test du RBC Agent
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("  TEST DU RBC AGENT")
    print("="*70)
    
    agent = RuleBasedAgent()
    
    # Test 1: Observation avec demande basse
    print("\n📊 Test 1: Demande BASSE (obs = [0.1, 0.2, 0.15])")
    obs_low = np.array([0.1, 0.2, 0.15])
    action = agent.select_action(obs_low)
    print(f"   Action: {action} (attendu: ~0.8 pour charger)")
    
    # Test 2: Observation avec demande haute
    print("\n📊 Test 2: Demande HAUTE (obs = [0.8, 0.75, 0.85])")
    obs_high = np.array([0.8, 0.75, 0.85])
    action = agent.select_action(obs_high)
    print(f"   Action: {action} (attendu: ~0.2 pour décharger)")
    
    # Test 3: Observation avec demande moyenne
    print("\n📊 Test 3: Demande MOYENNE (obs = [0.4, 0.5, 0.6])")
    obs_mid = np.array([0.4, 0.5, 0.6])
    action = agent.select_action(obs_mid)
    print(f"   Action: {action} (attendu: ~0.5 neutre)")
    
    # Test 4: Observation dict (format CityLearn)
    print("\n📊 Test 4: Observation DICT (format CityLearn)")
    obs_dict = {
        'building_0_electricity': 0.7,
        'building_0_temperature': 0.5,
        'electricity_price': 0.8
    }
    action = agent.select_action(obs_dict)
    print(f"   Action: {action} (attendu: ~0.2 car moyenne = 0.67 > 0.7)")
    
    print("\n✅ Tests complétés!")
    print("\n" + "="*70)
    print("  RBC Agent est prêt pour comparaison avec SAC")
    print("="*70)
