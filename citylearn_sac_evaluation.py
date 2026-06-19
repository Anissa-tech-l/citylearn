"""
citylearn_sac_evaluation.py
Évalue l'agent SAC entraîné vs baselines sur CityLearn
Version: 1.0 - Publication PFEE
"""

import numpy as np
import torch
import json
import time
from citylearn.citylearn import CityLearnEnv
from citylearn_sac_training import SACAgent, normalize_observation, rescale_action
from reference_agents import RandomAgent, RuleBasedAgent, ThresholdAgent, SmartChargingAgent, NoOpAgent
import matplotlib.pyplot as plt

# ============================================================
# COLLECTEUR DE MÉTRIQUES
# ============================================================

class MetricsCollector:
    """Collecte les métriques de performance"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Réinitialise les métriques"""
        self.rewards = []
        self.energy_consumptions = []
        self.peak_demands = []
        self.costs = []
        self.steps = 0
    
    def collect_step(self, reward, info=None):
        """Collecte les métriques d'une étape"""
        if isinstance(reward, (list, np.ndarray)):
            reward = np.mean(reward)
        self.rewards.append(float(reward))
        self.steps += 1
    
    def collect_episode_metrics(self, env):
        """Collecte les métriques de fin d'épisode"""
        # Essayer de collecter depuis l'environnement
        if hasattr(env, 'buildings'):
            total_energy = 0.0
            total_cost = 0.0
            total_co2 = 0.0
            peak_power = 0.0
            
            for building in env.buildings:
                if hasattr(building, 'energy_consumption'):
                    total_energy += building.energy_consumption
                if hasattr(building, 'cost'):
                    total_cost += building.cost
                if hasattr(building, 'electric_power'):
                    peak_power = max(peak_power, building.electric_power)
            
            self.energy_consumptions.append(total_energy)
            self.costs.append(total_cost)
            self.peak_demands.append(peak_power)
    
    def get_summary(self):
        """Retourne un résumé des métriques"""
        return {
            'mean_reward': float(np.mean(self.rewards)) if self.rewards else 0.0,
            'std_reward': float(np.std(self.rewards)) if self.rewards else 0.0,
            'min_reward': float(np.min(self.rewards)) if self.rewards else 0.0,
            'max_reward': float(np.max(self.rewards)) if self.rewards else 0.0,
            'total_steps': self.steps,
            'mean_energy': float(np.mean(self.energy_consumptions)) if self.energy_consumptions else 0.0,
            'mean_cost': float(np.mean(self.costs)) if self.costs else 0.0,
            'mean_peak_demand': float(np.mean(self.peak_demands)) if self.peak_demands else 0.0,
        }


# ============================================================
# ÉVALUATION DES AGENTS
# ============================================================

def evaluate_agent(agent, env, agent_name, episodes=5, steps_per_episode=168):
    """
    Évalue un agent sur CityLearn
    
    Args:
        agent: Agent à évaluer
        env: Environnement CityLearn
        agent_name: Nom de l'agent pour logs
        episodes: Nombre d'épisodes
        steps_per_episode: Steps par épisode
    
    Returns:
        Dictionnaire avec métriques
    """
    
    print(f"\n--- Évaluation: {agent_name} ---")
    collector = MetricsCollector()
    episode_rewards = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0.0
        
        for step in range(steps_per_episode):
            # Préparer observation
            if isinstance(obs, list):
                obs_single = obs[0]
            else:
                obs_single = obs
            
            # Normaliser observation (sauf pour baselines qui peuvent avoir leurs propres normes)
            if agent_name == 'SAC Agent':
                obs_input = normalize_observation(obs_single)
                action = agent.select_action(obs_input, evaluate=True)
                action_rescaled = rescale_action(action)
            else:
                obs_input = obs_single
                action_rescaled = agent.select_action(obs_input)
            
            # Exécuter action
            if isinstance(obs, list):
                action_env = [action_rescaled] * len(obs) if isinstance(action_rescaled, (int, float)) else action_rescaled
            else:
                action_env = action_rescaled
            
            obs, reward, done, info = env.step(action_env)
            
            # Collecter récompense
            if isinstance(reward, (list, np.ndarray)):
                reward = np.mean(reward)
            
            collector.collect_step(reward, info)
            episode_reward += float(reward)
            
            if done:
                break
        
        collector.collect_episode_metrics(env)
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode+1}/{episodes}: {episode_reward:.2f}")
    
    summary = collector.get_summary()
    return summary, episode_rewards


# ============================================================
# PIPELINE D'ÉVALUATION COMPLET
# ============================================================

def run_complete_evaluation(
    model_path='models/sac_citylearn_trained.pth',
    episodes=5,
    steps_per_episode=168
):
    """
    Exécute l'évaluation complète SAC vs baselines
    """
    
    print("="*80)
    print("  ÉVALUATION COMPLÈTE: SAC vs BASELINES sur CityLearn")
    print("="*80)
    
    # Créer environnement
    schema = 'citylearn_challenge_2022_phase_all'
    env = CityLearnEnv(schema=schema, random_seed=42)
    print(f"✅ Environnement: {schema}")
    
    # Obtenir dimensions
    obs_spaces = env.observation_space
    action_spaces = env.action_space
    
    if isinstance(obs_spaces, list):
        obs_shape = obs_spaces[0].shape
        n_buildings = len(obs_spaces)
    else:
        obs_shape = obs_spaces.shape
        n_buildings = 1
    
    if isinstance(action_spaces, list):
        action_shape = action_spaces[0].shape
    else:
        action_shape = action_spaces.shape
    
    state_dim = obs_shape[0] if len(obs_shape) > 0 else 10
    action_dim = action_shape[0] if len(action_shape) > 0 else 5
    
    print(f"   📊 Bâtiments: {n_buildings} | État: {state_dim} | Actions: {action_dim}\n")
    
    # Créer agent SAC
    sac_agent = SACAgent(state_dim, action_dim)
    try:
        sac_agent.load_model(model_path)
        print(f"✅ Modèle SAC chargé: {model_path}\n")
    except FileNotFoundError:
        print(f"⚠️  Modèle {model_path} non trouvé! Utilisation d'un modèle non entraîné.\n")
    
    # Créer baselines
    baselines = {
        'Random Agent': RandomAgent(action_dim=action_dim),
        'Rule-Based Agent': RuleBasedAgent(),
        'Threshold Agent': ThresholdAgent(),
        'Smart Charging Agent': SmartChargingAgent(),
        'No-Op Agent': NoOpAgent()
    }
    
    # Évaluer tous les agents
    results = {}
    
    # SAC
    print("\n" + "="*80)
    print("  ÉVALUATION SAC (Agent entraîné)")
    print("="*80)
    summary_sac, rewards_sac = evaluate_agent(
        sac_agent, env, 'SAC Agent', episodes, steps_per_episode
    )
    results['SAC Agent'] = {
        'summary': summary_sac,
        'rewards': rewards_sac
    }
    
    # Baselines
    print("\n" + "="*80)
    print("  ÉVALUATION BASELINES")
    print("="*80)
    
    for name, agent in baselines.items():
        summary, rewards = evaluate_agent(
            agent, env, name, episodes, steps_per_episode
        )
        results[name] = {
            'summary': summary,
            'rewards': rewards
        }
    
    # Afficher résumé
    print("\n" + "="*80)
    print("  RÉSUMÉ DES RÉSULTATS")
    print("="*80 + "\n")
    
    print(f"{'Agent':<25} {'Mean Reward':<15} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*76)
    
    for name, data in sorted(results.items(), key=lambda x: x[1]['summary']['mean_reward'], reverse=True):
        summary = data['summary']
        print(f"{name:<25} {summary['mean_reward']:>14.2f} {summary['std_reward']:>11.2f} "
              f"{summary['min_reward']:>11.2f} {summary['max_reward']:>11.2f}")
    
    # Sauvegarder résultats
    results_export = {}
    for name, data in results.items():
        results_export[name] = {
            'summary': data['summary'],
            'rewards': data['rewards']
        }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results_export, f, indent=2)
    
    print("\n✅ Résultats sauvegardés: evaluation_results.json")
    
    # Générer graphiques
    generate_comparison_charts(results)
    
    return results


def generate_comparison_charts(results):
    """Génère les graphiques de comparaison"""
    
    print("\n📊 Génération des graphiques...")
    
    agents = list(results.keys())
    mean_rewards = [results[a]['summary']['mean_reward'] for a in agents]
    std_rewards = [results[a]['summary']['std_reward'] for a in agents]
    
    # Couleurs
    colors = ['#2ecc71' if 'SAC' in a else '#3498db' for a in agents]
    
    # Graphique 1: Bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(agents, mean_rewards, yerr=std_rewards, capsize=10,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Ajouter valeurs sur barres
    for bar, reward in zip(bars, mean_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_xlabel('Agent', fontsize=12, fontweight='bold')
    ax.set_title('Agent Comparison on CityLearn', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('evaluation_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: evaluation_comparison.png")
    
    # Graphique 2: Box plot des rewards
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rewards_list = [results[a]['rewards'] for a in agents]
    bp = ax.boxplot(rewards_list, labels=agents, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.set_xlabel('Agent', fontsize=12, fontweight='bold')
    ax.set_title('Reward Distribution across Episodes', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('evaluation_boxplot.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: evaluation_boxplot.png")


if __name__ == "__main__":
    results = run_complete_evaluation(
        model_path='models/sac_citylearn_trained.pth',
        episodes=5,
        steps_per_episode=168
    )
