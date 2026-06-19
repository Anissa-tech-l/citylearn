"""
citylearn_sac_training.py
Entraîne l'agent SAC sur CityLearn avec collecte des métriques réelles
Version: 1.0 - Publication PFEE
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import json
import os
from collections import deque
import time
from citylearn.citylearn import CityLearnEnv
import matplotlib.pyplot as plt

# ============================================================
# 1. RÉSEAU NEURAL POUR SAC
# ============================================================

class SACNetwork(nn.Module):
    """Réseau du politique acteur pour SAC"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACNetwork, self).__init__()
        
        # Couches partagées
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Tête pour la moyenne
        self.mean = nn.Linear(hidden_dim, action_dim)
        
        # Tête pour l'écart-type (log)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        shared_output = self.shared(state)
        mean = self.mean(shared_output)
        log_std = self.log_std(shared_output)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample_action(self, state, evaluate=False):
        """Échantillonne une action avec ou sans bruit"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if evaluate:
            # Mode déterministe (test)
            return torch.tanh(mean)
        else:
            # Mode stochastique (entraînement)
            normal = Normal(mean, std)
            sample = normal.rsample()
            action = torch.tanh(sample)
            return action


class CriticNetwork(nn.Module):
    """Réseau critique Q(s,a) pour SAC"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ============================================================
# 2. AGENT SAC ENTRAÎNABLE
# ============================================================

class SACAgent:
    """Agent SAC avec entraînement sur CityLearn"""
    
    def __init__(self, state_dim, action_dim, learning_rate=3e-4, gamma=0.99, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Réseaux acteur et critique
        self.actor = SACNetwork(state_dim, action_dim).to(self.device)
        self.critic_1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_2 = CriticNetwork(state_dim, action_dim).to(self.device)
        
        # Cibles pour réduction variance
        self.critic_1_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_2_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Optimiseurs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=learning_rate)
        
        # Buffer de réplay
        self.replay_buffer = deque(maxlen=100000)
        self.alpha = 0.2  # Coefficient entropie
        
        print(f"✅ Agent SAC créé (device: {self.device})")
    
    def select_action(self, state, evaluate=False):
        """Sélectionne une action"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.sample_action(state_tensor, evaluate=evaluate)
            action = torch.clamp(action, -1, 1)
            return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Stocke une transition dans le buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size=64):
        """Une étape d'entraînement SAC"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Échantillonner batch
        indices = np.random.randint(0, len(self.replay_buffer), batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # Mise à jour critique
        with torch.no_grad():
            next_actions = self.actor.sample_action(next_states, evaluate=False)
            q1_next = self.critic_1_target(next_states, next_actions)
            q2_next = self.critic_2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + (1 - dones) * self.gamma * q_next
        
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        
        loss_q1 = nn.MSELoss()(q1, target_q)
        loss_q2 = nn.MSELoss()(q2, target_q)
        
        self.critic_1_optimizer.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
        self.critic_1_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
        self.critic_2_optimizer.step()
        
        # Mise à jour acteur
        new_actions = self.actor.sample_action(states, evaluate=False)
        q1_new = self.critic_1(states, new_actions)
        q2_new = self.critic_2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        loss_actor = -q_new.mean()
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Mise à jour des cibles (soft update)
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'loss_q1': loss_q1.item(),
            'loss_q2': loss_q2.item(),
            'loss_actor': loss_actor.item()
        }
    
    def save_model(self, path):
        """Sauvegarde le modèle entraîné"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict()
        }, path)
        print(f"✅ Modèle sauvegardé: {path}")
    
    def load_model(self, path):
        """Charge un modèle entraîné"""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic_1.load_state_dict(ckpt['critic_1_state_dict'])
        self.critic_2.load_state_dict(ckpt['critic_2_state_dict'])
        print(f"✅ Modèle chargé: {path}")


# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================

def normalize_observation(obs, obs_min=0, obs_max=1):
    """Normalise l'observation CityLearn"""
    if isinstance(obs, dict):
        obs_list = []
        for key in sorted(obs.keys()):
            if isinstance(obs[key], (list, np.ndarray)):
                obs_list.extend(obs[key])
            else:
                obs_list.append(obs[key])
        obs = np.array(obs_list, dtype=np.float32)
    else:
        obs = np.array(obs, dtype=np.float32)
    
    return np.clip(obs, obs_min, obs_max)


def rescale_action(action, action_min=0, action_max=1):
    """Rescale l'action de [-1, 1] vers [0, 1]"""
    return (action + 1) / 2 * (action_max - action_min) + action_min


# ============================================================
# 4. BOUCLE D'ENTRAÎNEMENT
# ============================================================

def train_sac_on_citylearn(
    episodes=100,
    steps_per_episode=168,
    learning_rate=3e-4,
    batch_size=64,
    save_path='models/sac_citylearn.pth'
):
    """Entraîne l'agent SAC sur CityLearn"""
    
    print("="*80)
    print("  ENTRAÎNEMENT SAC SUR CITYLEARN - PFEE 2024")
    print("="*80)
    
    # Créer l'environnement
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
    
    # Créer agent
    agent = SACAgent(state_dim, action_dim, learning_rate=learning_rate)
    
    # Historique
    episode_rewards = []
    episode_losses = []
    
    # Entraînement
    print(f"🚀 Début entraînement: {episodes} épisodes × {steps_per_episode} steps\n")
    
    start_training = time.time()
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(steps_per_episode):
            # Normaliser observation
            if isinstance(obs, list):
                obs_single = obs[0]
            else:
                obs_single = obs
            
            obs_normalized = normalize_observation(obs_single)
            
            # Sélectionner action (stochastique en entraînement)
            action = agent.select_action(obs_normalized, evaluate=False)
            action_rescaled = rescale_action(action)
            
            # Exécuter action
            if isinstance(obs, list):
                action_env = [action_rescaled] * len(obs)
            else:
                action_env = action_rescaled
            
            obs_next, reward, done, info = env.step(action_env)
            
            # Normaliser récompense
            if isinstance(reward, (list, np.ndarray)):
                reward = np.mean(reward)
            
            episode_reward += float(reward)
            
            # Normaliser observation suivante
            if isinstance(obs_next, list):
                obs_next_single = obs_next[0]
            else:
                obs_next_single = obs_next
            
            obs_next_normalized = normalize_observation(obs_next_single)
            
            # Stocker transition
            agent.store_transition(
                obs_normalized, action, reward, obs_next_normalized, done
            )
            
            # Entraîner agent
            loss = agent.train_step(batch_size=batch_size)
            if loss:
                episode_loss.append(loss)
            
            obs = obs_next
            
            if done:
                break
        
        # Stats épisode
        episode_rewards.append(episode_reward)
        if episode_loss:
            avg_loss = np.mean([l['loss_actor'] for l in episode_loss])
            episode_losses.append(avg_loss)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"[{episode+1:3d}/{episodes}] Reward: {episode_reward:8.2f} | Avg(10): {avg_reward:8.2f}")
    
    training_time = time.time() - start_training
    
    # Sauvegarder modèle
    agent.save_model(save_path)
    
    # Sauvegarder historique
    history = {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'training_time': training_time,
        'total_steps': episodes * steps_per_episode
    }
    
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Générer graphique d'entraînement
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Rewards
    ax1.plot(episode_rewards, linewidth=2, color='#2ecc71')
    ax1.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax1.set_title('SAC Training on CityLearn - Episode Rewards', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Losses
    if episode_losses:
        ax2.plot(episode_losses, linewidth=2, color='#e74c3c')
        ax2.set_ylabel('Actor Loss', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax2.set_title('Actor Network Loss', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("\n✅ Graphiques sauvegardés: training_curves.png")
    
    print("\n" + "="*80)
    print(f"✅ ENTRAÎNEMENT TERMINÉ!")
    print(f"   📁 Modèle: {save_path}")
    print(f"   📊 Historique: training_history.json")
    print(f"   ⏱️  Temps total: {training_time/60:.2f} min")
    print(f"   📈 Reward moyen: {np.mean(episode_rewards):.2f}")
    print(f"   📈 Reward max: {np.max(episode_rewards):.2f}")
    print("="*80)
    
    return agent, history


if __name__ == "__main__":
    agent, history = train_sac_on_citylearn(
        episodes=50,
        steps_per_episode=168,
        batch_size=64,
        save_path='models/sac_citylearn_trained.pth'
    )
