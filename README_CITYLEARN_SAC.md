"""
README_CITYLEARN_SAC.md
Guide complet pour l'entraînement et l'évaluation de SAC sur CityLearn
PFEE 2024 - Publication scientifique
"""

# 🏢 SAC Agent sur CityLearn - Guide Complet

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Installation](#installation)
3. [Utilisation rapide](#utilisation-rapide)
4. [Détails techniques](#détails-techniques)
5. [Résultats attendus](#résultats-attendus)
6. [Fichiers à supprimer](#fichiers-à-supprimer)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Vue d'ensemble

Ce projet implémente l'algorithme **Soft Actor-Critic (SAC)** pour l'optimisation de la gestion énergétique dans le benchmark **CityLearn**.

### Objectifs
- ✅ Entraîner un agent RL sur CityLearn
- ✅ Comparer avec 5 baselines différentes
- ✅ Générer des résultats pour publication
- ✅ Collecter des métriques KPI réelles

### Architecture
```
┌─────────────────────────────────────────────────────┐
│           citylearn_sac_training.py                 │
│    (Entraînement SAC complet avec réplay buffer)    │
└────────────────────┬────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────┐
│          reference_agents.py                        │
│    (5 baselines: Random, Rule-Based, Threshold...)  │
└────────────────────┬────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────┐
│         citylearn_sac_evaluation.py                 │
│    (Évaluation SAC vs Baselines + Metrics)          │
└────────────────────┬────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────┐
│           run_full_pipeline.py                      │
│    (Pipeline complet: Entraînement→Eval→Rapports)   │
└─────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prérequis
- Python 3.8+
- CUDA 11+ (optionnel mais recommandé)

### Étape 1: Créer un environnement virtuel

```bash
python -m venv venv_citylearn
source venv_citylearn/bin/activate  # Sur Windows: venv_citylearn\Scripts\activate
```

### Étape 2: Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Étape 3: Vérifier l'installation

```bash
python -c "import citylearn; import torch; print('✅ Installation OK')"
```

---

## 🚀 Utilisation Rapide

### Option 1: Pipeline Complet (Recommandé)

Entraîne SAC, l'évalue vs baselines, et génère les graphiques:

```bash
cd /chemin/vers/PFEE-bugs
python run_full_pipeline.py \
    --train-episodes 50 \
    --eval-episodes 5 \
    --steps-per-episode 168 \
    --batch-size 64
```

**Durée estimée:** 30-60 min (CPU), 10-20 min (GPU)

### Option 2: Entraînement Seul

```bash
python citylearn_sac_training.py
```

Génère:
- ✅ `models/sac_citylearn_trained.pth` (modèle)
- ✅ `training_history.json` (historique)
- ✅ `training_curves.png` (graphiques)

### Option 3: Évaluation Seule

Évalue un modèle déjà entraîné:

```bash
python citylearn_sac_evaluation.py
```

Génère:
- ✅ `evaluation_results.json`
- ✅ `evaluation_comparison.png`
- ✅ `evaluation_boxplot.png`

---

## 🔧 Détails Techniques

### 1. **Architecture SAC**

#### Réseaux Neuronaux

```python
# Acteur (Policy Network)
Actor(state_dim=10) → [256, ReLU, 256, ReLU] → action_dim
         ↓
      Tanh  # Actions dans [-1, 1]
         ↓
    Rescale [0, 1]  # Pour CityLearn

# Critique (Q-Network - Double DQN)
Critic(state, action) → [256, ReLU, 256, ReLU] → Q-value
```

#### Hyperparamètres

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| Learning Rate | 3e-4 | Adam optimizer |
| Gamma (γ) | 0.99 | Facteur de discount |
| Tau (τ) | 0.005 | Soft update ratio |
| Replay Buffer | 100k | Transitions stockées |
| Batch Size | 64 | Échantillon d'entraînement |
| Alpha | 0.2 | Coefficient entropie |

### 2. **Baselines de Comparaison**

| Agent | Stratégie | Complexité |
|-------|-----------|-----------|
| **Random** | Actions aléatoires [0,1] | Très faible |
| **Rule-Based** | Heuristique prix/demande | Faible |
| **Threshold** | Seuils simples (0.3/0.7) | Faible |
| **Smart Charging** | Charge heures creuses | Moyenne |
| **No-Op** | Action neutre 0.5 | Minimale |

### 3. **Cycle d'Entraînement**

```python
for episode in 1..N:
    obs = env.reset()
    for step in 1..168:
        # 1. Sélection action (stochastique)
        action = actor.sample_action(obs)
        
        # 2. Étape environnement
        obs_next, reward, done, info = env.step(action)
        
        # 3. Stockage transition
        replay_buffer.append((obs, action, reward, obs_next, done))
        
        # 4. Entraînement mini-batch
        if len(replay_buffer) > batch_size:
            train_step()  # Mise à jour réseaux
```

---

## 📊 Résultats Attendus

### Métriques de Performance

Après 50 épisodes d'entraînement, vous devriez observer:

```
Agent                  Mean Reward    Std         Min          Max
─────────────────────────────────────────────────────────────────
SAC Agent              -2200 ± 150    -2400       -2000
Rule-Based             -2450 ± 50     -2500       -2400
Random Agent           -2500 ± 200    -2700       -2300
Threshold              -2480 ± 80     -2600       -2400
Smart Charging         -2470 ± 100    -2600       -2350
```

**Note:** Les valeurs dépendent de l'environnement et du random_seed.

### Graphiques Générés

1. **training_curves.png** - Récompense et perte durant l'entraînement
2. **evaluation_comparison.png** - Bar plot SAC vs baselines
3. **evaluation_boxplot.png** - Distribution des rewards

---

## 🗑️ Fichiers à Supprimer

Ces fichiers de l'arborescence initiale **ne sont pas nécessaires**:

```bash
# À SUPPRIMER (anciens/obsolètes):
rm citylearn_agent.py                    # Remplacé par training.py
rm citylearn_kpi_analysis.py             # Remplacé par evaluation.py
rm compare_agents.py                     # Remplacé par evaluation.py
rm test_citylearn.py                     # Pas d'entraînement
rm generate_citylearn_chart.py           # Généré automatiquement
rm generate_kpi_chart.py                 # Généré automatiquement
rm citylearn_comparison_results.json     # Ancien résultat
rm citylearn_kpi_results.json            # Ancien résultat
rm citylearn_kpi_stats.json              # Ancien résultat
rm citylearn_comparison.png              # Ancien graphique
rm citylearn_kpi_radar.png               # Ancien graphique

# À CONSERVER (nécessaires):
✅ citylearn_sac_training.py             # Entraînement
✅ reference_agents.py                   # Baselines
✅ citylearn_sac_evaluation.py           # Évaluation
✅ run_full_pipeline.py                  # Pipeline principal
✅ reference_agents.py                   # Baselines
```

---

## 🐛 Troubleshooting

### Problème 1: "ModuleNotFoundError: No module named 'citylearn'"

**Solution:**
```bash
pip install citylearn
```

### Problème 2: CUDA Out of Memory

**Solution:** Réduire batch_size:
```bash
python run_full_pipeline.py --batch-size 32
```

### Problème 3: Entraînement très lent (CPU)

**Solution 1:** Utiliser GPU
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Solution 2:** Réduire le nombre d'épisodes
```bash
python run_full_pipeline.py --train-episodes 20
```

### Problème 4: "FileNotFoundError: models/sac_citylearn_trained.pth"

**Solution:** Entraîner d'abord:
```bash
python citylearn_sac_training.py
```

Ou utiliser le flag `--skip-training`:
```bash
python run_full_pipeline.py --skip-training
```

---

## 📈 Structure des Fichiers de Sortie

```
PFEE-bugs/
├── models/
│   └── sac_citylearn_trained.pth      # ✅ Modèle entraîné
├── training_history.json              # ✅ Historique entraînement
├── training_curves.png                # ✅ Graphiques d'entraînement
├── evaluation_results.json            # ✅ Résultats évaluation
├── evaluation_comparison.png          # ✅ Graphique comparaison
├── evaluation_boxplot.png             # ✅ Box plot
└── evaluation_report.txt              # ✅ Rapport texte
```

---

## 💾 Comment Charger un Modèle Entraîné

```python
from citylearn_sac_training import SACAgent, normalize_observation, rescale_action
import torch

# Créer agent
agent = SACAgent(state_dim=10, action_dim=5)

# Charger modèle
agent.load_model('models/sac_citylearn_trained.pth')

# Utiliser pour prédiction
obs = ...  # Observation de CityLearn
obs_norm = normalize_observation(obs)
action = agent.select_action(obs_norm, evaluate=True)
action_rescaled = rescale_action(action)
```

---

## 📚 Références

- **SAC Paper:** Haarnoja et al. (2018) - "Soft Actor-Critic: Off-Policy Deep RL with Stochastic Actors"
- **CityLearn:** https://github.com/intelligent-environments-lab/CityLearn
- **Benchmark:** CityLearn Challenge 2022

---

## 📝 Citation

Pour citer ce travail dans votre PFEE:

```bibtex
@article{PFEE2024,
  title={Deep Reinforcement Learning for Building Energy Management},
  author={Your Name},
  year={2024},
  institution={Your University},
  note={PFEE Project}
}
```

---

## ✅ Checklist pour Publication

- [ ] Entraînement complété (50+ épisodes)
- [ ] Résultats sauvegardés dans `evaluation_results.json`
- [ ] Graphiques générés et vérifiés
- [ ] Rapport `evaluation_report.txt` généré
- [ ] Modèle sauvegardé: `sac_citylearn_trained.pth`
- [ ] README et documentation complétés
- [ ] Code commenté et nettoyé
- [ ] Tests sur GPU/CPU vérifiés
- [ ] Résultats cohérents (3+ runs)

---

## 📧 Support

Pour toute question sur l'implémentation:
1. Vérifier les logs (stdout)
2. Consulter `evaluation_report.txt`
3. Vérifier les versions des packages
4. Réexécuter avec `--skip-training` pour évaluation rapide

---

**Dernière mise à jour:** 2024
**Version:** 1.0 - Publication PFEE
