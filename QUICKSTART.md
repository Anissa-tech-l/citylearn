"""
QUICKSTART.md
Guide de démarrage rapide - SAC sur CityLearn
Pour exécuter le pipeline complet en 5 minutes
"""

# ⚡ Démarrage Rapide - 5 Minutes

## 📥 Installation (2 min)

### Sur Windows:
```bash
# 1. Ouvrir un terminal PowerShell dans le répertoire du projet
cd C:\PFEE-bugs

# 2. Activer l'environnement virtuel (s'il existe)
venv_citylearn\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Sur Linux/Mac:
```bash
cd ~/PFEE-bugs
source venv_citylearn/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Exécution (3 min)

### Option 1: Pipeline Complet (Recommandé)
```bash
python run_full_pipeline.py --train-episodes 30 --eval-episodes 5
```

✅ Génère automatiquement:
- Modèle entraîné
- Graphiques
- Résultats de comparaison

**Durée:** 20-40 min (selon votre CPU/GPU)

### Option 2: Test Rapide (Évaluation seule)
```bash
python run_full_pipeline.py --skip-training --eval-episodes 3
```

**Durée:** 2-3 min (compare avec un modèle non entraîné)

### Option 3: Entraînement Seul
```bash
python citylearn_sac_training.py
```

---

## 📊 Résultats

Après exécution, vous verrez:

```
================================================================================
  PIPELINE COMPLET: SAC SUR CITYLEARN - PFEE 2024
================================================================================

──────────────────────────────────────────────────────────────────────────────
  📌 ÉTAPE 1: ENTRAÎNEMENT DE L'AGENT SAC
──────────────────────────────────────────────────────────────────────────────

[10/30] Reward:   -2450.23 | Avg(10):   -2455.12
[20/30] Reward:   -2380.45 | Avg(10):   -2415.67
[30/30] Reward:   -2350.89 | Avg(10):   -2375.23

✅ ENTRAÎNEMENT RÉUSSI
  📊 Récompense moyenne: -2375.23
  ⏱️  Temps total: 15.45 min

──────────────────────────────────────────────────────────────────────────────
  📌 ÉTAPE 2: ÉVALUATION SAC vs BASELINES
──────────────────────────────────────────────────────────────────────────────

--- Évaluation: SAC Agent ---
  Episode 1/5: -2350.23
  Episode 2/5: -2380.45
  Episode 3/5: -2345.67
  Episode 4/5: -2365.89
  Episode 5/5: -2355.12

✅ ÉVALUATION RÉUSSIE
  🏆 TOP 3 AGENTS:
    1. SAC Agent                 | Reward:   -2359.47
    2. Rule-Based Agent          | Reward:   -2450.12
    3. Smart Charging Agent      | Reward:   -2465.34

================================================================================
  📋 RAPPORT FINAL
================================================================================

Durée totale                       20.34 min
Épisodes d'entraînement            30
Reward moyen (entraînement)        -2375.23
Épisodes d'évaluation              5
Meilleur agent                     SAC Agent
Meilleur reward                    -2359.47

================================================================================
  📁 FICHIERS GÉNÉRÉS
================================================================================

✅ models/sac_citylearn_trained.pth  → Modèle SAC entraîné
✅ training_history.json             → Historique d'entraînement
✅ training_curves.png               → Graphiques d'entraînement
✅ evaluation_results.json           → Résultats d'évaluation
✅ evaluation_comparison.png         → Graphique de comparaison
✅ evaluation_boxplot.png            → Box plot des rewards

================================================================================
  ✨ PIPELINE TERMINÉ AVEC SUCCÈS!
================================================================================
```

---

## 📁 Fichiers Importants

| Fichier | Description | Sortie |
|---------|-------------|--------|
| `run_full_pipeline.py` | Pipeline principal | ✅ Graphiques + Résultats |
| `citylearn_sac_training.py` | Entraînement SAC | ✅ Modèle `.pth` |
| `citylearn_sac_evaluation.py` | Évaluation vs baselines | ✅ Résultats JSON |
| `reference_agents.py` | 5 baselines différentes | - |
| `models/` | Dossier modèles | ✅ `sac_citylearn_trained.pth` |

---

## 🔍 Vérifier les Résultats

### Graphiques
```bash
# Windows
start evaluation_comparison.png
start training_curves.png

# Linux
xdg-open evaluation_comparison.png

# Mac
open evaluation_comparison.png
```

### Résultats JSON
```bash
# Voir les résultats bruts
type evaluation_results.json  # Windows
cat evaluation_results.json   # Linux/Mac
```

---

## ⚙️ Options Avancées

### Augmenter l'entraînement
```bash
python run_full_pipeline.py \
    --train-episodes 100 \
    --eval-episodes 10 \
    --batch-size 128 \
    --steps-per-episode 168
```

### GPU si disponible
```bash
# Vérifier CUDA
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Les scripts détectent CUDA automatiquement
python run_full_pipeline.py --train-episodes 50
```

### Modèle personnalisé
```bash
python run_full_pipeline.py \
    --skip-training \
    --model-path /chemin/vers/mon_modele.pth \
    --eval-episodes 5
```

---

## 🐛 Problèmes Courants

### ❌ "No module named 'citylearn'"
```bash
pip install citylearn
```

### ❌ "CUDA out of memory"
```bash
python run_full_pipeline.py --batch-size 32
```

### ❌ "ModuleNotFoundError: torch"
```bash
pip install torch==2.0.1
```

### ❌ Le script s'arrête
Vérifier que `citylearn` et `torch` sont bien installés:
```bash
python -c "import citylearn, torch; print('✅ OK')"
```

---

## 📋 Checklist

- [ ] Environnement virtuel activé
- [ ] `pip install -r requirements.txt` exécuté
- [ ] `python run_full_pipeline.py` lancé
- [ ] Graphiques générés
- [ ] Résultats JSON vérifiés
- [ ] Rapport texte généré

---

## 📊 Interprétation des Résultats

### Reward Négatif = Normal ✅
CityLearn utilise un système de coûts (négatif = mieux):
- **-2000:** Excellent
- **-2300:** Bon
- **-2500:** Moyen
- **-2700+:** Mauvais

### SAC doit être meilleur que les baselines
```
Baseline typique:  -2450 ± 100
SAC entraîné:      -2350 ± 50   ✅ Meilleur!
```

---

## 🎓 Prochaines Étapes

1. **Entraîner plus longtemps** (100+ épisodes)
2. **Ajuster hyperparamètres** dans `citylearn_sac_training.py`
3. **Générer rapport** pour votre PFEE
4. **Comparer avec votre implémentation antérieure**

---

## 💾 Utiliser le Modèle Entraîné

```python
from citylearn_sac_training import SACAgent, normalize_observation, rescale_action
from citylearn.citylearn import CityLearnEnv

# Charger environnement et modèle
env = CityLearnEnv(schema='citylearn_challenge_2022_phase_all')
agent = SACAgent(state_dim=10, action_dim=5)
agent.load_model('models/sac_citylearn_trained.pth')

# Utiliser pour contrôle
obs = env.reset()
for step in range(168):
    obs_norm = normalize_observation(obs)
    action = agent.select_action(obs_norm, evaluate=True)
    action_rescaled = rescale_action(action)
    obs, reward, done, info = env.step([action_rescaled] * 5)
    if done:
        break
```

---

## 📞 Aide

Pour déboguer:
```bash
# Voir les logs détaillés
python run_full_pipeline.py --train-episodes 10 2>&1 | tee debug.log

# Vérifier l'installation
python -c "import citylearn; print(citylearn.__version__)"
```

---

**Version:** 1.0 | **PFEE 2024** | Dernière mise à jour: 2024
