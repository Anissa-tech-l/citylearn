"""
run_full_pipeline.py
Pipeline complet: Entraînement SAC → Évaluation → Graphiques
Version: 1.0 - Publication PFEE
"""

import sys
import time
import argparse
from pathlib import Path

# Import des modules
from citylearn_sac_training import train_sac_on_citylearn
from citylearn_sac_evaluation import run_complete_evaluation
import json
import matplotlib.pyplot as plt
import numpy as np


def print_header(text):
    """Affiche un header formaté"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def print_section(text):
    """Affiche une section"""
    print(f"\n{'─'*80}")
    print(f"  📌 {text}")
    print(f"{'─'*80}\n")


def main(
    train_episodes=50,
    eval_episodes=5,
    steps_per_episode=168,
    batch_size=64,
    learning_rate=3e-4,
    skip_training=False,
    model_path='models/sac_citylearn_trained.pth'
):
    """
    Exécute le pipeline complet
    
    Args:
        train_episodes: Nombre d'épisodes d'entraînement
        eval_episodes: Nombre d'épisodes d'évaluation
        steps_per_episode: Steps par épisode
        batch_size: Batch size pour l'entraînement
        learning_rate: Taux d'apprentissage
        skip_training: Sauter l'entraînement (utiliser un modèle existant)
        model_path: Chemin du modèle
    """
    
    print_header("PIPELINE COMPLET: SAC SUR CITYLEARN - PFEE 2024")
    
    start_time = time.time()
    
    # ============================================================
    # ÉTAPE 1: ENTRAÎNEMENT
    # ============================================================
    
    if not skip_training:
        print_section("ÉTAPE 1: ENTRAÎNEMENT DE L'AGENT SAC")
        
        try:
            agent, training_history = train_sac_on_citylearn(
                episodes=train_episodes,
                steps_per_episode=steps_per_episode,
                learning_rate=learning_rate,
                batch_size=batch_size,
                save_path=model_path
            )
            
            print_section("✅ ENTRAÎNEMENT RÉUSSI")
            print(f"  📊 Récompense moyenne: {np.mean(training_history['episode_rewards']):.2f}")
            print(f"  ⏱️  Temps total: {training_history['training_time']/60:.2f} min")
            
        except Exception as e:
            print(f"\n❌ ERREUR ENTRAÎNEMENT: {e}")
            print("   Vérifiez que CityLearn est correctement installé")
            return
    else:
        print_section("⏭️  ENTRAÎNEMENT IGNORÉ (Mode évaluation)")
        print(f"   Utilisation du modèle: {model_path}")
    
    # ============================================================
    # ÉTAPE 2: ÉVALUATION
    # ============================================================
    
    print_section("ÉTAPE 2: ÉVALUATION SAC vs BASELINES")
    
    try:
        evaluation_results = run_complete_evaluation(
            model_path=model_path,
            episodes=eval_episodes,
            steps_per_episode=steps_per_episode
        )
        
        print_section("✅ ÉVALUATION RÉUSSIE")
        
        # Afficher top 3
        sorted_results = sorted(
            evaluation_results.items(),
            key=lambda x: x[1]['summary']['mean_reward'],
            reverse=True
        )
        
        print("  🏆 TOP 3 AGENTS:")
        for i, (name, data) in enumerate(sorted_results[:3], 1):
            print(f"    {i}. {name:<25} | Reward: {data['summary']['mean_reward']:7.2f}")
        
    except Exception as e:
        print(f"\n❌ ERREUR ÉVALUATION: {e}")
        print("   Vérifiez que le modèle existe ou que l'entraînement s'est bien déroulé")
        return
    
    # ============================================================
    # ÉTAPE 3: RAPPORT FINAL
    # ============================================================
    
    total_time = time.time() - start_time
    
    print_header("📋 RAPPORT FINAL")
    
    print(f"{'Métrique':<30} {'Valeur':<40}")
    print("─"*70)
    print(f"{'Durée totale':<30} {total_time/60:>38.2f} min")
    
    if not skip_training:
        print(f"{'Épisodes d\'entraînement':<30} {train_episodes:>38}")
        print(f"{'Reward moyen (entraînement)':<30} {np.mean(training_history['episode_rewards']):>38.2f}")
    
    print(f"{'Épisodes d\'évaluation':<30} {eval_episodes:>38}")
    
    # Meilleur agent
    best_agent = sorted_results[0]
    print(f"{'Meilleur agent':<30} {best_agent[0]:>38}")
    print(f"{'Meilleur reward':<30} {best_agent[1]['summary']['mean_reward']:>38.2f}")
    
    print("\n" + "="*80)
    print("  📁 FICHIERS GÉNÉRÉS")
    print("="*80 + "\n")
    
    files_generated = [
        ("models/sac_citylearn_trained.pth", "Modèle SAC entraîné"),
        ("training_history.json", "Historique d'entraînement"),
        ("training_curves.png", "Graphiques d'entraînement"),
        ("evaluation_results.json", "Résultats d'évaluation"),
        ("evaluation_comparison.png", "Graphique de comparaison"),
        ("evaluation_boxplot.png", "Box plot des rewards"),
    ]
    
    for filename, description in files_generated:
        print(f"  ✅ {filename:<40} → {description}")
    
    print("\n" + "="*80)
    print("  ✨ PIPELINE TERMINÉ AVEC SUCCÈS!")
    print("="*80 + "\n")
    
    # Générer un rapport texte
    generate_text_report(
        training_history if not skip_training else None,
        evaluation_results,
        total_time,
        train_episodes if not skip_training else 0,
        eval_episodes
    )


def generate_text_report(training_history, evaluation_results, total_time, train_episodes, eval_episodes):
    """Génère un rapport texte détaillé"""
    
    report = []
    report.append("="*80)
    report.append("RAPPORT D'ÉVALUATION - SAC SUR CITYLEARN")
    report.append("="*80)
    report.append("")
    
    # Section entraînement
    if training_history:
        report.append("ENTRAÎNEMENT")
        report.append("-" * 80)
        report.append(f"Épisodes: {train_episodes}")
        report.append(f"Temps total: {training_history['training_time']/60:.2f} min")
        report.append(f"Récompense moyenne: {np.mean(training_history['episode_rewards']):.2f}")
        report.append(f"Récompense max: {np.max(training_history['episode_rewards']):.2f}")
        report.append(f"Récompense min: {np.min(training_history['episode_rewards']):.2f}")
        report.append("")
    
    # Section évaluation
    report.append("ÉVALUATION")
    report.append("-" * 80)
    report.append(f"Épisodes: {eval_episodes}")
    report.append("")
    
    report.append("Résultats par agent:")
    report.append("")
    
    sorted_results = sorted(
        evaluation_results.items(),
        key=lambda x: x[1]['summary']['mean_reward'],
        reverse=True
    )
    
    for rank, (name, data) in enumerate(sorted_results, 1):
        summary = data['summary']
        report.append(f"{rank}. {name}")
        report.append(f"   Mean Reward:  {summary['mean_reward']:10.2f}")
        report.append(f"   Std Reward:   {summary['std_reward']:10.2f}")
        report.append(f"   Min Reward:   {summary['min_reward']:10.2f}")
        report.append(f"   Max Reward:   {summary['max_reward']:10.2f}")
        
        if summary['mean_energy'] > 0:
            report.append(f"   Mean Energy:  {summary['mean_energy']:10.2f} kWh")
        if summary['mean_cost'] > 0:
            report.append(f"   Mean Cost:    {summary['mean_cost']:10.2f} DZD")
        
        report.append("")
    
    # Footer
    report.append("="*80)
    report.append(f"Rapport généré: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Durée totale: {total_time/60:.2f} min")
    report.append("="*80)
    
    # Sauvegarder rapport
    with open('evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("✅ Rapport texte généré: evaluation_report.txt\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline complet SAC sur CityLearn')
    
    parser.add_argument('--train-episodes', type=int, default=50,
                        help='Nombre d\'épisodes d\'entraînement (défaut: 50)')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Nombre d\'épisodes d\'évaluation (défaut: 5)')
    parser.add_argument('--steps-per-episode', type=int, default=168,
                        help='Steps par épisode (défaut: 168 = 7 jours)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (défaut: 64)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Taux d\'apprentissage (défaut: 3e-4)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Sauter l\'entraînement (utiliser modèle existant)')
    parser.add_argument('--model-path', type=str, default='models/sac_citylearn_trained.pth',
                        help='Chemin du modèle (défaut: models/sac_citylearn_trained.pth)')
    
    args = parser.parse_args()
    
    # Créer répertoire modèles si nécessaire
    Path('models').mkdir(exist_ok=True)
    
    main(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        steps_per_episode=args.steps_per_episode,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        skip_training=args.skip_training,
        model_path=args.model_path
    )
