# DeepQLearning — Banc de poissons (DQN & supervisé)

## Présentation rapide
- Environnement `fishEnv` (boids) dans `fish_class.py`, avec un agent contrôlable (actions: 0 rien, 1 gauche, 2 droite).
- Trois volets d’apprentissage :
  - **DQN contrôle** : notebook `rl/dqn.ipynb`, modèles `rl/dqn_model.h5`, `rl/test_dqn.h5`.
  - **Supervisé (imit. learning)** : notebook `supervised_learning/supervised_model.ipynb`, modèle `supervised_learning/supervised_model.h5`.
  - **DQN classifier (adaptation 2.8)** : notebook `rl/dqn_classifier.ipynb`, modèle `rl/supervised_q_net.h5`.
- UI/Simulation Pygame : `ui/game_loop.py` (DQN) et `ui/game_loop_sup.py` (DQN vs supervisé).

## Pré-requis système
- Python 3.11 (venv déjà présent dans ce dossier).
- Accélération GPU optionnelle : TensorFlow GPU + CUDA/cuDNN (non requis pour lancer les notebooks de démo).

## Dépendances Python principales
- numpy, matplotlib, tensorflow, keras, pygame.
- (Déjà installées dans `venv/`.) Pour recréer l’env : `python -m venv venv && venv/Scripts/activate` puis `pip install numpy matplotlib tensorflow pygame`.

## Lancer les notebooks
- Ouvrir les fichiers `.ipynb` correspondants (voir volets ci-dessus).
- Pour le DQN contrôle (`rl/dqn.ipynb`) : exécuter la collecte, l’entraînement, puis l’évaluation.
- Pour le supervisé : `supervised_learning/supervised_model.ipynb` (collecte des données, entraînement, évaluation, sauvegarde `.h5`).
- Pour l’adaptation DQN classifier : `rl/dqn_classifier.ipynb` (charge `sup_dataset.npz` généré par le notebook supervisé).

## Lancer les UI Pygame
- Activer l’environnement : `venv/Scripts/activate`.
- DQN : `python ui/game_loop.py`
- DQN vs supervisé toggle (touche M) : `python ui/game_loop_sup.py`
- Raccourcis : `SPACE` pause, `R` reset, `C` cône de vision, flèches UP/DOWN vitesse, sliders pour paramètres boids (cohesion/alignment/separation/vision/min_distance).

## Fichiers clés
- Environnement : `fish_class.py`, `utils.py`.
- RL utils : `rl/rl_utils.py`.
- Notebooks : `rl/dqn.ipynb`, `rl/dqn_classifier.ipynb`, `supervised_learning/supervised_model.ipynb`.
- Modèles sauvegardés : `rl/dqn_model.h5`, `rl/test_dqn.h5`, `rl/supervised_q_net.h5`, `supervised_learning/supervised_model.h5`.

## Notes
- Les notebooks loggent les courbes (reward, accuracy, epsilon, temps). Sauvegardes `.h5` sont écrites dans les dossiers `rl/` ou `supervised_learning/`.
- Pour réexécuter proprement, supprimer/ignorer les fichiers `.h5` ou `*.npz` si vous souhaitez un nouvel entraînement.


### Par: Nguio-Mathieu AKOUN et Norman ALIÉ