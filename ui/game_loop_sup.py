import os
import numpy as np
import pygame

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.models import load_model

from fish_class import fishEnv
from rl.rl_utils import select_action
from utils import draw_fish, show_text, Slider

# ------------------------------------------------------------------
# Charger le modèle supervisé (policy réseau, pas le DQN)
# ------------------------------------------------------------------
SUP_MODEL_PATH = "supervised_learning/supervised_model.h5"
policy_model = load_model(SUP_MODEL_PATH)

q_network = load_model("rl/dqn_fish_model.h5")

print("game_loop_sup")

# ------------------------------------------------------------------
# Constantes écran
# ------------------------------------------------------------------
WIDTH  = 1000
HEIGHT = 600

# 2/3 pour la simulation, 1/3 pour le panneau de contrôle
SIM_WIDTH = int(WIDTH * (2/3))
UI_WIDTH  = WIDTH - SIM_WIDTH

n_fish     = 100
font_color = (255, 255, 255)
bg_color   = (5, 5, 20)

supervised = False
epsilon_eval = 0.0
reward = 0.0

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulation de banc de poissons")
font  = pygame.font.SysFont("consolas", 14)
clock = pygame.time.Clock()

# ------------------------------------------------------------------
# Initialisation environnement
# ------------------------------------------------------------------
env   = fishEnv(SIM_WIDTH, HEIGHT, n_fish)
state = env.reset_rl()   # on initialise l’agent_idx, timestep et le premier state

running = True
paused  = False
speed   = 1  # nombre de steps simulés par frame (tu peux le garder si tu veux)
show_vision_cone = False

# ------------------------------------------------------------------
# Création des sliders dans la zone UI
# ------------------------------------------------------------------
slider_x       = SIM_WIDTH + 20
slider_width   = UI_WIDTH - 40
slider_y_start = 40
slider_y_step  = 60

reset_timer = 0 

sliders = {}

params = env.params  # BoidsParams partagé

sliders["w_cohesion"] = Slider(
    slider_x, slider_y_start,
    slider_width,
    min_val=0.0, max_val=0.2,
    value=params.w_cohesion,
    label="w_cohesion",
    font=font, color=font_color
)

sliders["w_alignment"] = Slider(
    slider_x, slider_y_start + slider_y_step,
    slider_width,
    min_val=0.0, max_val=0.2,
    value=params.w_alignment,
    label="w_alignment",
    font=font, color=font_color
)

sliders["w_separation"] = Slider(
    slider_x, slider_y_start + 2 * slider_y_step,
    slider_width,
    min_val=0.0, max_val=0.5,
    value=params.w_separation,
    label="w_separation",
    font=font, color=font_color
)

sliders["vision_range"] = Slider(
    slider_x, slider_y_start + 3 * slider_y_step,
    slider_width,
    min_val=20, max_val=200,
    value=params.vision_range,
    label="vision_range",
    font=font, color=font_color
)

sliders["min_distance"] = Slider(
    slider_x, slider_y_start + 4 * slider_y_step,
    slider_width,
    min_val=5, max_val=40,
    value=params.min_distance,
    label="min_distance",
    font=font, color=font_color
)

# ------------------------------------------------------------------
# Boucle principale (agent SUPERVISÉ uniquement)
# ------------------------------------------------------------------
while running:
    # ---------- EVENTS ----------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                # Reset complet de l’environnement + de l’agent
                env.reset()
                state = env.reset_rl()
            elif event.key == pygame.K_c:
                show_vision_cone = not show_vision_cone
            elif event.key == pygame.K_UP:
                speed = min(speed + 1, 10)
            elif event.key == pygame.K_DOWN:
                speed = max(speed - 1, 1)
            elif event.key == pygame.K_m:
                # Toggle du mode de contrôle
                supervised = not supervised
                mode_name = "Supervisé" if supervised else "RL (DQN)"
                print(f"[INFO] Changement de mode : {mode_name}")

        # events souris => sliders
        for s in sliders.values():
            s.handle_event(event)

    # ---------- APPLIQUER PARAMS DES SLIDERS ----------
    params.w_cohesion   = sliders["w_cohesion"].value
    params.w_alignment  = sliders["w_alignment"].value
    params.w_separation = sliders["w_separation"].value
    params.vision_range = sliders["vision_range"].value
    params.min_distance = sliders["min_distance"].value

    # ---------- LOGIQUE : agent supervisé ----------
    if not paused:
        if supervised:
            for _ in range(speed):
                # recalculer l’état à partir de l’env courant
                state = env.compute_state()        # shape (state_dim,)

                # prédiction softmax + argmax
                state_input = np.expand_dims(state, axis=0).astype(np.float32)  # (1, state_dim)
                logits = policy_model(state_input)          # (1, n_actions)
                probs  = logits.numpy()[0]                  # (n_actions,) si dernière couche = softmax
                action = int(np.argmax(probs))              # action la plus probable

                # appliquer l’action : on ignore reward et done ici
                next_state, _, _ = env.step_rl(action)
                state = next_state
        else:
            action = select_action(state, q_network, n_actions=3, epsilon=epsilon_eval)
            next_state, reward, done = env.step_rl(action)
            state = next_state
            # if done:
            #     state = env.reset_rl()
            #     reset_timer = 180

    # ---------- RENDU ----------
    screen.fill(bg_color)

    # 1) Zone simulation
    sim_rect = pygame.Rect(0, 0, SIM_WIDTH, HEIGHT)
    pygame.draw.rect(screen, (10, 10, 30), sim_rect)

    for i, fish in enumerate(env.fish_list):
        is_agent = (i == env.agent_idx)  # normalement 0
        draw_fish(screen, fish, reward, is_agent, show_vision_cone=show_vision_cone)

    # 2) Zone UI
    ui_rect = pygame.Rect(SIM_WIDTH, 0, UI_WIDTH, HEIGHT)
    pygame.draw.rect(screen, (15, 15, 35), ui_rect)

    show_text("PARAMÈTRES (sliders):", (SIM_WIDTH + 20, 10), screen, font, font_color)

    for s in sliders.values():
        s.draw(screen)

    # Infos générales
    show_text(f"ESC: quitter", (10, HEIGHT - 20), screen, font, font_color)
    show_text(f"SPACE: pause", (10, HEIGHT - 40), screen, font, font_color)
    show_text(f"R: reset env", (10, HEIGHT - 60), screen, font, font_color)
    show_text(f"UP/DOWN: speed = {speed}", (10, HEIGHT - 80), screen, font, font_color)
    show_text(f"Poissons: {n_fish}", (10, HEIGHT - 100), screen, font, font_color)
    
    mode_text = "Mode: Supervisé (M pour changer)" if supervised else "Mode: RL (DQN) (M pour changer)"
    show_text(mode_text, (SIM_WIDTH + 20, HEIGHT - 80), screen, font, font_color)

    show_text(
        f"C: cône de vision ({'ON' if show_vision_cone else 'OFF'})",
        (SIM_WIDTH + 20, HEIGHT - 40),
        screen, font, font_color
    )

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
