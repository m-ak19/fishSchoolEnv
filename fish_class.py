import numpy as np
import matplotlib.pyplot as plt
import math
import random
from typing import List
from utils import shortest_angle_diff




class Position():
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance(self, other: "Position") -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def add(self, dx: float, dy: float):
        self.x += dx
        self.y += dy

    def get_angle(self, other: "Position") -> float:
        delta_x = other.x - self.x
        delta_y = other.y - self.y
        return math.atan2(delta_y, delta_x)

    def copy(self) -> "Position":
        return Position(self.x, self.y)


class Direction():
    def __init__(self, angle: float):
        self.angle = angle

    def rotate(self, delta: float):
        self.angle += delta

    def to_vector(self, speed: float) -> (float, float):
        dx = math.cos(self.angle) * speed
        dy = math.sin(self.angle) * speed
        return dx, dy

    def copy(self) -> "Direction":
        return Direction(self.angle)

class BoidsParams():
    def __init__(self):
        self.w_cohesion = 0.03
        self.w_alignment = 0.07
        self.w_separation = 0.2
        self.vision_range = 90
        self.min_distance = 12

class Poisson():
    """
    params_set pour plusieurs bancs séparés:
        - speed = 2.5
        - vision_range = 50
        - vision_angle = math.pi * 1.25
        - alignment_range = 40
        - max_turn = 0.3
        - min_distance = 15
        - w_cohesion = 0.02
        - w_alignment = 0.06
        - w_separation = 0.12
        
    params_set pour banc unique soudé:
        - speed = 2.5
        - vision_range = 90
        - vision_angle = math.pi * 1.25
        - alignment_range = 80
        - max_turn = 0.3
        - min_distance = 12
        - w_cohesion = 0.03
        - w_alignment = 0.07
        - w_separation = 0.12
    """
    def __init__(self,
                    pos: Position,
                    dir: Direction,
                    speed: float = 2.5,
                    params = BoidsParams(),
                    vision_angle: float = math.pi * 1.25,
                    alignment_range: float = 80,
                    max_turn: float = 0.3,

                ):
        self.pos = pos
        self.dir = dir
        self.speed = speed
        self.params = params
        self.vision_angle = vision_angle
        self.alignment_range = alignment_range
        self.max_turn = max_turn
        
    def is_in_vision_cone(self, other_fish: "Poisson") -> bool:
        distance = self.pos.distance(other_fish.pos)
        # Condition 1: Distance (Déjà implémentée)
        if distance > self.params.vision_range:
            return False

        # Condition 2: Angle (Nouveau)
        angle_to_neighbor = self.pos.get_angle(other_fish.pos)

        # Utiliser shortest_angle_diff (que vous utilisez déjà pour la rotation)
        angle_diff = shortest_angle_diff(angle_to_neighbor, self.dir.angle)

        # Si la différence d'angle est plus petite que la moitié de l'angle de vision,
        # alors le poisson est dans le cône.
        if abs(angle_diff) < self.vision_angle / 2.0:
            return True

        return False

    def cohesion(self, fish_list: List["Poisson"]):
        mean = Position(0, 0)
        fish_count = 0
        for fish in fish_list:
            if fish is self:
                continue
            distance = self.pos.distance(fish.pos)
            if distance <= self.params.min_distance:
                continue
            if self.is_in_vision_cone(fish):
                    mean.x += fish.pos.x
                    mean.y += fish.pos.y
                    fish_count += 1
        if fish_count != 0:
            mean.x /= fish_count
            mean.y /= fish_count
            target_angle = self.pos.get_angle(mean)
            raw_turn = shortest_angle_diff(target_angle, self.dir.angle)
            angle_to_turn = max(-self.max_turn, min(self.max_turn, raw_turn))
            self.dir.rotate(self.params.w_cohesion * angle_to_turn)

    def alignment(self, fish_list: List["Poisson"]):
        sum_cos = 0
        sum_sin = 0
        fish_count = 0
        
        # 1. Parcourir les poissons voisins
        for fish in fish_list:
            if fish is self:
                continue

            distance = self.pos.distance(fish.pos)

            if distance < self.params.min_distance:
                continue

            if distance < self.alignment_range and self.is_in_vision_cone(fish):
                
                # 3. Calcul de la moyenne vectorielle (sum_cos et sum_sin)
                sum_cos += math.cos(fish.dir.angle)
                sum_sin += math.sin(fish.dir.angle)
                fish_count += 1
                
        # 4. Appliquer la correction de direction
        if fish_count != 0:
            # Calcul de l'angle moyen à partir de la moyenne des vecteurs
            avg_cos = sum_cos / fish_count
            avg_sin = sum_sin / fish_count
            mean_angle = math.atan2(avg_sin, avg_cos)
            
            # Utiliser la différence d'angle la plus courte pour la rotation
            max_turn = 0.3  # en radians
            raw_turn = shortest_angle_diff(mean_angle, self.dir.angle)
            angle_to_turn = max(-max_turn, min(max_turn, raw_turn))            
            # Appliquer la rotation pondérée
            self.dir.rotate(self.params.w_alignment * angle_to_turn)

    def separation(self, fish_list: List["Poisson"]):
        separation_vector = (0.0, 0.0)
        fish_count = 0
        for fish in fish_list:
            if fish is self:
                continue
            distance = self.pos.distance(fish.pos)
            
            if distance > 0 and distance < self.params.min_distance:
                # 1. Calculer l'angle vers le poisson voisin
                angle_to_neighbor = self.pos.get_angle(fish.pos)
                
                # 2. Calculer une magnitude de force inversement proportionnelle à la distance
                # Plus on est proche, plus la force (magnitude) est grande.
                magnitude = 1.0 - (distance / self.params.min_distance)

                dx_flee = math.cos(angle_to_neighbor + math.pi) * magnitude
                dy_flee = math.sin(angle_to_neighbor + math.pi) * magnitude
                separation_vector = (separation_vector[0] + dx_flee, separation_vector[1] + dy_flee)
                fish_count += 1
                
        if fish_count != 0:
            # Calculer la direction de fuite moyenne à partir du vecteur résultant
            # Vous ne divisez pas par fish_count ici car le vecteur incorpore déjà la force.
            
            flee_angle = math.atan2(separation_vector[1], separation_vector[0])
            max_turn = 0.3
            raw_turn = shortest_angle_diff(flee_angle, self.dir.angle)
            angle_to_turn = max(-max_turn, min(max_turn, raw_turn))            
            self.dir.rotate(self.params.w_separation * angle_to_turn)

    def clamp_position(self, width: int, height: int):
        # Barrière dure : le poisson ne peut pas sortir du rectangle
        if self.pos.x < 0:
            self.pos.x = 0
        elif self.pos.x > width:
            self.pos.x = width

        if self.pos.y < 0:
            self.pos.y = 0
        elif self.pos.y > height:
            self.pos.y = height
            
    def wrap_position(self, width: int, height: int):
        if self.pos.x < 0:
            self.pos.x += width
        elif self.pos.x > width:
            self.pos.x -= width

        if self.pos.y < 0:
            self.pos.y += height
        elif self.pos.y > height:
            self.pos.y -= height

    def border_management(self, width: int = 800, height: int = 600, margin: int = 150, k: float = 0.15): ## BUG DONC ON UTILISE PAS
        dx = dy = 0
        d_left = self.pos.x
        d_right = width - self.pos.x
        d_top = self.pos.y
        d_bottom = height - self.pos.y

        if d_left <= margin: dx += 1 - d_left / margin
        if d_right <= margin: dx -= 1 - d_right / margin
        if d_top <= margin: dy += 1 - d_top / margin
        if d_bottom <= margin: dy -= 1 - d_bottom / margin

        if dx != 0 or dy != 0:
            correction_angle = math.atan2(dy, dx)
            self.dir.rotate(k * (correction_angle - self.dir.angle))

    def apply_random_fluctuation(self, max_turn_deg: float = 0.5):
        """Ajoute une petite rotation aléatoire pour maintenir le désordre."""
        # Convertir le degré en radian
        max_turn_rad = math.radians(max_turn_deg)

        # Générer une rotation aléatoire entre -max_turn_rad et +max_turn_rad
        random_turn = random.uniform(-max_turn_rad, max_turn_rad)
        self.dir.rotate(random_turn)
    
    def update(self, fish_list: List["Poisson"], width: int, height: int, is_agent: bool = False):

        if not is_agent:
            self.cohesion(fish_list)
            self.alignment(fish_list)
            self.separation(fish_list)
            self.apply_random_fluctuation()  
        else:
            pass

        dx, dy = self.dir.to_vector(self.speed)
        self.pos.add(dx, dy)

        self.wrap_position(width, height)

class fishEnv:
    def __init__(self, width, height, n_fish: int = 30):
        self.width = width
        self.height = height
        self.n_fish = n_fish
        self.agent_idx = 0
        self.params = BoidsParams()
        self.timestep = 0
        self.max_steps = 2000
        self.fish_list: List["Poisson"] = []
        self._init_fishes()

    def _init_fishes(self):
        self.fish_list = []
        for _ in range(self.n_fish):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            dir = Direction(random.uniform(-math.pi, math.pi))
            self.fish_list.append(
                Poisson(
                    Position(x, y),
                    dir,
                    params=self.params
                ))

    def reset(self):
        self._init_fishes()

    def step(self):
        for fish in self.fish_list:
            fish.update(self.fish_list, self.width, self.height)

    def reset_rl(self):
        self._init_fishes()
        self.agent_idx = 0
        self.timestep = 0
        
        state = self.compute_state()
        return state

    def step_rl(self, action):
        agent = self.fish_list[self.agent_idx]
        
        delta = math.radians(15)
        
        if(action == 1): # tourner à gauche
            agent.dir.rotate(delta)
        if(action == 2): # tourner à droite
            agent.dir.rotate(-delta)
            
        for i, fish in enumerate(self.fish_list):
            is_agent = (i == self.agent_idx)
            fish.update(self.fish_list, self.width, self.height, is_agent=is_agent)
        self.timestep += 1
        reward, dist_center_norm = self.compute_reward()
        next_state = self.compute_state()

        timeout = (self.timestep >= self.max_steps)
        failure = (dist_center_norm > 0.8)

        done = timeout or failure

        info = {
            "timeout": timeout,
            "failure": failure,
            "dist_center_norm": dist_center_norm,
        }

        return next_state, reward, done, info



    def compute_state(self):
        agent = self.fish_list[self.agent_idx]
        neighbors = []
        Nmax = 10

        # --- Variables d'état (initialisation) ---
        neighbors_count_norm = 0.0
        cos_center = 0.0
        sin_center = 0.0
        cos_align = 0.0
        sin_align = 0.0
        avg_dist_norm = 1.0
        danger = 0.0

        mean_x = 0.0
        mean_y = 0.0
        avg_cos = 0.0
        avg_sin = 0.0
        mean_dist = 0.0

        for fish in self.fish_list:
            if fish is agent:
                continue
            if agent.is_in_vision_cone(fish):
                neighbors.append(fish)

        if len(neighbors) == 0:
            center_x = 0.0
            center_y = 0.0
            sum_cos = 0.0
            sum_sin = 0.0

            for fish in self.fish_list:
                center_x += fish.pos.x
                center_y += fish.pos.y
                sum_cos += math.cos(fish.dir.angle)
                sum_sin += math.sin(fish.dir.angle)

            n_total = len(self.fish_list)
            center_x /= n_total
            center_y /= n_total
            center = Position(center_x, center_y)

            angle_center = agent.pos.get_angle(center)
            delta_center = shortest_angle_diff(angle_center, agent.dir.angle)
            cos_center = math.cos(delta_center)
            sin_center = math.sin(delta_center)

            sum_cos /= n_total
            sum_sin /= n_total
            mean_dir = math.atan2(sum_sin, sum_cos)
            delta_align = shortest_angle_diff(mean_dir, agent.dir.angle)
            cos_align = math.cos(delta_align)
            sin_align = math.sin(delta_align)

            dist_center = agent.pos.distance(center)
            max_dist = math.hypot(self.width, self.height) / 2.0
            avg_dist_norm = min(1.0, dist_center / max_dist)

            neighbors_count_norm = 0.0
            danger = 0.0 

            state = [
                neighbors_count_norm,
                cos_center, sin_center,
                cos_align, sin_align,
                avg_dist_norm,
                danger
            ]
            return np.array(state, dtype=np.float32)

        else:
            n = len(neighbors)
            neighbors_count_norm = min(n, Nmax) / Nmax

            for fish in neighbors:
                mean_x += fish.pos.x
                mean_y += fish.pos.y
                avg_cos += math.cos(fish.dir.angle)
                avg_sin += math.sin(fish.dir.angle)
                neighbor_distance = agent.pos.distance(fish.pos)
                mean_dist += neighbor_distance
                if neighbor_distance < agent.params.min_distance:
                    danger = 1.0

            mean_x /= n
            mean_y /= n
            angle_center = agent.pos.get_angle(Position(mean_x, mean_y))
            delta = shortest_angle_diff(angle_center, agent.dir.angle)
            cos_center = math.cos(delta)
            sin_center = math.sin(delta)

            avg_cos /= n
            avg_sin /= n
            mean_dir = math.atan2(avg_sin, avg_cos)
            delta_align = shortest_angle_diff(mean_dir, agent.dir.angle)
            cos_align = math.cos(delta_align)
            sin_align = math.sin(delta_align)

            mean_dist /= n
            avg_dist_norm = min(1.0, mean_dist / agent.params.vision_range)

            state = [
                neighbors_count_norm,
                cos_center, sin_center,
                cos_align, sin_align,
                avg_dist_norm,
                danger
            ]
            return np.array(state, dtype=np.float32)


    def compute_reward(self):
        agent = self.fish_list[self.agent_idx]

        center_x = 0.0
        center_y = 0.0
        sum_cos = 0.0
        sum_sin = 0.0

        neighbors_count = 0
        Nmax = 10

        # poids des termes (à ajuster si besoin)
        w_dist = 1.0
        w_align = 0.5
        w_iso = 0.3

        # 1) centre global + direction moyenne + densité
        for fish in self.fish_list:
            center_x += fish.pos.x
            center_y += fish.pos.y
            sum_cos += math.cos(fish.dir.angle)
            sum_sin += math.sin(fish.dir.angle)
            if agent.is_in_vision_cone(fish) and fish is not agent:
                neighbors_count += 1

        n = len(self.fish_list)
        center_x /= n
        center_y /= n
        center = Position(center_x, center_y)

        # distance normalisée au centre du banc
        dist_center = agent.pos.distance(center)
        max_dist = math.hypot(self.width, self.height) / 2.0
        dist_center_norm = min(dist_center / max_dist, 1.0)   # 0 proche du centre, 1 très loin

        # alignement moyen du banc
        sum_cos /= n
        sum_sin /= n
        mean_dir = math.atan2(sum_sin, sum_cos)

        delta_align = abs(shortest_angle_diff(mean_dir, agent.dir.angle))
        align_norm = min(delta_align / math.pi, 1.0)          # 0 bien aligné, 1 opposé

        # isolement (peu de voisins dans le cône de vision)
        neighbors_count_norm = min(neighbors_count, Nmax) / Nmax
        isolation = 1.0 - neighbors_count_norm                # 1 très isolé, 0 entouré

        # pénalité globale
        penalty = (
            w_dist * dist_center_norm +
            w_align * align_norm +
            w_iso * isolation
        )

        # reward "positive" quand tout va bien
        reward = 1.0 - penalty

        return reward, dist_center_norm

