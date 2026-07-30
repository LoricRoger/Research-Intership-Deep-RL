# envs/cp_state_encoder.py
"""
Encodeurs obs_dict (RDDL brut) → state_idx (entier pour le serveur CP Java).

Chaque encodeur est un callable : (obs_dict: dict) -> int

Convention CrossingTraffic (portée depuis train.py du projet source) :
    state_idx = robot_pos * nObsConfigs + obs_bits

    robot_pos    = xi * nY + yj          (column-major, 0-based)
    obs_bits     = nX*(nY-2) bits, rangées intérieures r=0..nY-3 (yj=1..nY-2)
                   bit index pour (xi, r) = r*nX + xi
    nObsConfigs  = 2^(nX*(nY-2))
    DEAD         = nX * nY * nObsConfigs  (aucun robot-at à True)
    GOAL         = DEAD + 1               (robot à la position goal_robot_pos)

    goal_robot_pos = (nX-1)*nY + (nY-1)  (coin column-major, 0-based)

Format des clés RDDL (1-based, triple underscore) :
    robot-at___x{xi+1}__y{yj+1}
    obstacle-at___x{xi+1}__y{yj+1}

Exemples :
    3×3 →     74 états  (nObsConfigs=8,   DEAD=72, GOAL=73)
    4×4 →   4098 états  (nObsConfigs=256, DEAD=4096, GOAL=4097)
    5×5 → 819202 états
"""
from __future__ import annotations

from typing import Callable, Dict


StateEncoder = Callable[[Dict], int]


# ------------------------------------------------------------------
# CrossingTraffic
# ------------------------------------------------------------------

def make_crossing_traffic_encoder(nX: int, nY: int) -> StateEncoder:
    """
    Retourne un encodeur obs_dict → state_idx pour CrossingTraffic.

    Portage exact de _build_state_encoder() + make_obs_to_state_idx()
    depuis le projet CrossingTraffic source (train.py).
    """
    n_inner      = nY - 2
    nObsConfigs  = 1 << (nX * n_inner)
    dead_state   = nX * nY * nObsConfigs
    goal_state   = dead_state + 1

    # Position goal en column-major 0-based : (xi=nX-1, yj=nY-1)
    goal_robot_pos = (nX - 1) * nY + (nY - 1)

    # Clés robot-at dans l'ordre column-major (index i = xi*nY + yj)
    robot_keys = [
        f"robot-at___x{xi + 1}__y{yj + 1}"
        for xi in range(nX)
        for yj in range(nY)
    ]

    # Clés obstacle-at pour les rangées intérieures uniquement
    # r = rangée intérieure 0-based, bit index = r*nX + xi
    obs_keys = [
        f"obstacle-at___x{xi + 1}__y{r + 2}"   # yj 1-based = r+2
        for r in range(n_inner)
        for xi in range(nX)
    ]

    def encode(obs_dict: Dict) -> int:
        # 1. Trouver la position du robot (première clé True en order column-major)
        robot_pos = next(
            (i for i, k in enumerate(robot_keys) if obs_dict.get(k)),
            -1,
        )

        # 2. Aucun robot-at → collision → DEAD
        if robot_pos == -1:
            return dead_state

        # 3. Position goal → GOAL
        if robot_pos == goal_robot_pos:
            return goal_state

        # 4. Bits obstacles (rangées intérieures)
        obs_bits = sum(
            1 << b
            for b, k in enumerate(obs_keys)
            if obs_dict.get(k)
        )

        return robot_pos * nObsConfigs + obs_bits

    return encode


# ------------------------------------------------------------------
# SysAdmin
# ------------------------------------------------------------------

def make_sysadmin_encoder(n_computers: int) -> StateEncoder:
    """
    Retourne un encodeur action_idx → reboot_idx pour SysAdmin.

    SB3 action space: Discrete(N+1)
        0       → noop        → reboot_idx = -1
        1..N    → reboot c_i  → reboot_idx = i-1  (0-based)

    Le serveur SysAdmin attend reboot_idx dans {-1, 0..N-1}.
    Cet "encodeur" ignore obs_dict et encode l'action directement.
    """
    def encode(obs_dict: Dict) -> int:
        # Not used for SysAdmin — action encoding is handled by SysAdminCPShaper
        raise NotImplementedError("Use SysAdminCPShaper.action_to_reboot_idx() instead.")
    return encode


def action_to_reboot_idx(action_idx: int) -> int:
    """Converts a SB3 Discrete action index to a SysAdmin reboot_idx.

    action_idx=0 (noop)  → -1
    action_idx=k (k>=1)  → k-1  (0-based computer index)
    """
    return -1 if action_idx == 0 else action_idx - 1


# ------------------------------------------------------------------
# GameOfLife
# ------------------------------------------------------------------

def make_gameoflife_encoder(n_cells: int) -> StateEncoder:
    """
    Dummy encoder for GameOfLife — action encoding is handled by GameOfLifeCPShaper.
    """
    def encode(obs_dict: Dict) -> int:
        raise NotImplementedError("Use GameOfLifeCPShaper directly.")
    return encode


def action_to_set_idx(action_idx: int) -> int:
    """Converts a SB3 Discrete action index to a GameOfLife set_idx.

    action_idx=0 (noop)  → -1
    action_idx=k (k>=1)  → k-1  (0-based cell index)
    """
    return -1 if action_idx == 0 else action_idx - 1


# ------------------------------------------------------------------
# TriangleTireworld
# ------------------------------------------------------------------

def make_triangletireworld_encoder(loc_names: list) -> StateEncoder:
    """
    Retourne un encodeur obs_dict → state_idx pour TriangleTireworld.

    s = 4*p + 2*tau + h
      p   = index 0-based de la location courante (ordre alphabétique)
      tau = 1 si not-flattire, 0 si crevé
      h   = 1 si hasspare, 0 sinon

    loc_names : liste des noms de locations triés alphabétiquement,
                dans le même ordre que le JSON (index 0..N-1).
    """
    veh_keys = [f"vehicle-at___{loc}" for loc in loc_names]

    def encode(obs_dict: Dict) -> int:
        p = next(
            (i for i, k in enumerate(veh_keys) if obs_dict.get(k)),
            0,  # fallback sûr — ne devrait pas arriver
        )
        tau = 1 if obs_dict.get("not-flattire", False) else 0
        h   = 1 if obs_dict.get("hasspare",     False) else 0
        return 4 * p + 2 * tau + h

    return encode


# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------

def make_navigation_encoder(nX: int, nY: int,
                             xpos_sorted: list, ypos_sorted: list,
                             dead_state_idx: int) -> StateEncoder:
    """
    Retourne un encodeur obs_dict → state_idx pour Navigation MDP.

    state_idx = xi * nY + yj  (column-major, 0-based)
    Retourne dead_state_idx si aucun robot-at n'est True.

    xpos_sorted, ypos_sorted : listes triées de labels de position (ex. "x1", "y2").
    """
    def encode(obs_dict: Dict) -> int:
        for xi, xl in enumerate(xpos_sorted):
            for yj, yl in enumerate(ypos_sorted):
                if obs_dict.get(f"robot-at___{xl}__{yl}", False):
                    return xi * nY + yj
        return dead_state_idx

    return encode


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

def get_encoder(domain: str, **kwargs) -> StateEncoder:
    """
    Point d'entrée unique.

    Usage :
        encoder = get_encoder("CrossingTraffic_MDP_ippc2011", nX=3, nY=3)
        state_idx = encoder(obs_dict)

    Normalisation : "CrossingTraffic_MDP_ippc2011" → "crossingtraffic"

    Pour ajouter un domaine :
        1. Écrire make_<domain>_encoder(**kwargs) -> StateEncoder
        2. L'enregistrer dans _REGISTRY ci-dessous
    """
    domain_key = domain.lower().split("_")[0]

    _REGISTRY: Dict[str, Callable] = {
        "crossingtraffic": lambda **kw: make_crossing_traffic_encoder(
            kw["nX"], kw["nY"]
        ),
        "sysadmin": lambda **kw: make_sysadmin_encoder(kw["n_computers"]),
        "gameoflife": lambda **kw: make_gameoflife_encoder(kw["n_cells"]),
        "triangletireworld": lambda **kw: make_triangletireworld_encoder(
            kw["loc_names"]
        ),
        "navigation": lambda **kw: make_navigation_encoder(
            kw["nX"], kw["nY"], kw["xpos_sorted"], kw["ypos_sorted"],
            kw["dead_state_idx"]
        ),
    }

    if domain_key not in _REGISTRY:
        raise ValueError(
            f"Pas d'encodeur CP pour le domaine '{domain}' "
            f"(clé normalisée : '{domain_key}'). "
            f"Domaines disponibles : {list(_REGISTRY)}"
        )

    return _REGISTRY[domain_key](**kwargs)