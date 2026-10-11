import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

VERSION = "v9.1 (2026-09-23) | 플레이어 반경 조절(--player-radius) + 반응형 NPC"


_FALLBACK_LAYOUT = [
    "################",
    "#..............#",
    "#.......####.G.#",
    "#.......####...#",
    "#..............#",
    "#..............#",
    "#..............#",
    "#..............#",
    "#..............#",
    "#.S............#",
    "#..............#",
    "################",
]

try:
    from random_map_gen import generate_random_layout
    HAS_RANDOM_MAPS = True
except ImportError:
    HAS_RANDOM_MAPS = False

MAPS = {"_fallback": _FALLBACK_LAYOUT}

try:
    from maps import MAPS as _EXTRA_MAPS
    for _k, _v in _EXTRA_MAPS.items():
        if _k == "_fallback":
            continue          # 내장 기준 맵은 덮어쓰지 않습니다
        MAPS[_k] = _v
except ImportError:
    print("[안내] maps.py 를 찾지 못했습니다. 내장 _fallback 맵만 사용합니다.")

DEFAULT_MAP = "_fallback"
TURN_RIGHT = 0       # 제자리 우회전 (각도 -30도, 화면상 시계방향)
TURN_LEFT = 1        # 제자리 좌회전 (각도 +30도, 화면상 반시계방향)
MOVE_FORWARD = 2     # 직진
WAIT = 3             # 대기
MOVE_FWD_RIGHT = 4   # 우회전 + 전진 (곡선 주행)
MOVE_FWD_LEFT = 5    # 좌회전 + 전진 (곡선 주행)
N_ACTIONS = 6

# 정규화 상수 (★ 맵 크기와 무관한 고정 물리 단위 ★)
RAY_ANGLES_DEG = (-90, -60, -30, 0, 30, 60, 90)  # 진행 방향 기준
RAY_MAX = 5.0         # 레이캐스트 최대 사거리 (월드 단위)
RAY_STEP = 0.15
SENSE_RADIUS = 6.0    # 장애물 감지 반경
MAX_OBS_SPEED = 0.35  # 장애물 속도 정규화 기준
GOAL_DIST_MAX = 12.0  # 목표까지 거리 정규화 기준 (클립)

MAX_OBSTACLES = 3     # 관측 슬롯 수
OBS_DIM = 3 + 2 + len(RAY_ANGLES_DEG) + MAX_OBSTACLES * 5 + N_ACTIONS + 1  # = 34


@dataclass
class EnvConfig:
    width: int = 16
    height: int = 12
    heading_step_deg: int = 30
    move_step: float = 0.25
    max_steps: int = 500
    goal_radius: float = 0.6
    agent_radius: float = 0.22
    move_substeps: int = 10

    # Rewards
    goal_reward: float = 20.0
    time_penalty: float = -0.01
    wall_collision_penalty: float = -0.5
    obstacle_collision_penalty: float = -10.0

    progress_weight: float = 1.0      # potential-based shaping 계수 (대칭)
    align_weight: float = 0.05        # BFS 방향 정렬

    rotation_penalty: float = -0.08       # 회전 1회당
    turn_reversal_penalty: float = -0.06  # 좌->우 / 우->좌 전환 시 추가 (지그재그 억제)

    stagnation_patience: int = 180
    stagnation_penalty: float = -2.0

    # 개인 공간(personal space): 가까워지는 것 자체에 페널티
    personal_space: float = 1.5
    proximity_weight: float = 0.0

    # 곡선 주행(회전+전진 동시)의 회전 비용 배율
    # 1.0보다 작으면 '멈춰서 돌기'보다 '걸으면서 틀기'를 선호
    curved_move_penalty_scale: float = 0.5

    # 관측에서 장애물을 거리순으로 정렬할지 여부.
    # True  = 순서 불변성 확보. 두 장애물의 거리 순위가 바뀌면 입력 슬롯이 뒤바뀌어 관측이 불연속적.
    # False = 슬롯 고정. 특정 장애물을 시간에 걸쳐 추적하기 쉬움.
    sort_obstacles_by_distance: bool = True

    # 활성 장애물 수
    # 맵에 따라 통로가 막힐 수 있음
    num_obstacles: int = 3

    # 플레이어
    #   ""        플레이어 없음 (기존 동작)
    #   "goal"    목적지형 : 자기 목표를 향해 이동 (마주 오는 보행자)
    #   "chase"   추적형 : NPC 쪽으로 접근 (가장 어려움)
    #   "random"  무작위 전환형 : 주기적으로 방향 변경 (예측 불가능성)
    #   "cross"   교차형 : NPC 의 목표 직선을 가로지름 (마주치는 보행자)
    #   "mix"     무작위형: 에피소드마다 무작위 선택
    player_policy: str = ""
    player_speed: float = 0.22        # NPC 이동속도(0.25)보다 약간 느리게
    player_turn_period: int = 18      # random 형이 방향을 바꾸는 주기(스텝)
    obstacle_radius: float = 0.40
    player_radius: float = 0.40

    # 랜덤 맵 학습
    # True 면 reset() 마다 random_map_gen 으로 새 맵을 생성해 학습
    # 맵 크기 불변 -> 맵이 달라져도 같은 정책
    # 통로 폭이 좁은 경우(winding_corridor / sparse_maze 스타일) 장애물을 자동으로 끔. (물리적으로 통과 불가하므로) 
    random_maps: bool = False
    random_map_styles: tuple = ()   # 빈 튜플이면 4종 전부 사용

    # 랜덤화 (일반화용, 기본 off로 두어 베이스라인 비교 가능)
    randomize_obstacles: bool = False
    min_spawn_distance: float = 3.0   # 랜덤 스폰 시 시작점에서 최소 이 거리 밖에 배치
                                      # (반응할 시간 없이 실패하는 상황 방지)


# 맵 검증
def validate_layout(name, layout):
    if not layout:
        raise ValueError(f"맵 '{name}'이 비어 있습니다.")
    widths = {len(r) for r in layout}
    if len(widths) != 1:
        raise ValueError(f"맵 '{name}'의 행 길이가 제각각입니다: {sorted(widths)}")
    flat = "".join(layout)
    if flat.count("S") != 1:
        raise ValueError(f"맵 '{name}'의 'S'가 {flat.count('S')}개입니다. 1개여야 합니다.")
    if flat.count("G") != 1:
        raise ValueError(f"맵 '{name}'의 'G'가 {flat.count('G')}개입니다. 1개여야 합니다.")
    unknown = set(flat) - set("#.SG")
    if unknown:
        raise ValueError(f"맵 '{name}'에 알 수 없는 문자: {sorted(unknown)}")
    return len(layout), len(layout[0])


# Environment
class GymHumanLikeNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig = None, map_name: str = DEFAULT_MAP):
        super().__init__()

        if map_name not in MAPS:
            raise KeyError(f"'{map_name}' 맵 없음. 사용 가능: {sorted(MAPS.keys())}")

        # cfg 를 먼저 확보 (랜덤 맵 여부를 알아야 하므로)
        self.cfg = config if config is not None else EnvConfig()
        if self.cfg.random_maps and not HAS_RANDOM_MAPS:
            raise ImportError("random_maps=True 인데 random_map_gen.py 를 찾을 수 없습니다. "
                              "visual_gym.py 와 같은 폴더에 넣어주세요.")

        self.map_name = map_name
        self.n_headings = int(360 / self.cfg.heading_step_deg)
        self.action_space = Discrete(N_ACTIONS)
        self.observation_space = Box(low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32)

        # 장애물 배열 (관측 슬롯 수는 항상 MAX_OBSTACLES 로 고정)
        self.num_obstacles = MAX_OBSTACLES
        self.goal_guard_angle = 0.0
        self._wanderer_frac = [(0.625, 0.25), (0.25, 0.667), (0.5, 0.5)]
        self._wanderer_vel = np.array([[-0.10, 0.20], [0.18, -0.12], [0.14, 0.16]],
                                      dtype=np.float32)
        self.obs_pos = np.zeros((MAX_OBSTACLES, 2), dtype=np.float32)
        self.obs_vel = np.zeros((MAX_OBSTACLES, 2), dtype=np.float32)
        self.obs_prev_pos = np.zeros((MAX_OBSTACLES, 2), dtype=np.float32)

        # 재생(playback) 모드 - convert_pedestrian.py 가 사용.

        self.playback = False
        self.scripted_obs_fn = None   # callable(step_idx) -> (K,2) 배열 또는 None

        # 첫 레이아웃 적용
        if self.cfg.random_maps:
            layout, style, _ = self._new_random_layout()
            self._apply_layout(layout, style=style)
        else:
            layout = MAPS[map_name]
            map_h, map_w = validate_layout(map_name, layout)
            if config is not None and not self.cfg.random_maps:
                assert config.width == map_w and config.height == map_h, (
                    f"EnvConfig({config.width}x{config.height}) != 맵 '{map_name}' ({map_w}x{map_h})"
                )
            self._apply_layout(layout, style=None)

        self.agent_pos = self.start.copy()
        self.heading_idx = 0
        self.last_action = MOVE_FORWARD
        self.last_turn_dir = 0
        self.reset()

    # --------------------------------------------------
    def _apply_layout(self, layout, style=None):

        # 기본 형식 검사 (랜덤 맵도 동일 규격이어야 함)
        widths = {len(r) for r in layout}
        assert len(widths) == 1, f"행 길이 불일치: {sorted(widths)}"
        flat = "".join(layout)
        assert flat.count("S") == 1 and flat.count("G") == 1, "S/G 는 각각 1개여야 함"

        map_h, map_w = len(layout), len(layout[0])
        self.cfg.width, self.cfg.height = map_w, map_h
        self.cfg.max_steps = self._suggest_max_steps(map_w, map_h, self.cfg.move_step)
        self.map_style = style

        self.grid, self.start, self.goal = self._build_map(layout)
        self.grid_goal_distance = self._build_grid_goal_distance()
        if self.grid_goal_distance[int(self.start[1]), int(self.start[0])] < 0:
            raise ValueError("시작점에서 목표에 도달할 수 없는 레이아웃입니다.")

        # 활성 장애물 수: 통로 폭 1칸짜리 스타일은 이동 장애물 통과가 물리적으로 불가능 -> 자동으로 끔.
        self.n_active = max(0, min(self.cfg.num_obstacles, MAX_OBSTACLES))
        if style in ("winding_corridor", "sparse_maze"):
            self.n_active = 0

        self.has_player = bool(self.cfg.player_policy) and self.n_active > 0
        self.guard_radius = self._fit_guard_radius()
        self.has_guard = (self.guard_radius is not None
                          and self.n_active > 0 and not self.has_player)
        if self.guard_radius is None:
            self.guard_radius = 1.2

        self.bounce_min = np.array([1.5, 1.5], dtype=np.float32)
        self.bounce_max = np.array([self.cfg.width - 1.5, self.cfg.height - 1.5],
                                   dtype=np.float32)
        for k in range(2):
            if self.bounce_min[k] >= self.bounce_max[k]:
                mid = (self.cfg.width if k == 0 else self.cfg.height) / 2.0
                self.bounce_min[k] = self.bounce_max[k] = mid

    def _new_random_layout(self):
        #np_random 에서 시드를 뽑아 재현 가능한 랜덤 맵을 생성

        for _ in range(50):
            gen_seed = int(self.np_random.integers(0, 2 ** 31 - 1))
            style = None
            if self.cfg.random_map_styles:
                style = str(self.np_random.choice(list(self.cfg.random_map_styles)))
            layout, used_style, path_len = generate_random_layout(seed=gen_seed, style=style)

            flat = "".join(layout)
            if flat.count("S") != 1 or flat.count("G") != 1:
                continue                      # 고립 구멍에 S/G 겹침 -> 재시도
            w, h = len(layout[0]), len(layout)
            if path_len < max(6, (w + h) // 4):
                continue                      # 경로가 너무 짧은 맵 -> 재시도
            return layout, used_style, path_len
        raise RuntimeError("랜덤 맵 생성 50회 연속 실패 (random_map_gen 확인 필요)")

    # --------------------------------------------------
    def _fit_guard_radius(self):
        # 좁은 복도처럼 자리가 없는 맵에서는 None 반환, 일반 배회 장애물로 대체.
        for r in (1.2, 1.1, 1.0, 0.9):
            if r <= self.cfg.goal_radius:
                break
            ok = True
            for k in range(72):
                a = k * 2 * math.pi / 72
                x = float(self.goal[0] + r * math.cos(a))
                y = float(self.goal[1] + r * math.sin(a))
                ix, iy = int(x), int(y)
                if not (0 <= ix < self.cfg.width and 0 <= iy < self.cfg.height) \
                        or self.grid[iy, ix] == 1:
                    ok = False
                    break
            if ok:
                return r
        return None

    def _reset_player(self):
        # 플레이어 초기화. 슬롯 0 사용.
        self.player_delay = 0
        self.player_kind = self.cfg.player_policy
        if self.player_kind == "mix":
            self.player_kind = str(self.np_random.choice(
                ["goal", "chase", "random", "cross"]))
        self.player_timer = 0

        free = np.argwhere(self.grid == 0)
        # NPC 와 너무 가깝지 않은 곳에서 시작
        cand = free[(free[:, 1] + 0.5 - self.start[0]) ** 2
                    + (free[:, 0] + 0.5 - self.start[1]) ** 2 >= 16.0]
        if len(cand) == 0:
            cand = free
        cy, cx = cand[self.np_random.integers(len(cand))]
        self.obs_pos[0] = [cx + 0.5, cy + 0.5]

        # 목적지형이 향할 지점 (NPC 시작점 근처로 잡아 마주 오게)
        far = free[(free[:, 1] + 0.5 - self.obs_pos[0][0]) ** 2
                   + (free[:, 0] + 0.5 - self.obs_pos[0][1]) ** 2 >= 25.0]
        if len(far) == 0:
            far = free
        gy, gx = far[self.np_random.integers(len(far))]
        self.player_goal = np.array([gx + 0.5, gy + 0.5], dtype=np.float32)

        if self.player_kind == "cross":
            self._setup_cross()

        ang = float(self.np_random.uniform(0, 2 * math.pi))
        self.obs_vel[0] = [self.cfg.player_speed * math.cos(ang),
                           self.cfg.player_speed * math.sin(ang)]

    def _setup_cross(self):
        d = self.goal - self.start
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return
        u = d / n                                   # 진행 방향
        perp = np.array([-u[1], u[0]], dtype=np.float32)   # 수직 방향
        mid = self.start + u * (n * float(self.np_random.uniform(0.40, 0.65)))

        free = np.argwhere(self.grid == 0)
        side = 1.0 if self.np_random.random() < 0.5 else -1.0

        def nearest_free(p):
            d2 = (free[:, 1] + 0.5 - p[0]) ** 2 + (free[:, 0] + 0.5 - p[1]) ** 2
            cy, cx = free[int(np.argmin(d2))]
            return np.array([cx + 0.5, cy + 0.5], dtype=np.float32)

        # 한쪽에서 출발해 반대쪽으로 건너감
        spawn = nearest_free(mid + perp * side * 4.0)
        self.obs_pos[0] = spawn
        self.player_goal = nearest_free(mid - perp * side * 5.0)

        # 타이밍 맞추기 (그냥 배치하면 플레이어가 먼저 지나가 버리거나 너무 늦게 와서 NPC 와 마주치지 않음.
        # NPC 가 교차점에 닿을 쯤 플레이어도 위치하도록 출발 시각 지연.
        npc_steps = float(np.linalg.norm(mid - self.start)) / max(self.cfg.move_step, 1e-6)
        pl_steps = float(np.linalg.norm(mid - spawn)) / max(self.cfg.player_speed, 1e-6)
        self.player_delay = max(0, int(npc_steps - pl_steps))

    def _move_player(self):
        pos = self.obs_pos[0]
        kind = self.player_kind
        self.player_timer += 1
        if self.player_timer <= self.player_delay:
            return                      # 교차 타이밍을 맞추기 위한 대기

        if kind == "chase":
            target = self.agent_pos
        elif kind == "cross":
            target = self.player_goal
        elif kind == "goal":
            target = self.player_goal
            if float(np.linalg.norm(target - pos)) < 1.0:
                # 목적지에 닿으면 새 목적지를 설정한 후 계속 배회
                free = np.argwhere(self.grid == 0)
                gy, gx = free[self.np_random.integers(len(free))]
                self.player_goal = np.array([gx + 0.5, gy + 0.5], dtype=np.float32)
                target = self.player_goal
        else:                                   # random
            if self.player_timer % self.cfg.player_turn_period == 0:
                ang = float(self.np_random.uniform(0, 2 * math.pi))
                self.obs_vel[0] = [self.cfg.player_speed * math.cos(ang),
                                   self.cfg.player_speed * math.sin(ang)]
            target = None

        if target is not None:
            d = target - pos
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                self.obs_vel[0] = (d / n) * self.cfg.player_speed

        # 축별로 시도해서 벽에 막힌 축 제외 (코너 끼임 방지)
        vel = self.obs_vel[0]
        moved = False
        for k in range(2):
            cand = pos.copy()
            cand[k] += vel[k]
            inside = (self.bounce_min[k] <= cand[k] <= self.bounce_max[k])
            if inside and not self._circle_hits_wall(cand, 0.35):
                pos[k] = cand[k]
                moved = True
        if not moved and kind == "random":
            ang = float(self.np_random.uniform(0, 2 * math.pi))
            self.obs_vel[0] = [self.cfg.player_speed * math.cos(ang),
                               self.cfg.player_speed * math.sin(ang)]

    def _move_wanderer(self, i):
        pos, vel = self.obs_pos[i], self.obs_vel[i]
        for k in range(2):
            cand = pos.copy()
            cand[k] += vel[k]
            out_of_bounds = cand[k] > self.bounce_max[k] or cand[k] < self.bounce_min[k]
            if out_of_bounds or self._circle_hits_wall(cand, 0.35):
                vel[k] *= -1.0
            else:
                pos[k] = cand[k]

    @staticmethod
    def _suggest_max_steps(w, h, move_step):
        return max(500, int(math.hypot(w, h) * 4 / move_step))

    def _build_map(self, layout):
        height, width = len(layout), len(layout[0])
        grid = np.zeros((height, width), dtype=np.int32)
        start = goal = None
        for r, row in enumerate(reversed(layout)):
            for c, ch in enumerate(row):
                grid[r, c] = 1 if ch == "#" else 0
                if ch == "S":
                    start = np.array([c + 0.5, r + 0.5], dtype=np.float32)
                elif ch == "G":
                    goal = np.array([c + 0.5, r + 0.5], dtype=np.float32)
        return grid, start, goal

    def _build_grid_goal_distance(self):
        dist = np.full((self.cfg.height, self.cfg.width), -1, dtype=np.int32)
        gx, gy = int(self.goal[0]), int(self.goal[1])
        dist[gy, gx] = 0
        q = deque([(gx, gy)])
        while q:
            cx, cy = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.cfg.width and 0 <= ny < self.cfg.height:
                    if self.grid[ny, nx] == 0 and dist[ny, nx] == -1:
                        dist[ny, nx] = dist[cy, cx] + 1
                        q.append((nx, ny))
        return dist

    # 기하 유틸
    def heading_rad(self):
        return math.radians(self.heading_idx * self.cfg.heading_step_deg)

    def heading_unit_vector(self, heading_idx=None):
        idx = self.heading_idx if heading_idx is None else heading_idx
        th = math.radians(idx * self.cfg.heading_step_deg)
        return np.array([math.cos(th), math.sin(th)], dtype=np.float32)

    def _to_agent_frame(self, vec):
        th = self.heading_rad()
        c, s = math.cos(th), math.sin(th)
        return np.array([c * vec[0] + s * vec[1],
                         -s * vec[0] + c * vec[1]], dtype=np.float32)

    def _bfs_dist_at(self, pos):
        d = self.grid_goal_distance[int(pos[1]), int(pos[0])]
        return float(d) if d >= 0 else float(self.cfg.width * self.cfg.height)

    def _distance_to_goal(self, pos):
        return float(np.linalg.norm(pos - self.goal))

    def _get_ideal_direction(self, pos):
        cx, cy = int(pos[0]), int(pos[1])
        cur = self.grid_goal_distance[cy, cx]
        if cur <= 0:
            v = self.goal - pos
            return v / (np.linalg.norm(v) + 1e-8)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < self.cfg.width and 0 <= ny < self.cfg.height:
                if self.grid_goal_distance[ny, nx] == cur - 1:
                    v = np.array([nx + 0.5 - pos[0], ny + 0.5 - pos[1]], dtype=np.float32)
                    n = np.linalg.norm(v)
                    if n > 0:
                        return v / n
        v = self.goal - pos
        return v / (np.linalg.norm(v) + 1e-8)

    def _circle_hits_wall(self, pos, radius):
        min_x = max(0, int(pos[0] - radius)); max_x = min(self.cfg.width - 1, int(pos[0] + radius))
        min_y = max(0, int(pos[1] - radius)); max_y = min(self.cfg.height - 1, int(pos[1] + radius))
        for cy in range(min_y, max_y + 1):
            for cx in range(min_x, max_x + 1):
                if self.grid[cy, cx] == 1:
                    nx = min(max(pos[0], cx), cx + 1)
                    ny = min(max(pos[1], cy), cy + 1)
                    if (pos[0] - nx) ** 2 + (pos[1] - ny) ** 2 < radius ** 2:
                        return True
        return False

    def _swept_move(self, pos, heading_idx, total_distance, radius, substeps):
        d = self.heading_unit_vector(heading_idx)
        step = total_distance / substeps
        cur = pos.copy()
        for _ in range(substeps):
            cand = cur + d * step
            if self._circle_hits_wall(cand, radius):
                return cur, True
            cur = cand
        return cur, False

    def _apply_turn(self, turn_dir, while_moving=False):
        extra = 0.0
        self.heading_idx = (self.heading_idx + turn_dir) % self.n_headings
        self.n_turns += 1
        if self.last_turn_dir != 0 and turn_dir != self.last_turn_dir:
            extra += self.cfg.turn_reversal_penalty
            self.n_turn_reversals += 1
        self.last_turn_dir = turn_dir

        # 측정 전용: 전진이 끼어 있어도 방향 전환을 놓치지 않음
        self.total_turn_deg += self.cfg.heading_step_deg
        if while_moving:
            self.moving_turn_deg += self.cfg.heading_step_deg
        else:
            self.static_turn_deg += self.cfg.heading_step_deg
        if self._metric_last_dir != 0 and turn_dir != self._metric_last_dir:
            self.heading_reversals += 1
        self._metric_last_dir = turn_dir
        return extra

    def _do_forward_after_turn(self, turn_dir):
        new_heading = (self.heading_idx + turn_dir) % self.n_headings
        new_pos, blocked = self._swept_move(
            self.agent_pos, new_heading,
            self.cfg.move_step, self.cfg.agent_radius, self.cfg.move_substeps)
        if blocked:
            self.n_blocked += 1
        dist = float(np.linalg.norm(new_pos - self.agent_pos))
        self.path_length += dist
        self.agent_pos = new_pos
        return dist > 1e-9, blocked

    def _do_forward(self):
        new_pos, blocked = self._swept_move(
            self.agent_pos, self.heading_idx,
            self.cfg.move_step, self.cfg.agent_radius, self.cfg.move_substeps)
        if blocked:
            self.n_blocked += 1
        dist = float(np.linalg.norm(new_pos - self.agent_pos))
        self.path_length += dist
        self.agent_pos = new_pos
        return dist > 1e-9, blocked

    def _raycast(self, angle_rad):
        dx, dy = math.cos(angle_rad), math.sin(angle_rad)
        x0, y0 = float(self.agent_pos[0]), float(self.agent_pos[1])
        d = 0.0
        while d < RAY_MAX:
            d += RAY_STEP
            ix, iy = int(x0 + dx * d), int(y0 + dy * d)
            if not (0 <= ix < self.cfg.width and 0 <= iy < self.cfg.height):
                return d
            if self.grid[iy, ix] == 1:
                return d
        return RAY_MAX

    # 관측 (egocentric + 스케일 불변)
    def _get_obs(self):
        th = self.heading_rad()
        feats = []

        # 목표 방향/거리 (에이전트 기준)
        to_goal = self.goal - self.agent_pos
        gd = float(np.linalg.norm(to_goal))
        gb = math.atan2(float(to_goal[1]), float(to_goal[0])) - th
        feats += [math.sin(gb), math.cos(gb), min(gd, GOAL_DIST_MAX) / GOAL_DIST_MAX]

        # BFS 이상 진행 방향 (에이전트 기준)
        iv = self._get_ideal_direction(self.agent_pos)
        ib = math.atan2(float(iv[1]), float(iv[0])) - th
        feats += [math.sin(ib), math.cos(ib)]

        # 전방 부채꼴 레이캐스트
        for a_deg in RAY_ANGLES_DEG:
            feats.append(self._raycast(th + math.radians(a_deg)) / RAY_MAX)

        rel_list = []
        for i in range(self.num_obstacles):
            rel_p = self.obs_pos[i] - self.agent_pos
            rel_v = self.obs_pos[i] - self.obs_prev_pos[i]
            rel_list.append((float(np.linalg.norm(rel_p)), rel_p, rel_v))
        if self.cfg.sort_obstacles_by_distance:
            rel_list.sort(key=lambda t: t[0])

        for dist, rel_p, rel_v in rel_list:
            if dist > SENSE_RADIUS:
                feats += [0.0, 0.0, 0.0, 0.0, 0.0]
                continue
            fp = self._to_agent_frame(rel_p) / SENSE_RADIUS
            fv = self._to_agent_frame(rel_v) / MAX_OBS_SPEED
            feats += [float(np.clip(fp[0], -1, 1)), float(np.clip(fp[1], -1, 1)),
                      float(np.clip(fv[0], -1, 1)), float(np.clip(fv[1], -1, 1)), 1.0]

        # 직전 행동
        oh = [0.0] * N_ACTIONS
        oh[int(self.last_action)] = 1.0
        feats += oh

        # 직전 회전 방향 (-1 좌 / 0 없음 / +1 우)
        feats.append(float(self.last_turn_dir))

        return np.array(feats, dtype=np.float32)

    # 렌더러 호환용 별칭
    def get_discrete_state(self):
        return self._get_obs()

    # --------------------------------------------------
    def _reset_obstacles(self):
        FAR = -10000.0
        self.obs_pos[:] = FAR
        self.obs_vel[:] = 0.0

        self.goal_guard_angle = 0.0
        if self.cfg.randomize_obstacles:
            self.goal_guard_angle = float(self.np_random.uniform(0, 2 * math.pi))

        free = np.argwhere(self.grid == 0)  # (row=y, col=x)
        if self.cfg.randomize_obstacles:
            cand = free[
                (free[:, 1] + 0.5 - self.start[0]) ** 2
                + (free[:, 0] + 0.5 - self.start[1]) ** 2
                >= self.cfg.min_spawn_distance ** 2
            ]
            if len(cand) == 0:
                cand = free
        else:
            cand = free

        idx = 0
        if self.has_player:
            self._reset_player()
            idx = 1
        elif self.has_guard:
            self.obs_pos[0] = [
                self.goal[0] + self.guard_radius * math.cos(self.goal_guard_angle),
                self.goal[1] + self.guard_radius * math.sin(self.goal_guard_angle),
            ]
            self.obs_vel[0] = [0.0, 0.0]
            idx = 1

        # 남은 활성 슬롯을 배회 장애물로 채움
        for j, i in enumerate(range(idx, self.n_active)):
            if self.cfg.randomize_obstacles:
                cy, cx = cand[self.np_random.integers(len(cand))]
                self.obs_pos[i] = [cx + 0.5, cy + 0.5]
                ang = float(self.np_random.uniform(0, 2 * math.pi))
                spd = float(self.np_random.uniform(0.10, 0.24))
                self.obs_vel[i] = [spd * math.cos(ang), spd * math.sin(ang)]
            else:
                fx, fy = self._wanderer_frac[j % len(self._wanderer_frac)]
                tx = min(max(int(fx * self.cfg.width), 0), self.cfg.width - 1)
                ty = min(max(int(fy * self.cfg.height), 0), self.cfg.height - 1)
                d2 = (free[:, 1] - tx) ** 2 + (free[:, 0] - ty) ** 2
                cy, cx = free[int(np.argmin(d2))]
                self.obs_pos[i] = [cx + 0.5, cy + 0.5]
                self.obs_vel[i] = self._wanderer_vel[j % len(self._wanderer_vel)]

        self.obs_prev_pos[:] = self.obs_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.cfg.random_maps:
            layout, style, _ = self._new_random_layout()
            self._apply_layout(layout, style=style)
        self.agent_pos = self.start.copy()
        self.heading_idx = 0
        self.step_count = 0
        self.last_action = MOVE_FORWARD
        self.last_turn_dir = 0
        self.stagnation_steps = 0
        self._reset_obstacles()
        self.player_min_dist = 99.0      # 에피소드 중 플레이어와의 최소 거리
        self.player_near_steps = 0       # 플레이어 1.5칸 이내에 있던 스텝
        self.prev_potential = -self._bfs_dist_at(self.agent_pos)
        self.best_bfs = self._bfs_dist_at(self.agent_pos)
        self._ep_bfs0 = float(self.best_bfs)   # 에피소드 시작 시 최단경로 길이 (효율 분모용)

        # 평가용 통계
        self.n_turns = 0
        self.n_turn_reversals = 0      # 페널티와 동일 기준 (전진하면 리셋)
        self.path_length = 0.0
        # 자연스러움 지표 (페널티와 무관, 순수 측정용)
        self.total_turn_deg = 0.0      # 총 회전량. 필요 최소량 대비 클수록 흔들림
        self.heading_reversals = 0     # 전진이 포함 되어도 방향 전환 감지
        self._metric_last_dir = 0      # 전진/대기로 리셋되지 않는 회전 방향 기록
        self.n_waits = 0               # 대기 횟수. 0이면 타이밍 판단 안 한다는 신호
        self.n_blocked = 0             # 벽에 막힌 전진 횟수
        self.n_close_calls = 0         # 개인 공간 침범 스텝 수
        self.n_stationary = 0          # 이동이 전혀 없었던 스텝 수
        self.moving_turn_deg = 0.0     # 걸으면서 튼 각도
        self.static_turn_deg = 0.0     # 멈춰서 튼 각도
        self._steps_in_view = np.zeros(self.num_obstacles, dtype=np.int32)
        return self._get_obs(), {}

    # --------------------------------------------------
    def step(self, action: int):
        action = int(action)
        self.step_count += 1
        reward = self.cfg.time_penalty
        terminated = truncated = False
        info = {}

        # 장애물 이동
        self.obs_prev_pos[:] = self.obs_pos
        if self.playback:
            # 스크립트 주입: 그 장면의 실제 이웃 보행자 위치
            self.obs_prev_pos[:] = self.obs_pos
            self.obs_pos[:] = -10000.0
            if self.scripted_obs_fn is not None:
                pts = self.scripted_obs_fn(self.step_count)
                if pts is not None and len(pts):
                    k = min(len(pts), MAX_OBSTACLES)
                    self.obs_pos[:k] = np.asarray(pts[:k], dtype=np.float32)
        else:
            start_idx = 0
            if self.has_player:
                self._move_player()
                start_idx = 1
            elif self.has_guard:
                self.goal_guard_angle += 0.15
                self.obs_pos[0][0] = self.goal[0] + self.guard_radius * math.cos(self.goal_guard_angle)
                self.obs_pos[0][1] = self.goal[1] + self.guard_radius * math.sin(self.goal_guard_angle)
                start_idx = 1
            for i in range(start_idx, self.n_active):
                self._move_wanderer(i)

        # 행동 처리
        if action in (TURN_LEFT, TURN_RIGHT):
            reward += self.cfg.rotation_penalty
            reward += self._apply_turn(-1 if action == TURN_RIGHT else 1)
            self.n_stationary += 1

        elif action in (MOVE_FWD_LEFT, MOVE_FWD_RIGHT):
            reward += self.cfg.rotation_penalty * self.cfg.curved_move_penalty_scale
            turn_dir = -1 if action == MOVE_FWD_RIGHT else 1
            moved, blocked = self._do_forward_after_turn(turn_dir)
            reward += self._apply_turn(turn_dir, while_moving=moved)
            if blocked:
                reward += self.cfg.wall_collision_penalty
            if not moved:
                self.n_stationary += 1

        elif action == MOVE_FORWARD:
            moved, blocked = self._do_forward()
            if blocked:
                reward += self.cfg.wall_collision_penalty
            if not moved:
                self.n_stationary += 1
            self.last_turn_dir = 0

        elif action == WAIT:
            self.n_waits += 1
            self.n_stationary += 1
            self.last_turn_dir = 0

        self.last_action = action

        # 장애물 충돌 + 개인 공간 페널티
        for i in range(self.num_obstacles):
            d = float(np.linalg.norm(self.agent_pos - self.obs_pos[i]))
            if i == 0 and self.has_player:
                self.player_min_dist = min(self.player_min_dist, d)
                if d < 1.5:
                    self.player_near_steps += 1

            # 감지 반경 안에 머문 연속 스텝 수 기록
            # 충돌 시 '몇 스텝 전부터 보였는지 = 피할 시간이 있었는지' 판단
            if d <= SENSE_RADIUS:
                self._steps_in_view[i] += 1
            else:
                self._steps_in_view[i] = 0

            # 슬롯0이 플레이어면 플레이어 반경, 아니면 장애물 반경
            other_r = (self.cfg.player_radius if (i == 0 and self.has_player)
                       else self.cfg.obstacle_radius)
            hit_dist = self.cfg.agent_radius + other_r
            if d < hit_dist and not self.playback:
                reward += self.cfg.obstacle_collision_penalty
                terminated = True
                info["is_collision"] = True
                if i == 0 and self.has_player:
                    info["collided_with"] = "player"
                elif i == 0 and self.has_guard:
                    info["collided_with"] = "guard"
                else:
                    info["collided_with"] = "wanderer"
                info["fail_dist_to_goal"] = round(self._distance_to_goal(self.agent_pos), 2)
                info["view_steps_before_hit"] = int(self._steps_in_view[i])
                break

            # 카운터는 항상 측정 (페널티를 끈 실행과도 비교할 수 있어야 하므로)
            if d < self.cfg.personal_space:
                self.n_close_calls += 1
                if self.cfg.proximity_weight > 0.0:
                    reward -= self.cfg.proximity_weight * (1.0 - d / self.cfg.personal_space)

        # 진행 보상: potential-based shaping (대칭이라 왕복 해킹 불가)
        potential = -self._bfs_dist_at(self.agent_pos)
        reward += self.cfg.progress_weight * (potential - self.prev_potential)
        self.prev_potential = potential

        # 방향 정렬: 가중치를 크게 낮춰 '격자 정렬 강제' 제거
        iv = self._get_ideal_direction(self.agent_pos)
        reward += self.cfg.align_weight * float(np.dot(iv, self.heading_unit_vector()))

        # 정체: 유클리드 대신 BFS 거리 기준
        cur_bfs = self._bfs_dist_at(self.agent_pos)
        if cur_bfs < self.best_bfs - 1e-6:
            self.best_bfs = cur_bfs
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1
        if self.stagnation_steps >= self.cfg.stagnation_patience and not self.playback:
            reward += self.cfg.stagnation_penalty
            terminated = True
            info["is_stagnation"] = True

        # 목표 도달
        if self._distance_to_goal(self.agent_pos) <= self.cfg.goal_radius:
            reward += self.cfg.goal_reward
            terminated = True
            info["is_success"] = True

        if self.step_count >= self.cfg.max_steps:
            truncated = True

        if terminated or truncated:
            # SB3는 info에 'is_success' 키가 있을 때만 success_rate 버퍼에 담음.
            # 실패 시 키를 아예 넣지 않으면 True만 쌓여 success_rate가 항상 1.0
            info.setdefault("is_success", False)
            if info["is_success"]:
                outcome = "success"
            elif info.get("is_collision"):
                outcome = "collision"
            elif info.get("is_stagnation"):
                outcome = "stagnation"
            else:
                outcome = "timeout"
            info.update(outcome=outcome,
                        straight_dist=round(float(np.linalg.norm(self.goal - self.start)), 3),
                        bfs_dist=self._ep_bfs0,
                        player_kind=(getattr(self, "player_kind", None)
                                     if self.has_player else None),
                        player_min_dist=(round(self.player_min_dist, 3)
                                         if self.has_player else None),
                        player_near_steps=(self.player_near_steps
                                           if self.has_player else None),
                        map_w=self.cfg.width, map_h=self.cfg.height,
                        map_style=getattr(self, "map_style", None),
                        steps=self.step_count, n_turns=self.n_turns,
                        n_turn_reversals=self.n_turn_reversals,
                        path_length=round(self.path_length, 2),
                        total_turn_deg=self.total_turn_deg,
                        heading_reversals=self.heading_reversals,
                        n_waits=self.n_waits, n_blocked=self.n_blocked,
                        n_close_calls=self.n_close_calls,
                        n_stationary=self.n_stationary,
                        stationary_ratio=round(self.n_stationary / max(self.step_count, 1), 3),
                        moving_turn_deg=self.moving_turn_deg,
                        static_turn_deg=self.static_turn_deg)

        return self._get_obs(), float(reward), terminated, truncated, info


# =========================
def _require_model(path, what):
    import os
    if os.path.exists(path):
        return
    cand = path if path.endswith(".zip") else path + ".zip"
    if os.path.exists(cand):
        return
    msg = [f"{what} 모델 파일을 찾을 수 없습니다: {path}"]
    d = os.path.dirname(path) or "."
    if os.path.isdir(d):
        zips = sorted(f for f in os.listdir(d) if f.endswith(".zip"))
        if zips:
            msg.append(f"  '{d}' 안의 모델: " + ", ".join(zips[:12])
                       + (" ..." if len(zips) > 12 else ""))
        else:
            msg.append(f"  '{d}' 폴더에 .zip 모델이 없습니다. 학습을 먼저 실행하세요.")
    else:
        msg.append(f"  '{d}' 폴더가 없습니다.")
    raise SystemExit("\n".join(msg))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--map", default=DEFAULT_MAP)
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--out", default=None)
    p.add_argument("--randomize", action="store_true", help="장애물 초기 상태 랜덤화")
    p.add_argument("--check-all", action="store_true", help="학습 없이 모든 맵 검증")
    p.add_argument("--eval", default=None, help="모델 경로. 성공률/경로길이/회전수 측정")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--deterministic", action="store_true",
                   help="평가 시 행동 샘플링 없이 최선 행동만 사용 (샘플링 노이즈 영향 분리)")
    p.add_argument("--player", default="",
                   choices=["", "goal", "chase", "random", "cross", "mix"],
                   help="장애물 슬롯0을 플레이어로 교체. "
                        "goal=목적지형 chase=추적형 random=무작위전환 "
                        "cross=교차형(진로 가로지름) mix=혼합. "
                        "좁은 통로 스타일(winding_corridor/sparse_maze)은 장애물이 "
                        "자동 비활성되어 플레이어도 생성되지 않으므로, 랜덤 맵에서는 "
                        "--styles open_clusters,rooms 를 함께 쓰세요")
    p.add_argument("--player-speed", type=float, default=0.22)
    p.add_argument("--player-radius", type=float, default=0.40,
                   help="플레이어 반경. 기본 0.40(충돌거리 0.62). "
                        "0.25 로 줄이면 충돌거리 0.47 이 되어 좁은 곳에서 비켜가기 쉬움")
    p.add_argument("--random-maps", action="store_true",
                   help="reset 마다 random_map_gen 으로 새 맵 생성 (학습/평가 공용)")
    p.add_argument("--styles", default=None,
                   help="랜덤 맵 스타일 제한. 쉼표 구분 "
                        "(open_clusters,winding_corridor,rooms,sparse_maze)")
    p.add_argument("--init-from", default=None,
                   help="기존 모델(.zip)에서 이어서 학습 (관측 34차원 모델이면 어떤 것이든 가능)")
    p.add_argument("--n-obstacles", type=int, default=3,
                   help="활성 장애물 수 (0~3). 좁은 복도형 맵은 1개 이하 권장. "
                        "관측 차원은 항상 같으므로 같은 모델을 그대로 평가할 수 있음")
    p.add_argument("--seed", type=int, default=None,
                   help="학습 시드. RL은 시드마다 결과가 크게 갈리므로 "
                        "결론을 내리기 전 최소 3개 시드로 돌려 비교하세요")
    p.add_argument("--turn-reversal", type=float, default=None,
                   help="방향 전환(좌<->우) 페널티. 기본 -0.06. "
                        "-0.15 처럼 크게 주면 좌우로 떠는 슬라럼 주행이 줄어듭니다")
    p.add_argument("--fixed-obstacle-order", action="store_true",
                   help="관측에서 장애물을 거리순 정렬하지 않고 슬롯을 고정 "
                        "(0번=수문장). 순위 뒤바뀜에 의한 관측 점프를 없앰")
    p.add_argument("--personal-space", type=float, default=0.0,
                   help="개인 공간 페널티 가중치. 0.3 정도부터 시작 권장. 0이면 비활성")
    p.add_argument("--frame-stack", type=int, default=1,
                   help="최근 N프레임을 겹쳐서 관측으로 사용. 2 이상이면 "
                        "속도 변화(가속·곡선)를 신경망이 스스로 추론할 수 있음. "
                        "학습과 평가에 반드시 같은 값을 써야 함")
    p.add_argument("--version", action="store_true", help="파일 버전만 출력")
    args = p.parse_args()

    if args.version:
        print(f"visual_gym {VERSION}")
        print(f"관측 {OBS_DIM}차원 / 행동 {N_ACTIONS}개 / 기본 맵 '{DEFAULT_MAP}'")
        print(f"등록된 맵: {sorted(MAPS.keys())}")
        raise SystemExit

    if args.check_all:
        print(f"visual_gym {VERSION}")
        print(f"관측 차원: {OBS_DIM} / 행동 {N_ACTIONS}개 / 기본 맵 '{DEFAULT_MAP}'\n")
        for name in MAPS:
            try:
                e = GymHumanLikeNavEnv(map_name=name)
                d = e.grid_goal_distance[int(e.start[1]), int(e.start[0])]
                o, _ = e.reset()
                guard = f"수문장 r={e.guard_radius:.1f}" if e.has_guard else "수문장 없음(공간부족)"
                print(f"  [OK] {name:20s} {e.cfg.width:2d}x{e.cfg.height:<2d} BFS={d:3d} "
                      f"max_steps={e.cfg.max_steps:4d} 활성장애물={e.n_active} {guard}")
            except Exception as exc:
                print(f"  [FAIL] {name:24s} {exc}")
        if HAS_RANDOM_MAPS:
            print("\n  랜덤 맵 샘플 검증:")
            cfg = EnvConfig(); cfg.random_maps = True
            e = GymHumanLikeNavEnv(config=cfg)
            for i in range(4):
                o, _ = e.reset(seed=100 + i)
                d = e.grid_goal_distance[int(e.start[1]), int(e.start[0])]
                guard = f"수문장 r={e.guard_radius:.1f}" if e.has_guard else "수문장 없음"
                print(f"  [OK] random({e.map_style or '?':16s}) {e.cfg.width:2d}x{e.cfg.height:<2d} "
                      f"BFS={d:3d} 활성장애물={e.n_active} {guard}")
        else:
            print("\n  (random_map_gen.py 없음 - 랜덤 맵 기능 비활성)")
        raise SystemExit

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    def _build_env(map_name, randomize):
        c = GymHumanLikeNavEnv(map_name=map_name).cfg
        c.randomize_obstacles = randomize
        c.proximity_weight = args.personal_space
        c.sort_obstacles_by_distance = not args.fixed_obstacle_order
        c.num_obstacles = args.n_obstacles
        c.player_policy = args.player
        if args.player and args.random_maps and not args.styles:
            print("[안내] 좁은 통로 스타일에서는 플레이어가 생성되지 않습니다. "
                  "--styles open_clusters,rooms 권장")
        c.player_speed = args.player_speed
        c.player_radius = args.player_radius
        c.random_maps = args.random_maps
        if args.random_maps:
            c.randomize_obstacles = True   # 랜덤 맵이면 장애물 초기 상태도 함께 랜덤화
        if args.styles:
            c.random_map_styles = tuple(x.strip() for x in args.styles.split(",") if x.strip())
        if args.turn_reversal is not None:
            c.turn_reversal_penalty = args.turn_reversal
        return GymHumanLikeNavEnv(config=c, map_name=map_name)

    def _build_vec(map_name, randomize, n_stack):
        venv = DummyVecEnv([lambda: _build_env(map_name, randomize)])
        if n_stack > 1:
            venv = VecFrameStack(venv, n_stack=n_stack)
        return venv

    def _base_env(venv, n_stack):
        return (venv.venv if n_stack > 1 else venv).envs[0]

    if args.eval:
        venv = _build_vec(args.map, args.randomize, args.frame_stack)
        base = _base_env(venv, args.frame_stack)
        _require_model(args.eval, "평가할")
        model = PPO.load(args.eval)

        straight = float(np.linalg.norm(base.goal - base.start))
        succ = 0
        outcomes, culprits = {}, {}
        fail_steps, fail_dists, view_steps = [], [], []
        M = {k: [] for k in ("steps", "n_turns", "n_turn_reversals", "path_length",
                             "total_turn_deg", "heading_reversals", "n_waits", "n_blocked",
                             "n_close_calls", "stationary_ratio",
                             "moving_turn_deg", "static_turn_deg")}
        player_mins, player_nears = [], []
        eff_list = []       # 에피소드별 경로/직선 비율 (랜덤 맵은 직선거리가 매번 다름)
        eff_bfs_list = []   # 에피소드별 경로/BFS최단 비율 (미로형 맵에서도 유효한 효율)
        by_style = {}       # 랜덤 맵: 스타일별 (성공수, 전체수, 경로/BFS 리스트)

        obs = venv.reset()
        done_count = 0
        while done_count < args.episodes:
            a, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rew, dones, infos = venv.step(a)
            if dones[0]:
                info = infos[0]
                done_count += 1
                outcomes[info["outcome"]] = outcomes.get(info["outcome"], 0) + 1
                if info.get("is_success"):
                    succ += 1
                    for k in M:
                        M[k].append(info[k])
                    if info.get("straight_dist", 0) > 0:
                        eff_list.append(info["path_length"] / info["straight_dist"])
                    if info.get("bfs_dist", 0) > 0:
                        eff_bfs_list.append(info["path_length"] / info["bfs_dist"])
                if info.get("player_min_dist") is not None:
                    player_mins.append(info["player_min_dist"])
                    player_nears.append(info["player_near_steps"])
                st = info.get("map_style") or "-"
                rec = by_style.setdefault(st, [0, 0, []])
                rec[1] += 1
                if info.get("is_success"):
                    rec[0] += 1
                    if info.get("bfs_dist", 0) > 0:
                        rec[2].append(info["path_length"] / info["bfs_dist"])
                else:
                    fail_steps.append(info["steps"])
                    if "collided_with" in info:
                        culprits[info["collided_with"]] = culprits.get(info["collided_with"], 0) + 1
                        fail_dists.append(info["fail_dist_to_goal"])
                        view_steps.append(info["view_steps_before_hit"])

        mode = "deterministic" if args.deterministic else "stochastic"
        print(f"visual_gym {VERSION}")
        map_desc = "랜덤" if args.random_maps else args.map
        dist_desc = "가변" if args.random_maps else f"{straight:.2f}"
        print(f"맵 {map_desc} / {args.episodes} 에피소드  (직선거리 {dist_desc}, "
              f"randomize={base.cfg.randomize_obstacles}, {mode}, frame_stack={args.frame_stack}, "
              f"personal_space={base.cfg.proximity_weight})")
        print(f"  성공률          : {succ / args.episodes * 100:.1f}%")

        print("  --- 실패 원인 ---")
        for k in ("collision", "stagnation", "timeout"):
            n = outcomes.get(k, 0)
            if n:
                print(f"    {k:10s}: {n / args.episodes * 100:5.1f}%")
        if culprits:
            tot = sum(culprits.values())
            for k, n in sorted(culprits.items(), key=lambda kv: -kv[1]):
                label = {"guard": "수문장(목표 공전)",
                         "player": "플레이어",
                         "wanderer": "배회 장애물"}.get(k, k)
                print(f"    충돌 상대 - {label:18s}: {n / tot * 100:5.1f}%")
            note = (f"  (수문장 궤도 반경 {base.guard_radius})"
                    if getattr(base, "has_guard", False) else "")
            print(f"    충돌 시점 목표까지 거리 평균: {np.mean(fail_dists):.2f}{note}")
            vs = np.array(view_steps)
            ambush = int((vs <= 5).sum())
            print(f"    충돌 전 그 장애물이 보였던 스텝: 평균 {vs.mean():.1f}, 중앙값 {np.median(vs):.0f}")
            print(f"      5스텝 이하만 보이고 충돌(회피 불가에 가까움): {ambush}/{len(vs)}건"
                  f" = {ambush / len(vs) * 100:.0f}%")
        if fail_steps:
            early = sum(1 for s in fail_steps if s <= 15)
            print(f"    실패까지 평균 {np.mean(fail_steps):.1f}스텝 "
                  f"(15스텝 이내 조기실패 {early}/{len(fail_steps)}건)")

        if succ:
            m = {k: float(np.mean(v)) for k, v in M.items()}
            s = {k: float(np.std(v)) for k, v in M.items()}
            print("  --- 성공 궤적 특성 ---")
            print(f"    평균 스텝       : {m['steps']:.1f}  (표준편차 {s['steps']:.1f})")
            print(f"    평균 경로길이   : {m['path_length']:.2f}  (표준편차 {s['path_length']:.2f})")
            eff = float(np.mean(eff_list)) if eff_list else m['path_length'] / straight
            print(f"    경로 효율(직선) : {eff:.3f}   <- 열린 맵 기준. 미로형 맵에선 1보다 훨씬 커지는 게 정상")
            if eff_bfs_list:
                effb = float(np.mean(eff_bfs_list))
                print(f"    경로 효율(BFS)  : {effb:.3f}   <- 4방향 최단경로 대비. "
                      f"1 미만 = 대각선으로 질러감, 1.3 이상 = 배회")
            print(f"    평균 회전수     : {m['n_turns']:.1f}")
            print(f"    총 회전량       : {m['total_turn_deg']:.0f}도")
            print(f"      ├ 걸으면서 튼 각도: {m['moving_turn_deg']:.0f}도  <- 사람도 이렇게 함, 많아도 괜찮음")
            print(f"      └ 멈춰서 튼 각도  : {m['static_turn_deg']:.0f}도  <- 낮을수록 사람다움 (핵심)")
            print(f"    방향 전환 횟수  : {m['heading_reversals']:.1f}"
                  f"  (이동 1칸당 {m['heading_reversals'] / max(m['path_length'], 1e-9):.2f}회"
                  f"  <- 0.4 넘으면 좌우로 떠는 중)")
            print(f"    대기 횟수       : {m['n_waits']:.1f}")
            print(f"    벽에 막힌 전진  : {m['n_blocked']:.1f}")
            if args.player:
                pm = [i for i in player_mins if i is not None]
                pn = [i for i in player_nears if i is not None]
                if pm:
                    print(f"    --- 플레이어 상호작용 ---")
                    print(f"    최소 거리 평균  : {np.mean(pm):.2f} 칸  <- 클수록 여유롭게 피함")
                    print(f"    1.5칸 이내 스텝 : {np.mean(pn):.1f}")
            print(f"    개인공간 침범   : {m['n_close_calls']:.1f} 스텝  <- 낮을수록 여유롭게 피함")
            print(f"    정지 스텝 비율  : {m['stationary_ratio'] * 100:.1f}%  "
                  f"<- 낮을수록 멈추지 않고 걸으면서 방향을 틈 (사람다움)")
            if s["path_length"] < 0.3 and m["n_waits"] < 0.5:
                print("\n    ⚠ 경로 편차가 거의 없고 대기도 안 합니다. 단일 궤적 암기 의심.")

        if args.random_maps and len(by_style) > 1:
            print("  --- 스타일별 분해 ---")
            for st in sorted(by_style):
                ok, tot, effs = by_style[st]
                eb = f"{float(np.mean(effs)):.2f}" if effs else "  - "
                print(f"    {st:18s}: {ok:3d}/{tot:3d} = {ok / max(tot, 1) * 100:5.1f}%  "
                      f"경로/BFS {eb}")

        # 95% 신뢰구간: 실행 간 차이가 유의한지 판단
        pr = succ / args.episodes
        se = math.sqrt(max(pr * (1 - pr), 1e-9) / args.episodes)
        print(f"\n  성공률 95% 신뢰구간: {max(0, pr - 1.96 * se) * 100:.1f}% "
              f"~ {min(1, pr + 1.96 * se) * 100:.1f}%  (n={args.episodes})")
        raise SystemExit

    venv = _build_vec(args.map, args.randomize, args.frame_stack)
    base = _base_env(venv, args.frame_stack)
    # 파일명에 소수점은 'p'로 치환
    ps_tag = f"_ps{args.personal_space}".replace(".", "p") if args.personal_space > 0 else ""
    out = args.out or (f"ppo_nav_{args.map}" + ("_rand" if args.randomize else "")
                       + (f"_fs{args.frame_stack}" if args.frame_stack > 1 else "")
                       + ps_tag
                       + ("_fix" if args.fixed_obstacle_order else "")
                       + (f"_o{args.n_obstacles}" if args.n_obstacles != 3 else "")
                       + ("_rm" if args.random_maps else "")
                       + (f"_pl{args.player}" if args.player else "")
                       + (f"_r{args.player_radius}".replace(".", "p")
                          if args.player and args.player_radius != 0.40 else "")
                       + (f"_tr{args.turn_reversal}".replace(".", "p").replace("-", "m")
                          if args.turn_reversal is not None else "")
                       + (f"_s{args.seed}" if args.seed is not None else ""))
    if "." in out:
        out = out.replace(".", "p")
    print(f"visual_gym {VERSION}")
    map_banner = "랜덤 (에피소드마다 생성)" if args.random_maps else f"'{args.map}' ({base.cfg.width}x{base.cfg.height})"
    print(f"=== 맵 {map_banner} / obs_dim={OBS_DIM}"
          f" x frame_stack {args.frame_stack} / randomize={base.cfg.randomize_obstacles} ===")
    if args.seed is not None:
        venv.seed(args.seed)
    if args.init_from:
        print(f"    기존 모델에서 이어 학습: {args.init_from}")
        _require_model(args.init_from, "이어 학습할")
        model = PPO.load(args.init_from, env=venv)
        if args.seed is not None:
            model.set_random_seed(args.seed)
    else:
        model = PPO("MlpPolicy", venv, verbose=1,
                    learning_rate=3e-4, n_steps=2048, batch_size=64,
                    ent_coef=0.01, gamma=0.99, gae_lambda=0.95,
                    seed=args.seed)
    model.learn(total_timesteps=args.steps)
    model.save(out)
    import os
    actual = out + ".zip" if os.path.exists(out + ".zip") else out
    print(f"=== 완료: 실제 저장된 파일 = {actual} ===")

    eval_flags = ""
    if args.random_maps:
        eval_flags += " --random-maps"
        if args.styles:
            eval_flags += f" --styles {args.styles}"
    else:
        if args.map != DEFAULT_MAP:
            eval_flags += f" --map {args.map}"
        if args.randomize:
            eval_flags += " --randomize"
    if args.player:
        eval_flags += f" --player {args.player}"
    if args.player_speed != 0.22:
        eval_flags += f" --player-speed {args.player_speed}"
    if args.n_obstacles != 3:
        eval_flags += f" --n-obstacles {args.n_obstacles}"
    if args.personal_space > 0:
        eval_flags += f" --personal-space {args.personal_space}"
    if args.frame_stack > 1:
        eval_flags += f" --frame-stack {args.frame_stack}"
    if args.fixed_obstacle_order:
        eval_flags += " --fixed-obstacle-order"
    print(f"    평가:  python visual_gym.py --eval {actual}{eval_flags} --episodes 500")
