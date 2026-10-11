import argparse
import json
import math
import os
import time
import zlib

import numpy as np

from visual_gym import (GymHumanLikeNavEnv, EnvConfig, DEFAULT_MAP, VERSION,
                        TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, WAIT,
                        MOVE_FWD_LEFT, MOVE_FWD_RIGHT, HAS_RANDOM_MAPS)

DATA_FORMAT_VERSION = 1


# --------------------------------------------------
def action_from_keys(fwd, left, right):
    if left and right:
        left = right = False
    if fwd and left:
        return MOVE_FWD_LEFT
    if fwd and right:
        return MOVE_FWD_RIGHT
    if fwd:
        return MOVE_FORWARD
    if left:
        return TURN_LEFT
    if right:
        return TURN_RIGHT
    return WAIT


def action_from_direction(up, down, left, right, heading_idx, n_headings,
                          step_deg, turn_threshold_deg=75.0):
    dx = (1 if right else 0) - (1 if left else 0)
    dy = (1 if up else 0) - (1 if down else 0)      # 화면 위 = +y (환경 좌표와 동일)
    if dx == 0 and dy == 0:
        return WAIT

    want = math.degrees(math.atan2(dy, dx)) % 360.0
    cur = (heading_idx * step_deg) % 360.0
    diff = (want - cur + 180.0) % 360.0 - 180.0     # -180 ~ +180

    if abs(diff) <= step_deg / 2:                   # 거의 맞음
        return MOVE_FORWARD
    if abs(diff) <= turn_threshold_deg:             # 조금 틀어짐 -> 걸으면서 보정
        return MOVE_FWD_LEFT if diff > 0 else MOVE_FWD_RIGHT
    return TURN_LEFT if diff > 0 else TURN_RIGHT    # 크게 틀어짐 -> 제자리 회전


# --------------------------------------------------
class Recorder:
    def __init__(self):
        self.obs, self.acts, self.pos, self.ep_starts = [], [], [], []
        self.ep_meta = []
        self._cur = None

    def begin_episode(self):
        self._cur = {"obs": [], "acts": [], "pos": []}

    def record(self, obs, action, pos):
        self._cur["obs"].append(np.asarray(obs, dtype=np.float32))
        self._cur["acts"].append(int(action))
        self._cur["pos"].append(np.asarray(pos, dtype=np.float32))

    def discard_episode(self):
        self._cur = None

    def commit_episode(self, info, seed, trim_lead_wait=True):
        o, a, p = self._cur["obs"], self._cur["acts"], self._cur["pos"]
        self._cur = None
        if not a:
            return False
        start = 0
        if trim_lead_wait:
            while start < len(a) - 1 and a[start] == WAIT:
                start += 1
        o, a, p = o[start:], a[start:], p[start:]
        if len(a) < 3:
            return False
        for i in range(len(a)):
            self.ep_starts.append(i == 0)
        self.obs.extend(o)
        self.acts.extend(a)
        self.pos.extend(p)
        meta = {k: info.get(k) for k in
                ("is_success", "outcome", "steps", "path_length", "n_turns",
                 "heading_reversals", "n_waits", "stationary_ratio",
                 "total_turn_deg", "moving_turn_deg", "static_turn_deg",
                 "map_style", "map_w", "map_h", "bfs_dist", "straight_dist")}
        meta["seed"] = seed
        meta["recorded_steps"] = len(a)
        self.ep_meta.append(json.dumps(meta))
        return True

    def save(self, path, subject, fps):
        np.savez_compressed(
            path,
            obs=np.asarray(self.obs, dtype=np.float32),
            acts=np.asarray(self.acts, dtype=np.int8),
            pos=np.asarray(self.pos, dtype=np.float32),
            ep_starts=np.asarray(self.ep_starts, dtype=bool),
            ep_meta=np.asarray(self.ep_meta),
            subject=subject, fps=fps,
            env_version=VERSION, data_format=DATA_FORMAT_VERSION,
        )

    @property
    def n_episodes(self):
        return len(self.ep_meta)

# 피험자별로 재현 가능하되 서로 다른 맵 시퀀스
def episode_seed(subject, ep_index):
    return zlib.crc32(f"{subject}/{ep_index}".encode()) % (2 ** 31 - 1)

# 수집용 환경
def build_env(args):
    player = getattr(args, "player", "cross")
    styles = getattr(args, "styles", "open_clusters,rooms")

    cfg = EnvConfig()
    cfg.randomize_obstacles = True
    cfg.num_obstacles = args.n_obstacles
    cfg.player_policy = player
    cfg.player_speed = getattr(args, "player_speed", 0.22)
    cfg.player_radius = getattr(args, "player_radius", 0.30)

    if args.no_random or not HAS_RANDOM_MAPS:
        cfg.random_maps = False
        env = GymHumanLikeNavEnv(config=cfg, map_name=args.map)
        return env

    cfg.random_maps = True
    if styles:
        cfg.random_map_styles = tuple(x.strip() for x in styles.split(",") if x.strip())
    return GymHumanLikeNavEnv(config=cfg)


# --------------------------------------------------
def scripted_player(obs, rng):
    sin_i, cos_i = float(obs[3]), float(obs[4])   # BFS 이상 방향 (에이전트 기준)
    fp_x, fp_y, vis = float(obs[12]), float(obs[13]), float(obs[16])
    if vis > 0.5 and 0.0 < fp_x < 0.22 and abs(fp_y) < 0.13 and rng.random() < 0.7:
        return WAIT
    if cos_i > 0.94:
        return MOVE_FORWARD
    if sin_i > 0:
        return MOVE_FWD_LEFT if cos_i > 0.25 else TURN_LEFT
    return MOVE_FWD_RIGHT if cos_i > 0.25 else TURN_RIGHT


def run_selftest(args):
    print("자가 검증")
    env = build_env(args)
    rec = Recorder()
    rng = np.random.default_rng(0)
    n_ok = 0
    for ep in range(args.episodes):
        seed = episode_seed(args.subject, ep)
        obs, _ = env.reset(seed=seed)
        rec.begin_episode()
        while True:
            a = scripted_player(obs, rng)
            rec.record(obs, a, env.agent_pos)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        rec.commit_episode(info, seed)
        n_ok += bool(info.get("is_success"))
        print(f"  ep{ep + 1:02d} {info.get('outcome'):10s} "
              f"{info.get('steps'):4d}스텝  {info.get('map_style') or args.map}")
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.subject}_selftest.npz")
    rec.save(path, args.subject, args.fps)
    print(f"\n성공 {n_ok}/{args.episodes} | 저장: {path}")
    d = np.load(path, allow_pickle=False)
    print(f"검증: obs {d['obs'].shape}, acts {d['acts'].shape}, "
          f"에피소드 {int(d['ep_starts'].sum())}개")


# 본 수집 (pygame)
def find_korean_font(pygame):
    for name in ("nanumgothic", "malgungothic", "applegothic",
                 "notosanscjkkr", "notosanskr", "nanumbarungothic"):
        path = pygame.font.match_font(name)
        if path:
            return path
    return None


def run_collect(args):
    import pygame

    env = build_env(args)
    rec = Recorder()

    pygame.init()
    font_path = find_korean_font(pygame)
    KO = font_path is not None

    def L(ko, en):
        return ko if KO else en

    WIN_W, WIN_H = 1180, 820
    PANEL_W = 270                      # 오른쪽 안내문 영역

    try:
        screen = pygame.display.set_mode((WIN_W, WIN_H))
    except pygame.error as e:
        print(f"화면을 열 수 없습니다: {e}")
        print("WSL이라면 Windows 11 WSLg가 필요합니다. 또는 Windows Python으로 실행하세요.")
        return

    WIN_W, WIN_H = screen.get_size()

    layout = {"cell": 20, "ox": 20, "oy": 30}

    def fit_layout():
        avail_w = WIN_W - PANEL_W - 40
        avail_h = WIN_H - 60
        cell = max(10, min(avail_w // env.cfg.width, avail_h // env.cfg.height))
        mw, mh = env.cfg.width * cell, env.cfg.height * cell
        layout["cell"] = cell
        layout["ox"] = 20 + max(0, (avail_w - mw) // 2)
        layout["oy"] = 30 + max(0, (avail_h - mh) // 2)
        return cell
    pygame.display.set_caption(f"이동 데이터 수집 - {args.subject}")
    print(f"창 크기 {WIN_W}x{WIN_H}. 창을 클릭해 포커스를 준 뒤 Enter 또는 Space 로 시작하세요.")
    print("창이 안 보이면 작업표시줄에서 'Python' 창을 찾아보세요.")
    font = pygame.font.Font(font_path, 20) if KO else pygame.font.Font(None, 24)
    big = pygame.font.Font(font_path, 34) if KO else pygame.font.Font(None, 42)
    clock = pygame.time.Clock()

    C_FLOOR, C_WALL = (232, 228, 214), (110, 104, 92)
    C_GRID = (214, 210, 198)
    C_START, C_GOAL, C_OBS = (46, 125, 50), (249, 168, 37), (198, 40, 40)
    C_PERSON = (123, 31, 162)
    C_AGENT, C_TRAIL = (21, 101, 192), (21, 101, 192)

    def to_px(x, y):
        c = layout["cell"]
        return (int(layout["ox"] + x * c),
                int(layout["oy"] + (env.cfg.height - y) * c))

    def draw_world(trail):
        cell = layout["cell"]
        screen.fill((246, 243, 234))
        for gy in range(env.cfg.height):
            for gx in range(env.cfg.width):
                px, py = to_px(gx, gy + 1)
                rect = pygame.Rect(px, py, cell, cell)
                pygame.draw.rect(screen, C_WALL if env.grid[gy, gx] else C_FLOOR, rect)
                pygame.draw.rect(screen, C_GRID, rect, 1)
        pygame.draw.circle(screen, C_START, to_px(*env.start), cell // 3)
        gx, gy = to_px(*env.goal)
        pygame.draw.circle(screen, C_GOAL, (gx, gy), int(env.cfg.goal_radius * cell), 3)
        pygame.draw.circle(screen, C_GOAL, (gx, gy), cell // 4)
        if len(trail) > 1:
            pygame.draw.lines(screen, C_TRAIL, False, [to_px(*p) for p in trail], 2)
        for i in range(env.n_active):
            ox, oy = env.obs_pos[i]
            if ox > -5:
                is_person = (i == 0 and env.has_player)
                col = C_PERSON if is_person else C_OBS
                rad = env.cfg.player_radius if is_person else env.cfg.obstacle_radius
                pygame.draw.circle(screen, col, to_px(ox, oy), int(rad * cell))
                if is_person:
                    # 흰 테두리로 한 번 더 구분
                    pygame.draw.circle(screen, (255, 255, 255),
                                       to_px(ox, oy), int(rad * cell), 2)
        # 에이전트 진행 방향을 향하는 삼각형
        import math
        th = env.heading_rad()
        cx, cy = env.agent_pos
        pts = []
        for ang, r in ((0, 0.42), (2.5, 0.30), (-2.5, 0.30)):
            pts.append(to_px(cx + r * math.cos(th + ang), cy + r * math.sin(th + ang)))
        pygame.draw.polygon(screen, C_AGENT, pts)

    def hud(lines, y0=24):
        x = WIN_W - PANEL_W + 16
        for i, t in enumerate(lines):
            screen.blit(font.render(t, True, (60, 56, 48)), (x, y0 + i * 26))

    def banner(text, color=(60, 56, 48)):
        s = big.render(text, True, color)
        screen.blit(s, s.get_rect(center=((WIN_W - PANEL_W) // 2, WIN_H // 2)))

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(
        args.outdir, f"{args.subject}_{time.strftime('%m%d_%H%M%S')}.npz")

    saved = 0
    ep = 0
    total_needed = args.practice + args.episodes
    print(f"연습 {args.practice}판(기록 안 함) + 본 수집 {args.episodes}판")

    running = True
    while running and ep < total_needed:
        seed = episode_seed(args.subject, ep)
        obs, _ = env.reset(seed=seed)
        fit_layout()          # 맵 크기가 바뀌었을 수 있으므로 배치만 다시 계산
        practice = ep < args.practice
        trail = [tuple(env.agent_pos)]

        # 시작 대기: 지형 훑어볼 시간
        waiting = True
        while waiting and running:
            draw_world(trail)
            banner(L("Enter 또는 Space 를 누르면 시작", "Press Enter or Space to start"))
            hud([L(f"피험자: {args.subject}", f"subject: {args.subject}"),
                 L(f"{'연습' if practice else '본 수집'} "
                   f"{ep + 1}/{total_needed}", f"ep {ep + 1}/{total_needed}"),
                 "",
                 L("목적지(노란 원)까지", "walk to the yellow goal"),
                 L("평소 걷듯이 이동하세요", "as you normally would"),
                 "",
                 (L("방향키: 가려는 방향", "arrows: direction to go")
                  if args.control == "direct"
                  else L("↑전진  ←→회전", "up=forward, L/R=turn")),
                 "",
                 L("보라 = 마주 오는 사람", "purple = oncoming person"),
                 L("빨강 = 장애물", "red = obstacle"),
                 "",
                 L("R: 다시   ESC: 종료", "R: retry  ESC: quit")])
            pygame.display.flip()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                        waiting = False
                    elif e.key == pygame.K_ESCAPE:
                        running = False
            clock.tick(30)
        if not running:
            break

        # 플레이
        rec.begin_episode()
        info = {}
        retry = False
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_r:
                        retry = True
            if retry or not running:
                break

            keys = pygame.key.get_pressed()
            up = keys[pygame.K_UP] or keys[pygame.K_w]
            down = keys[pygame.K_DOWN] or keys[pygame.K_s]
            lf = keys[pygame.K_LEFT] or keys[pygame.K_a]
            rt = keys[pygame.K_RIGHT] or keys[pygame.K_d]
            if args.control == "direct":
                a = action_from_direction(up, down, lf, rt, env.heading_idx,
                                          env.n_headings, env.cfg.heading_step_deg)
            else:
                a = action_from_keys(up, lf, rt)
            rec.record(obs, a, env.agent_pos)
            obs, r, term, trunc, info = env.step(a)
            trail.append(tuple(env.agent_pos))

            draw_world(trail)
            hud([L(f"{'연습' if practice else '기록 중'} "
                   f"{ep + 1}/{total_needed}", f"ep {ep + 1}/{total_needed}"),
                 L(f"저장된 판: {saved}", f"saved: {saved}"),
                 "", L("R: 다시   ESC: 종료", "R: retry  ESC: quit")])
            pygame.display.flip()
            clock.tick(args.fps)

            if term or trunc:
                break

        if retry:
            rec.discard_episode()
            continue                        # 같은 ep 번호로 다시
        if not running:
            rec.discard_episode()
            break

        ok = bool(info.get("is_success"))
        if practice:
            rec.discard_episode()
        else:
            if rec.commit_episode(info, seed):
                saved += 1
        draw_world(trail)
        banner(L("도착!", "SUCCESS") if ok else L("실패...", "FAILED"),
               (46, 125, 50) if ok else (198, 40, 40))
        pygame.display.flip()
        pygame.time.wait(900)
        ep += 1

    pygame.quit()
    if rec.n_episodes:
        rec.save(out_path, args.subject, args.fps)
        succ = sum(json.loads(m).get("is_success", False) for m in rec.ep_meta)
        print(f"저장 완료: {out_path}")
        print(f"  에피소드 {rec.n_episodes}개 (성공 {succ}) / 총 {len(rec.acts)}스텝")
    else:
        print("저장할 에피소드가 없습니다.")


# --------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="anon", help="피험자 ID (예: P01)")
    p.add_argument("--episodes", type=int, default=10, help="본 수집 판 수")
    p.add_argument("--practice", type=int, default=2, help="연습 판 수 (기록 안 함)")
    p.add_argument("--control", default="direct", choices=["direct", "tank"],
                   help="direct=8방향 직접 조작(쉬움, 기본) / tank=바라보는 방향 기준 회전")
    p.add_argument("--fps", type=int, default=10,
                   help="초당 스텝 수. 10이면 이동속도 2.5칸/초")
    p.add_argument("--map", default=DEFAULT_MAP)
    p.add_argument("--no-random", action="store_true", help="랜덤 맵 대신 --map 고정")
    p.add_argument("--n-obstacles", type=int, default=3)
    p.add_argument("--player", default="cross",
                   choices=["", "goal", "chase", "random", "cross", "mix"],
                   help="마주치는 사람 역할. 기본 cross(진로를 가로질러 걸어옴). "
                        "''를 주면 사람 없이 장애물만")
    p.add_argument("--player-speed", type=float, default=0.22)
    p.add_argument("--player-radius", type=float, default=0.30)
    p.add_argument("--styles", default="open_clusters,rooms",
                   help="수집에 쓸 맵 스타일. 기본은 사람과 마주칠 수 있는 두 종류. "
                        "좁은 통로 스타일은 장애물·사람이 자동 비활성되므로 제외")
    p.add_argument("--outdir", default="demos")
    p.add_argument("--selftest", action="store_true",
                   help="화면 없이 스크립트 조작자로 파이프라인 검증")
    args = p.parse_args()

    if args.selftest:
        run_selftest(args)
    else:
        run_collect(args)


if __name__ == "__main__":
    main()
