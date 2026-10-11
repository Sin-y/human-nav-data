import random
from collections import deque


def _blank_walled(width, height):
    return [['#'] * width for _ in range(height)]


def _open_interior(grid):
    h, w = len(grid), len(grid[0])
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            grid[y][x] = '.'


def _carve_h(grid, x1, x2, y):
    for x in range(min(x1, x2), max(x1, x2) + 1):
        grid[y][x] = '.'


def _carve_v(grid, y1, y2, x):
    for y in range(min(y1, y2), max(y1, y2) + 1):
        grid[y][x] = '.'


def _carve_corridor(grid, rng, x1, y1, x2, y2):
    if rng.random() < 0.5:
        _carve_h(grid, x1, x2, y1)
        _carve_v(grid, y1, y2, x2)
    else:
        _carve_v(grid, y1, y2, x1)
        _carve_h(grid, x1, x2, y2)


# 스타일 1: 개방형 + 장애물 군집
def _style_open_clusters(grid, rng, density=0.10):
    h, w = len(grid), len(grid[0])
    _open_interior(grid)
    n_clusters = max(1, int((w * h) * density / 6))
    for _ in range(n_clusters):
        cx, cy = rng.randint(2, w - 3), rng.randint(2, h - 3)
        size = rng.randint(2, 5)
        cells = [(cx, cy)]
        frontier = [(cx, cy)]
        while len(cells) < size and frontier:
            fx, fy = frontier.pop(rng.randrange(len(frontier)))
            for dx, dy in rng.sample([(1, 0), (-1, 0), (0, 1), (0, -1)], 4):
                nx, ny = fx + dx, fy + dy
                if 1 <= nx < w - 1 and 1 <= ny < h - 1 and (nx, ny) not in cells:
                    cells.append((nx, ny))
                    frontier.append((nx, ny))
                if len(cells) >= size:
                    break
        for (x, y) in cells:
            grid[y][x] = '#'


# 스타일 2: 굽이치는 좁은 통로
def _style_winding_corridor(grid, rng):
    h, w = len(grid), len(grid[0])
    corridor_width = 1 if min(w, h) < 14 else rng.choice([1, 1, 2])
    cx = w // 2
    for y in range(1, h - 1):
        step = rng.choice([-1, 0, 0, 0, 1])
        cx = max(1 + corridor_width, min(w - 2 - corridor_width, cx + step))
        for dx in range(-corridor_width, corridor_width + 1):
            grid[y][cx + dx] = '.'


# 스타일 3: 방 + 복도\
def _style_rooms(grid, rng):
    h, w = len(grid), len(grid[0])
    n_rooms = rng.randint(3, 6)
    rooms = []
    for _ in range(n_rooms):
        rw = rng.randint(3, max(3, min(7, w - 4)))
        rh = rng.randint(3, max(3, min(6, h - 4)))
        rx = rng.randint(1, max(1, w - 2 - rw))
        ry = rng.randint(1, max(1, h - 2 - rh))
        for yy in range(ry, min(ry + rh, h - 1)):
            for xx in range(rx, min(rx + rw, w - 1)):
                grid[yy][xx] = '.'
        rooms.append((rx + rw // 2, ry + rh // 2))
    for i in range(len(rooms) - 1):
        x1, y1 = rooms[i]
        x2, y2 = rooms[i + 1]
        _carve_corridor(grid, rng, x1, y1, x2, y2)
    if len(rooms) > 2 and rng.random() < 0.6:
        a, b = rng.sample(rooms, 2)
        _carve_corridor(grid, rng, a[0], a[1], b[0], b[1])


# 스타일 4: 성긴 미로
def _style_sparse_maze(grid, rng, extra_opening_rate=0.12):
    h, w = len(grid), len(grid[0])
    ncols = (w - 2 - 1) // 2 + 1
    nrows = (h - 2 - 1) // 2 + 1
    visited = [[False] * ncols for _ in range(nrows)]

    def cell_pos(cx, cy):
        return (1 + 2 * cx, 1 + 2 * cy)

    def carve(cx, cy):
        visited[cy][cx] = True
        x, y = cell_pos(cx, cy)
        grid[y][x] = '.'
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        rng.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < ncols and 0 <= ny < nrows and not visited[ny][nx]:
                nx_pos, ny_pos = cell_pos(nx, ny)
                if nx_pos < w - 1 and ny_pos < h - 1:
                    grid[y + dy][x + dx] = '.'
                    carve(nx, ny)

    carve(0, 0)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if grid[y][x] == '#' and rng.random() < extra_opening_rate:
                if grid[y][x - 1] == '.' and grid[y][x + 1] == '.':
                    grid[y][x] = '.'
                elif grid[y - 1][x] == '.' and grid[y + 1][x] == '.':
                    grid[y][x] = '.'

    # 마지막 행/열이 격자 크기와 안 맞아 남는 경우 마저 개방
    last_row, last_col = h - 2, w - 2
    for x in range(1, w - 1):
        if grid[last_row][x] == '#' and grid[last_row - 1][x] == '.':
            grid[last_row][x] = '.'
    for y in range(1, h - 1):
        if grid[y][last_col] == '#' and grid[y][last_col - 1] == '.':
            grid[y][last_col] = '.'


STYLES = ["open_clusters", "winding_corridor", "rooms", "sparse_maze"]


def _bfs(grid, start):
    h, w = len(grid), len(grid[0])
    dist = {start: 0}
    q = deque([start])
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny][nx] != '#' and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(cx, cy)] + 1
                q.append((nx, ny))
    return dist


def generate_random_layout(width=None, height=None, seed=None, style=None):
    """
    랜덤 맵 1개를 생성해서 List[str] (#, ., S, G) 형식으로 반환.
    - width/height 미입력 시 12~30 / 10~22 범위에서 무작위 설정
    - style 미입력 시 4가지 스타일 중 무작위 설정
    - 항상 해결 가능한 맵 보장
    """
    rng = random.Random(seed)
    width = width or rng.randint(12, 30)
    height = height or rng.randint(10, 22)
    # 너무 작으면 스타일 알고리즘이 깨질 수 있어 최소 크기 보정
    width = max(width, 9)
    height = max(height, 8)

    style = style or rng.choice(STYLES)
    grid = _blank_walled(width, height)

    if style == "open_clusters":
        _style_open_clusters(grid, rng)
    elif style == "winding_corridor":
        _style_winding_corridor(grid, rng)
    elif style == "rooms":
        _style_rooms(grid, rng)
    elif style == "sparse_maze":
        _style_sparse_maze(grid, rng)
    else:
        raise ValueError(f"unknown style: {style}")

    # 극단적 실패 케이스 방어
    open_cells = [(x, y) for y in range(height) for x in range(width) if grid[y][x] == '.']
    if not open_cells:
        # 폴백
        grid = _blank_walled(width, height)
        _style_open_clusters(grid, rng, density=0.05)
        open_cells = [(x, y) for y in range(height) for x in range(width) if grid[y][x] == '.']

    anchor = rng.choice(open_cells)
    d1 = _bfs(grid, anchor)
    a = max(d1, key=d1.get)          # anchor에서 가장 먼 지점
    d2 = _bfs(grid, a)
    b = max(d2, key=d2.get)          # a에서 가장 먼 지점 -> a와 b는 서로 100% 도달 가능

    grid[a[1]][a[0]] = 'S'
    grid[b[1]][b[0]] = 'G'

    return [''.join(row) for row in grid], style, d2[b]  # (layout, 사용된 스타일, 경로 길이)


def validate_layout(layout):
    # 맵이 기존 코드와 호환되는 포맷인지 + 사용 가능한지 검증"""
    height, width = len(layout), len(layout[0])
    assert all(len(r) == width for r in layout), "행 길이가 서로 다름"
    assert all(c == '#' for c in layout[0]) and all(c == '#' for c in layout[-1]), "상/하 테두리가 벽이 아님"
    assert all(r[0] == '#' and r[-1] == '#' for r in layout), "좌/우 테두리가 벽이 아님"
    assert sum(r.count('S') for r in layout) == 1, "S가 정확히 1개가 아님"
    assert sum(r.count('G') for r in layout) == 1, "G가 정확히 1개가 아님"

    grid = [list(r) for r in layout]
    s = g = None
    for y, row in enumerate(layout):
        for x, ch in enumerate(row):
            if ch == 'S':
                s = (x, y)
            if ch == 'G':
                g = (x, y)
    dist = _bfs(grid, s)
    assert g in dist, "S에서 G로 가는 경로가 없음"
    return dist[g]


if __name__ == "__main__":
    for style in STYLES:
        layout, used_style, path_len = generate_random_layout(width=16, height=12, seed=0, style=style)
        checked_len = validate_layout(layout)  # 여기서 예외가 안 나면 정상
        print(f"=== style={used_style}  (경로 길이={path_len}, 검증 결과={checked_len}) ===")
        for row in layout:
            print(row)
        print()

    print("완전 무작위 예시 (크기/스타일 모두 랜덤):")
    layout, used_style, path_len = generate_random_layout()
    validate_layout(layout)
    print(f"style={used_style}, size={len(layout[0])}x{len(layout)}, path_len={path_len}")
