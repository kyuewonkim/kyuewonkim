import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import heapq

# =========================
# 기본 설정
# =========================
WIDTH = 30
HEIGHT = 30

WALL = 0
PATH = 1
START1 = 2
START2 = 3
START3 = 4
GOAL = 5


class Maze:
    def __init__(self, width=30, height=30):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)

        self.starts = []
        self.goal = None

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def carve_cell(self, x, y, value=PATH):
        if self.in_bounds(x, y):
            self.grid[y, x] = value

    def carve_h(self, x1, x2, y, value=PATH):
        if x1 > x2:
            x1, x2 = x2, x1
        for x in range(x1, x2 + 1):
            self.carve_cell(x, y, value)

    def carve_v(self, x, y1, y2, value=PATH):
        if y1 > y2:
            y1, y2 = y2, y1
        for y in range(y1, y2 + 1):
            self.carve_cell(x, y, value)

    def carve_polyline(self, points, value=PATH):
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            if x1 == x2:
                self.carve_v(x1, y1, y2, value)
            elif y1 == y2:
                self.carve_h(x1, x2, y1, value)
            else:
                raise ValueError("Only orthogonal polyline segments are allowed.")

    def set_start(self, x, y, idx):
        self.starts.append((x, y, idx))
        self.grid[y, x] = idx

    def set_goal(self, x, y):
        self.goal = (x, y)
        self.grid[y, x] = GOAL

    def walkable(self, x, y):
        if not self.in_bounds(x, y):
            return False
        return self.grid[y, x] in [PATH, START1, START2, START3, GOAL]

    def neighbors(self, x, y):
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if self.walkable(nx, ny):
                yield (nx, ny)


def build_one_maze_three_starts():
    """
    30x30 grid
    - Goal: 중심부 근처
    - Starts: 꼭지점 3개
    - 구조: outer ring + middle ring + inner ring + connectors + dead-ends
    """

    maze = Maze(WIDTH, HEIGHT)

    # -------------------------
    # 1) 외곽 ring
    # -------------------------
    maze.carve_polyline([
        (2, 2), (27, 2), (27, 27), (2, 27), (2, 2)
    ])

    # -------------------------
    # 2) 중간 ring
    # -------------------------
    maze.carve_polyline([
        (6, 6), (23, 6), (23, 23), (6, 23), (6, 6)
    ])

    # -------------------------
    # 3) 내부 ring
    # -------------------------
    maze.carve_polyline([
        (10, 10), (19, 10), (19, 19), (10, 19), (10, 10)
    ])

    # -------------------------
    # 4) ring 간 연결 복도
    # -------------------------
    connectors = [
        [(14, 2), (14, 6)],
        [(20, 6), (20, 10)],
        [(12, 19), (12, 23)],
        [(23, 14), (27, 14)],
        [(2, 16), (6, 16)],
        [(10, 12), (6, 12)],
        [(16, 23), (16, 27)],
        [(19, 16), (23, 16)],
        [(8, 6), (8, 10)],
        [(6, 20), (10, 20)],
    ]

    for line in connectors:
        maze.carve_polyline(line)

    # -------------------------
    # 5) 내부 탐색용 cross 구조
    # -------------------------
    maze.carve_polyline([(14, 10), (14, 19)])
    maze.carve_polyline([(10, 14), (19, 14)])

    # -------------------------
    # 6) dead-end / spur 추가
    #    탐색 시간을 늘리기 위한 가지
    # -------------------------
    spurs = [
    [(4, 2), (4, 0)],
    [(9, 2), (9, 4)],
    [(24, 2), (24, 0)],
    [(27, 8), (29, 8)],
    [(27, 20), (29, 20)],
    [(23, 24), (27, 24)],
    [(20, 27), (20, 29)],
    [(8, 27), (8, 29)],
    [(2, 22), (0, 22)],
    [(2, 8), (0, 8)],
    [(6, 9), (3, 9)],
    [(23, 11), (27, 11)],
    [(11, 23), (11, 27)],
    [(18, 6), (18, 3)],
    [(16, 19), (16, 22)],
    ]

    for line in spurs:
        maze.carve_polyline(line)

    # -------------------------
    # 7) Goal
    # 중앙보다 약간 offset
    # -------------------------
    maze.set_goal(16, 14)

    # -------------------------
    # 8) Start 3개
    # 꼭지점 4개 중 3개 사용
    # -------------------------
    maze.set_start(2, 2, START1)     # 좌상
    maze.set_start(27, 2, START2)    # 우상
    maze.set_start(2, 27, START3)    # 좌하

    return maze


def dijkstra(maze, start, goal):
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]

    while pq:
        cur_dist, cur = heapq.heappop(pq)

        if cur == goal:
            break

        if cur_dist > dist[cur]:
            continue

        for nxt in maze.neighbors(*cur):
            nd = cur_dist + 1
            if nxt not in dist or nd < dist[nxt]:
                dist[nxt] = nd
                prev[nxt] = cur
                heapq.heappush(pq, (nd, nxt))

    if goal not in dist:
        return None, []

    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()

    return dist[goal], path


def draw_maze(maze):
    cmap = ListedColormap([
        "black",       # WALL
        "white",       # PATH
        "limegreen",   # START1
        "deepskyblue", # START2
        "orange",      # START3
        "red"          # GOAL
    ])

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(maze.grid, cmap=cmap, origin="upper", vmin=0, vmax=5)

    # grid line
    ax.set_xticks(np.arange(-0.5, maze.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.height, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.35)

    ax.set_xticks(np.arange(0, maze.width, 2))
    ax.set_yticks(np.arange(0, maze.height, 2))
    ax.set_xlim(-0.5, maze.width - 0.5)
    ax.set_ylim(maze.height - 0.5, -0.5)

    # 각 start에서 goal까지 최단경로 표시
    colors = {
        START1: "limegreen",
        START2: "deepskyblue",
        START3: "orange",
    }

    for sx, sy, idx in maze.starts:
        dist, path = dijkstra(maze, (sx, sy), maze.goal)
        if path:
            xs = [x for x, y in path]
            ys = [y for x, y in path]
            ax.plot(xs, ys, color=colors[idx], linewidth=2.5, alpha=0.9,
                    label=f"Start {idx-1} path (d={dist})")

    ax.set_title("One Maze with 3 Starts and 1 Goal", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()


def print_distances(maze):
    print("=== Shortest path lengths ===")
    for sx, sy, idx in maze.starts:
        dist, _ = dijkstra(maze, (sx, sy), maze.goal)
        print(f"Start {idx-1} at ({sx},{sy}) -> Goal {maze.goal}: {dist}")


if __name__ == "__main__":
    maze = build_one_maze_three_starts()
    print_distances(maze)
    draw_maze(maze)
















