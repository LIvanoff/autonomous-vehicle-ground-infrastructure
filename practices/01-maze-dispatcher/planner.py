"""Готовая обёртка CBS. Студент не изменяет этот файл."""
from contextlib import redirect_stdout
from collections import deque
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
_old_path = sys.path[:]
try:
    sys.path.insert(0, str(ROOT / "vendor"))
    from cbs.cbs import CBS, Environment
finally:
    # Исходный модуль добавляет '../' в sys.path: локализуем этот эффект.
    sys.path[:] = _old_path


def load_map(path=None):
    path = path or ROOT / "maps/maze-32-32-2.map"
    lines = Path(path).read_text(encoding="ascii").splitlines()
    height, width = int(lines[1].split()[1]), int(lines[2].split()[1])
    rows = lines[4:]
    if lines[3] != "map" or len(rows) != height or any(len(r) != width for r in rows):
        raise ValueError("Неверный размер карты")
    if set("".join(rows)) - set(".@GOTSW"):
        raise ValueError("Неизвестный символ карты Moving AI")
    return ["".join("." if c in ".G" else "@" for c in row) for row in rows]


class LabEnvironment(Environment):
    """Лимиты поиска и проверка будущих ограничений в конечной клетке.

    Сам CBS и A* взяты из vendor без изменений. В исходном is_at_goal
    проверяются только координаты; здесь учитывается пребывание в цели.
    """
    def __init__(self, rows, agents, timeout=5, horizon=256):
        self.deadline = time.monotonic() + timeout
        self.horizon = horizon
        obstacles = {(x, y) for y, row in enumerate(rows) for x, c in enumerate(row) if c == "@"}
        # Точные расстояния до цели ускоряют готовый A* в извилистом лабиринте.
        free = {(x, y) for y, row in enumerate(rows) for x, c in enumerate(row) if c == "."}
        self.distances = {}
        for agent in agents:
            goal = tuple(agent["goal"])
            distance, queue = {goal: 0}, deque([goal])
            while queue:
                x, y = queue.popleft()
                for cell in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                    if cell in free and cell not in distance:
                        distance[cell] = distance[x, y] + 1
                        queue.append(cell)
            if tuple(agent["start"]) not in distance:
                raise ValueError("Цель недостижима даже для одного робота")
            self.distances[agent["name"]] = distance
        super().__init__([len(rows[0]), len(rows)], agents, obstacles)

    def admissible_heuristic(self, state, agent_name):
        return self.distances[agent_name].get((state.location.x, state.location.y), float("inf"))

    def check_time(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError("CBS: превышено время поиска; это не доказательство отсутствия пути")

    def get_neighbors(self, state):
        self.check_time()
        return super().get_neighbors(state) if state.time < self.horizon else []

    def is_at_goal(self, state, agent_name):
        self.check_time()
        if not super().is_at_goal(state, agent_name):
            return False
        return not any(c.time >= state.time and c.location == state.location
                       for c in self.constraints.vertex_constraints)

    def get_first_conflict(self, solution):
        self.check_time()
        return super().get_first_conflict(solution)


def validate_routes(rows, routes, starts, goals):
    if set(routes) != {"R1", "R2"} or len({len(p) for p in routes.values()}) != 1:
        raise ValueError("Ожидаются два маршрута одинаковой длины")
    for r, path in routes.items():
        if not path or path[0] != tuple(starts[r]) or path[-1] != tuple(goals[r]):
            raise ValueError("Неверные начало/конец маршрута")
        for x, y in path:
            if not (0 <= y < len(rows) and 0 <= x < len(rows[0]) and rows[y][x] == "."):
                raise ValueError("Маршрут проходит через стену")
        for a, b in zip(path, path[1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1:
                raise ValueError("Недопустимый переход")
    a, b = routes["R1"], routes["R2"]
    for t in range(len(a)):
        if a[t] == b[t] or (t and a[t] == b[t-1] and b[t] == a[t-1]):
            raise ValueError("Конфликт в плане")


def plan_routes(rows, starts, goals, timeout=5):
    if set(starts) != {"R1", "R2"} or set(goals) != set(starts):
        raise ValueError("Нужны старты и цели двух роботов R1/R2")
    if len(set(map(tuple, starts.values()))) < 2 or len(set(map(tuple, goals.values()))) < 2:
        raise ValueError("Старты и цели должны быть разными")
    for x, y in list(starts.values()) + list(goals.values()):
        if not (0 <= y < len(rows) and 0 <= x < len(rows[0]) and rows[y][x] == "."):
            raise ValueError("Старт или цель находится в стене")
    agents = [{"name": r, "start": starts[r], "goal": goals[r]} for r in starts]
    env = LabEnvironment(rows, agents, timeout=timeout)
    with redirect_stdout(io.StringIO()):
        result = CBS(env).search()
    if not result:
        raise ValueError("Не найден совместный план в заданном горизонте")
    routes = {r: [(s["x"], s["y"]) for s in path] for r, path in result.items()}
    length = max(map(len, routes.values()))
    for path in routes.values():
        path.extend([path[-1]] * (length - len(path)))
    validate_routes(rows, routes, starts, goals)
    return routes


def load_case(name="crossing"):
    cases = json.loads((ROOT / "data/cases.json").read_text(encoding="utf-8"))
    return cases[name]
