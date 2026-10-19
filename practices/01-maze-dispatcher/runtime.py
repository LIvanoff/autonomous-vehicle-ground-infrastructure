"""Готовый исполнитель: одна команда, затем подтверждение. Не задание студента."""


class Controller:
    def __init__(self, routes, mission_id, publish, choose_robot):
        self.routes = routes
        self.mission_id = mission_id
        self.publish = publish
        self.choose_robot = choose_robot
        self.positions = {r: path[0] for r, path in routes.items()}
        self.pending = {}
        self.step = 1
        self.finished = False
        self.error = None
        self.decision = None

    def targets(self):
        return {r: path[min(self.step, len(path)-1)] for r, path in self.routes.items()}

    def dispatch(self):
        if self.pending or self.finished or self.error:
            return
        length = len(self.routes["R1"])
        while self.step < length and self.positions == self.targets():
            self.step += 1
        if self.step == length:
            self.finished = True
            return
        try:
            robot = self.choose_robot(self.positions.copy(), self.targets())
        except Exception as exc:
            self.error = f"Ошибка choose_robot: {exc}"
            return
        self.decision = robot
        if robot is None:
            return
        if robot not in ("R1", "R2"):
            self.error = "choose_robot должна вернуть R1, R2 или None"
            return
        target = self.targets()[robot]
        if target == self.positions[robot]:
            self.error = f"{robot} уже находится в своей целевой клетке шага"
            return
        command = {"mission_id": self.mission_id, "command_id": self.step,
                   "target": list(target)}
        self.pending[robot] = command
        # Проверку занятой клетки выполняет студент. Ошибка провалит опыт.
        self.publish(f"robots/{robot}/command", command)

    def on_state(self, robot, message):
        command = self.pending.get(robot)
        if not command or message.get("mission_id") != self.mission_id:
            return
        if message.get("command_id") != command["command_id"]:
            return
        if message.get("status") == "failed":
            self.error = f"{robot}: {message.get('reason', 'ошибка робота')}"
            return
        if message.get("status") != "completed":
            return
        if message.get("position") != command["target"]:
            self.error = f"{robot}: подтверждена неожиданная позиция"
            return
        self.positions[robot] = tuple(message["position"])
        del self.pending[robot]
        self.dispatch()

    def retry_pending(self):
        if not self.error and not self.finished:
            for robot, command in self.pending.items():
                self.publish(f"robots/{robot}/command", command)
