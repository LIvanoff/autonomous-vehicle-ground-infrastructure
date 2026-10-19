"""Готовая модель роботов. Не планирует и не исправляет диспетчера.

Одна команда = переход в соседнюю клетку. Робот затем стоит до новой команды.
Повтор команды с тем же mission_id/command_id не повторяет перемещение.
Недопустимая команда завершает эксперимент ошибкой, а не ставится в очередь.
"""
import copy


class World:
    def __init__(self, rows, starts, mission_id, publish, fault="normal", move_time=.04):
        self.rows = rows
        self.positions = {r: tuple(p) for r, p in starts.items()}
        self.mission_id = mission_id
        self.publish = publish
        self.fault = fault
        self.move_time = move_time
        self.active = {}
        self.commands = {}
        self.acks = {}
        self.latest_id = {r: 0 for r in starts}
        self.moves = {r: 0 for r in starts}
        self.fault_used = False
        self.silent_key = None
        self.error = None

    def fail(self, robot, command, reason):
        self.error = f"{robot}: {reason}"
        self.publish(robot, {"mission_id": self.mission_id,
                             "command_id": command.get("command_id"),
                             "status": "failed", "position": list(self.positions[robot]),
                             "reason": reason})

    def ready(self):
        for robot, position in self.positions.items():
            self.publish(robot, {"mission_id": self.mission_id, "command_id": 0,
                                 "status": "ready", "position": list(position)})

    def receive(self, robot, command, now):
        if self.error or command.get("mission_id") != self.mission_id:
            return
        command_id, target = command.get("command_id"), command.get("target")
        if type(command_id) is not int or not isinstance(target, list) or len(target) != 2 or any(type(v) is not int for v in target):
            self.fail(robot, command, "неверный формат команды")
            return
        key = robot, command_id
        if key in self.commands:
            if command != self.commands[key]:
                self.fail(robot, command, "тот же ID использован для другой команды")
            elif key in self.acks and key != self.silent_key:
                self.publish(robot, copy.deepcopy(self.acks[key]))
            return  # Повтор не перезапускает таймер движения.
        if command_id <= self.latest_id[robot] or robot in self.active:
            self.fail(robot, command, "старая команда или новая команда во время движения")
            return
        target = tuple(target)
        x, y = target
        origin = self.positions[robot]
        if not (0 <= y < len(self.rows) and 0 <= x < len(self.rows[0]) and self.rows[y][x] == "."):
            self.fail(robot, command, "движение в стену")
            return
        if abs(x-origin[0]) + abs(y-origin[1]) != 1:
            self.fail(robot, command, "нужна соседняя клетка")
            return
        for other, position in self.positions.items():
            if other != robot and (target == position or
                                   (other in self.active and target == self.active[other]["target"])):
                self.fail(robot, command, "конфликт: целевая клетка занята или другой робот уже движется в неё")
                return
        self.commands[key] = copy.deepcopy(command)
        self.latest_id[robot] = command_id
        duration = self.move_time * (8 if self.fault == "slow" and robot == "R2" else 1)
        self.active[robot] = {"target": target, "start": now, "until": now + duration, "key": key}

    def visual_positions(self, now):
        """Координаты для наблюдателя; диспетчер к ним доступа не получает."""
        positions = {r: list(p) for r, p in self.positions.items()}
        for robot, move in self.active.items():
            fraction = max(0, min(1, (now-move["start"]) / (move["until"]-move["start"])))
            a, b = self.positions[robot], move["target"]
            positions[robot] = [a[i] + (b[i]-a[i])*fraction for i in (0, 1)]
        return positions

    def tick(self, now):
        if self.error:
            return
        for robot, move in list(self.active.items()):
            if now < move["until"]:
                continue
            target, key = move["target"], move["key"]
            if any(other != robot and p == target for other, p in self.positions.items()):
                self.fail(robot, self.commands[key], "столкновение при завершении движения")
                return
            self.positions[robot] = target
            self.moves[robot] += 1
            del self.active[robot]
            ack = {"mission_id": self.mission_id, "command_id": key[1],
                   "status": "completed", "position": list(target)}
            self.acks[key] = ack
            if not self.fault_used and robot == "R2" and self.fault in {"lost_ack", "silence"}:
                self.fault_used = True
                if self.fault == "silence":
                    self.silent_key = key
                continue
            self.publish(robot, copy.deepcopy(ack))
            if self.fault == "duplicates":
                self.publish(robot, copy.deepcopy(ack))
