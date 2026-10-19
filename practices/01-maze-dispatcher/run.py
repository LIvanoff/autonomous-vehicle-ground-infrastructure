"""python run.py --fault slow"""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import uuid

from mqtt_support import Endpoint, local_broker
from planner import ROOT, load_case, load_map, plan_routes
from simulator import World
from runtime import Controller


def load_policy(path):
    spec = importlib.util.spec_from_file_location("student_dispatcher", Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.choose_robot


def experiment(choose_robot, case_name="crossing", fault="normal", port=None,
               move_time=.08, timeout=30, retry_period=1, quiet=False):
    case = load_case(case_name)
    rows = load_map(ROOT / "maps" / case["map"])
    routes = plan_routes(rows, case["starts"], case["goals"])
    mission_id = uuid.uuid4().hex[:12]
    logs, clients, frames = [], [], []
    started = time.monotonic()

    def log(kind, robot, message):
        item = {"time": round(time.monotonic()-started, 3), "kind": kind,
                "robot": robot, "message": dict(message)}
        logs.append(item)
        if not quiet:
            pos = message.get("target", message.get("position"))
            print(f'{item["time"]:6.2f} {kind:7} {robot} id={message.get("command_id")} {pos}', flush=True)

    try:
        control = Endpoint("dispatcher", "robots/+/state", port, f"lab/{mission_id}/")
        clients.append(control)
        robots = {}
        for r in ("R1", "R2"):
            robots[r] = Endpoint(r, f"robots/{r}/command", port, f"lab/{mission_id}/")
            clients.append(robots[r])

        def robot_publish(robot, message):
            log(message["status"], robot, message)
            robots[robot].publish(f"robots/{robot}/state", message)

        world = World(rows, case["starts"], mission_id, robot_publish, fault, move_time)
        world.ready()
        ready = {}
        until = time.monotonic() + 5
        while len(ready) < 2 and time.monotonic() < until:
            for topic, message in control.drain():
                r = topic.split("/")[1]
                if r in robots and message.get("mission_id") == mission_id and message.get("status") == "ready":
                    ready[r] = tuple(message["position"])
            time.sleep(.002)
        if ready != {r: routes[r][0] for r in routes}:
            raise RuntimeError("Нет согласованных начальных состояний от обоих роботов")

        def command_publish(topic, message):
            log("command", topic.split("/")[1], message)
            control.publish(topic, message)

        dispatcher = Controller(routes, mission_id, command_publish, choose_robot)

        def record(now):
            frames.append({"time": round(now-started, 3),
                           "physical": world.visual_positions(now),
                           "confirmed": dict(dispatcher.positions),
                           "targets": dispatcher.targets(),
                           "pending": dict(dispatcher.pending),
                           "step": min(dispatcher.step, len(routes["R1"])-1),
                           "decision": dispatcher.decision,
                           "finished": dispatcher.finished,
                           "error": world.error or dispatcher.error})

        record(time.monotonic())
        dispatcher.dispatch()
        deadline = time.monotonic() + timeout
        retry_at = time.monotonic() + retry_period
        frame_at = time.monotonic()
        idle_since = None
        while time.monotonic() < deadline and not dispatcher.finished and not dispatcher.error and not world.error:
            now = time.monotonic()
            for robot, endpoint in robots.items():
                for topic, message in endpoint.drain():
                    world.receive(robot, message, now)
            world.tick(now)
            for topic, message in control.drain():
                robot = topic.split("/")[1]
                if robot in robots:
                    log("receive", robot, message)
                    dispatcher.on_state(robot, message)
            if now >= retry_at:
                dispatcher.retry_pending()
                retry_at = now + retry_period
            if not dispatcher.pending and not dispatcher.finished and not dispatcher.error:
                if idle_since is None:
                    idle_since = now
                if now-idle_since >= 2:
                    dispatcher.error = "Функция выбрала None: движение не началось за 2 секунды"
            else:
                idle_since = None
            if now >= frame_at:
                record(now)
                frame_at = now + .05
            time.sleep(.002)

        record(time.monotonic())

        actual_goals = world.positions == {r: tuple(p) for r, p in case["goals"].items()}
        success = dispatcher.finished and actual_goals and not world.error and not dispatcher.error
        return {"success": bool(success), "case": case_name, "fault": fault,
                "error": world.error or dispatcher.error, "timed_out": not dispatcher.finished and not world.error and not dispatcher.error,
                "positions": world.positions, "confirmed": dispatcher.positions,
                "pending": dispatcher.pending, "moves": world.moves, "routes": routes,
                "logs": logs, "frames": frames, "map": rows, "map_name": case["map"], "goals": case["goals"],
                "plan_steps": len(routes["R1"])-1}
    finally:
        for client in reversed(clients):
            client.close()


def main():
    parser = argparse.ArgumentParser(description="Два робота, готовый CBS и MQTT-диспетчер")
    parser.add_argument("--case", default="crossing")
    parser.add_argument("--fault", choices=["normal", "slow", "lost_ack", "duplicates", "silence"], default="normal")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="сохранить GIF/PNG без окна matplotlib")
    args = parser.parse_args()
    path = ROOT / "dispatcher.py"
    with local_broker() as port:
        result = experiment(load_policy(path), args.case, args.fault, port, timeout=args.timeout, quiet=args.quiet)
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    name = f"student_{args.case}_{args.fault}"
    (output / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["success", "case", "fault", "error", "positions", "moves", "plan_steps"]}, ensure_ascii=False, indent=2))
    print("Log:", output / f"{name}.json")
    from visualize import render
    render(result, output / name, show=not args.no_show)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
