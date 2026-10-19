"""Готовый транспорт: локальный MQTT-брокер и три независимых клиента."""
import asyncio
from contextlib import contextmanager
import json
import logging
from pathlib import Path
from queue import Queue
import socket
import subprocess
import sys
import threading
import time


async def serve(port):
    from amqtt.broker import Broker
    logging.basicConfig(level=logging.ERROR)
    broker = Broker({
        "listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{port}"}},
        "plugins": {"amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True}},
    })
    await broker.start()
    try:
        await asyncio.Event().wait()
    finally:
        await broker.shutdown()


@contextmanager
def local_broker():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(port)], creationflags=flags)
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("MQTT-брокер не запустился")
            try:
                with socket.create_connection(("127.0.0.1", port), .1):
                    break
            except OSError:
                time.sleep(.05)
        else:
            raise TimeoutError("Запуск MQTT-брокера занял более 5 секунд")
        yield port
    finally:
        process.terminate()
        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(5)


class Endpoint:
    def __init__(self, name, subscription, port, prefix):
        import paho.mqtt.client as mqtt
        self.prefix = prefix
        self.messages = Queue()
        self.ready = threading.Event()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=prefix.replace("/", "-") + name)

        def connected(client, userdata, flags, reason_code, properties):
            if not reason_code.is_failure:
                client.subscribe(prefix + subscription, qos=1)

        def subscribed(client, userdata, mid, reason_codes, properties):
            if not any(code.is_failure for code in reason_codes):
                self.ready.set()

        def received(client, userdata, message):
            try:
                data = json.loads(message.payload)
                if isinstance(data, dict):
                    self.messages.put((message.topic[len(prefix):], data))
            except (ValueError, UnicodeError):
                pass

        self.client.on_connect = connected
        self.client.on_subscribe = subscribed
        self.client.on_message = received
        self.client.connect("127.0.0.1", port, keepalive=20)
        self.client.loop_start()
        if not self.ready.wait(5):
            self.close()
            raise TimeoutError("Клиент не получил подтверждения MQTT-подписки")

    def publish(self, topic, message):
        result = self.client.publish(self.prefix + topic, json.dumps(message), qos=1, retain=False)
        if result.rc != 0:
            raise ConnectionError("Не удалось отправить MQTT-сообщение")

    def drain(self):
        while not self.messages.empty():
            yield self.messages.get_nowait()

    def close(self):
        self.client.disconnect()
        self.client.loop_stop()


if __name__ == "__main__":
    asyncio.run(serve(int(sys.argv[1])))
