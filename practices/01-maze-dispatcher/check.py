"""Четыре коротких примера без запуска MQTT. python check.py"""
from dispatcher import choose_robot

examples = [
    ("R1 ждёт освобождения клетки R2",
     {"R1": (0, 0), "R2": (1, 0)}, {"R1": (1, 0), "R2": (2, 0)}, {"R2"}),
    ("R2 ждёт освобождения клетки R1",
     {"R1": (1, 0), "R2": (2, 0)}, {"R1": (0, 0), "R2": (1, 0)}, {"R1"}),
    ("Оба могут двигаться — можно выбрать любого",
     {"R1": (0, 0), "R2": (3, 0)}, {"R1": (0, 1), "R2": (3, 1)}, {"R1", "R2"}),
    ("Оба уже стоят в нужных клетках",
     {"R1": (0, 0), "R2": (3, 0)}, {"R1": (0, 0), "R2": (3, 0)}, {None}),
]

if __name__ == "__main__":
    passed = 0
    for name, positions, targets, expected in examples:
        answer = choose_robot(positions, targets)
        ok = answer in expected
        passed += ok
        print(f'{"OK" if ok else "FAIL"}: {name}; ответ: {answer}')
    print(f"Пройдено {passed} из {len(examples)}")
    raise SystemExit(0 if passed == len(examples) else 1)
