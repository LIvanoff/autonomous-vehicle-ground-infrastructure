"""Готовая визуализация фактического проезда, записанного стендом.

python visualize.py output/solution_crossing_slow.json
python visualize.py output/solution_crossing_slow.json --no-show
"""
import argparse
import json
from pathlib import Path
import textwrap


def make_figure(result):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    frames, rows = result["frames"], result["map"]
    fig = plt.figure(figsize=(10, 7.4), facecolor="#f8fafc")
    ax = fig.add_axes([.06, .16, .61, .73])
    ax.imshow([[c == "@" for c in row] for row in rows],
              cmap=ListedColormap(["#ffffff", "#c4ccd5"]), vmin=0, vmax=1,
              origin="upper", interpolation="nearest")
    ax.set_xlabel("x — столбец")
    ax.set_ylabel("y — строка")
    ax.set_xticks(range(0, len(rows[0]), max(1, len(rows[0])//8)))
    ax.set_yticks(range(0, len(rows), max(1, len(rows)//8)))
    fig.suptitle("Два робота в лабиринте", x=.06, ha="left", fontsize=19, fontweight="bold", color="#1e293b")
    fig.text(.06, .92, f'{result.get("map_name", "карта")}   |   сценарий: {result["fault"]}', fontsize=10, color="#475569")
    panel = fig.add_axes([.71, .17, .27, .70])
    panel.axis("off")
    marks, confirmed, trails, labels = {}, {}, {}, {}
    colors = {"R1": "#1877bf", "R2": "#e07b1b"}
    for robot, color in colors.items():
        path = result["routes"][robot]
        ax.plot([p[0] for p in path], [p[1] for p in path], "--", color=color, alpha=.35, linewidth=1.1)
        goal = result["goals"][robot]
        ax.plot(*goal, marker="*", markersize=15, color=color, markeredgecolor="white", zorder=4)
        trails[robot], = ax.plot([], [], color=color, linewidth=2, alpha=.7)
        confirmed[robot], = ax.plot([], [], "o", markersize=15, markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.3, zorder=5)
        marks[robot], = ax.plot([], [], "o", markersize=10, color=color, markeredgecolor="white", zorder=6)
        labels[robot] = ax.text(0, 0, robot, fontsize=10, color=color, weight="bold", zorder=7,
                                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 1})
    status = panel.text(0, 1, "", va="top", fontsize=12, fontweight="bold", color="#1e293b")
    info = panel.text(0, .87, "", va="top", fontsize=10, linespacing=1.65, color="#334155")
    panel.text(0, .12, "Кружок — фактическая позиция\nКольцо — подтверждённая\nЗвезда — цель робота\nПунктир — готовый план\nСплошная линия — проезд", va="top", fontsize=9, linespacing=1.5, color="#475569")
    footer = fig.text(.06, .065, "", fontsize=10, color="#475569")

    def update(index):
        frame = frames[index]
        for robot in colors:
            x, y = frame["physical"][robot]
            cx, cy = frame["confirmed"][robot]
            marks[robot].set_data([x], [y])
            confirmed[robot].set_data([cx], [cy])
            labels[robot].set_position((x+.55, y-.55))
            history = [f["physical"][robot] for f in frames[:index+1]]
            trails[robot].set_data([p[0] for p in history], [p[1] for p in history])
        status.set_text(f't = {frame["time"]:.2f} с\nШаг плана {frame["step"]} / {result["plan_steps"]}')
        lines = []
        for robot in colors:
            x, y = frame["physical"][robot]
            pos, target = tuple(frame["confirmed"][robot]), tuple(frame["targets"][robot])
            lines += [robot, f"Факт: ({x:.1f}, {y:.1f})", f"Подтверждено: {pos}", f"Следующая клетка: {target}", ""]
        waiting = ", ".join(frame["pending"]) or "—"
        lines += [f"Ждём ответ: {waiting}"]
        if frame["error"]:
            lines += ["", "ОШИБКА", textwrap.fill(frame["error"], width=29)]
        elif frame["finished"]:
            lines += ["", "Обе цели достигнуты"]
        info.set_text("\n".join(lines))
        footer.set_text("Одна команда → подтверждение → следующий выбор. Координаты берутся из записи исполнения.")
        return list(marks.values()) + list(confirmed.values()) + list(trails.values()) + list(labels.values()) + [status, info, footer]

    update(0)
    return fig, update


def render(result, output, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.widgets import Button, Slider

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = result["frames"]
    if not frames:
        raise ValueError("В журнале нет кадров фактического исполнения")
    fig, update = make_figure(result)

    # GIF сохраняет реальное время: выбираем ближайший кадр для каждой 0.1 с.
    first, last = frames[0]["time"], frames[-1]["time"]
    count = max(2, min(400, int((last-first)*10)+1))
    indices, j = [], 0
    for n in range(count):
        t = first + (last-first)*n/(count-1)
        while j+1 < len(frames) and frames[j+1]["time"] <= t:
            j += 1
        indices.append(j)
    indices[-1] = len(frames)-1
    indices += [len(frames)-1] * 10
    animation = FuncAnimation(fig, update, frames=indices, interval=100, repeat=False, blit=False)
    animation.save(output.with_suffix(".gif"), writer=PillowWriter(fps=10), dpi=90)
    # Превью показывает движение; ошибки — последний кадр с причиной.
    preview = len(frames)-1 if not result["success"] else len(frames)//2
    update(preview)
    fig.savefig(output.with_suffix(".png"), dpi=130)
    print("GIF:", output.with_suffix(".gif"))
    print("PNG:", output.with_suffix(".png"))
    plt.close(fig)

    if show:
        # В окне можно остановить запись и разобрать любой кадр.
        fig, update = make_figure(result)
        slider_ax = fig.add_axes([.16, .015, .49, .025])
        slider = Slider(slider_ax, "Кадр", 0, max(1, len(frames)-1), valinit=0, valstep=1)
        button_ax = fig.add_axes([.75, .025, .19, .045])
        button = Button(button_ax, "Пуск / Пауза")
        playing = [False]
        moving_slider = [False]

        def scrub(value):
            update(min(int(value), len(frames)-1))
            if not moving_slider[0]:
                playing[0] = False
            fig.canvas.draw_idle()

        def advance():
            if playing[0]:
                index = int(slider.val) + 1
                if index >= len(frames):
                    playing[0] = False
                else:
                    moving_slider[0] = True
                    slider.set_val(index)
                    moving_slider[0] = False

        def toggle(event):
            if int(slider.val) >= len(frames)-1:
                slider.set_val(0)
            playing[0] = not playing[0]

        slider.on_changed(scrub)
        button.on_clicked(toggle)
        timer = fig.canvas.new_timer(interval=50)
        timer.add_callback(advance)
        timer.start()
        plt.show()
        timer.stop()
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.log.read_text(encoding="utf-8"))
    render(data, args.log.with_suffix(""), show=not args.no_show)
