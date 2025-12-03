from multiprocessing import Queue
import matplotlib.pyplot as plt
import numpy as np
import time
import signal

def plot_training(q: Queue, update_interval=0.01):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    plt.ion()
    fig, ax = plt.subplots()

    graph1, = ax.plot([], [], color="g")
    graph2 = ax.axhline(0, color="r")
    graph3, = ax.plot([], [], color="b")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training...")

    scores = []
    best_avg_reward = -float("inf")

    try:
        while plt.fignum_exists(fig.number):
            while not q.empty():
                scores, best_avg_reward = q.get_nowait()

            graph1.set_data([i for i in range(0, len(scores))], scores)
            graph2.set_ydata([best_avg_reward, best_avg_reward])
            graph3.set_data([0] + [i for i in range(100, len(scores) + 1, 100)] + [len(scores)], [np.float64(0)] + [np.mean(scores[i - 100:i]) for i in range(100, len(scores) + 1, 100)] + [np.mean(scores[-100:])])

            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            plt.pause(0.001)

            time.sleep(update_interval)
    except KeyboardInterrupt:
        pass
