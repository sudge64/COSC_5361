import threading
import queue
import time
from typing import Callable, Any

class AsyncHand:
    def __init__(self, hand_factory: Callable[[], Any], delay: float = 0.1):
        self._queue = queue.Queue()
        self._delay = delay
        self._hand_factory = hand_factory
        self._stop_event = threading.Event()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def rock(self):
        self._queue.put(("rock", ()))

    def paper(self):
        self._queue.put(("paper", ()))

    def scissors(self):
        self._queue.put(("scissors", ()))

    def numeric_one(self):
        self._queue.put(("numeric_one", ()))

    def numeric_two(self):
        self._queue.put(("numeric_two", ()))

    def numeric_three(self):
        self._queue.put(("numeric_three", ()))

    def numeric_four(self):
        self._queue.put(("numeric_four", ()))

    def numeric_five(self):
        self._queue.put(("numeric_five", ()))

    def spiderman(self):
        self._queue.put(("spiderman", ()))

    def tea_time(self):
        self._queue.put(("tea_time", ()))

    def hang_time(self):
        self._queue.put(("hang_time", ()))

    def boy_scout(self):
        self._queue.put(("boy_scout", ()))

    def thumbs_up(self):
        self._queue.put(("thumbs_up", ()))

    def grab(self):
        self._queue.put(("grab", ()))

    def victory(self):
        self._queue.put(("victory", ()))

    def reset(self):
        self._queue.put(("reset", ()))

    def _worker(self):
        hand = self._hand_factory()

        while not self._stop_event.is_set():
            try:
                name, args = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                getattr(hand, name)(*args)
            except Exception as exc:
                import traceback
                print(f"[AsyncHand] error executing {name}: {exc}")
                traceback.print_exc()

            time.sleep(self._delay)

        try:
            hand.reset()
        except Exception:
            pass

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)

