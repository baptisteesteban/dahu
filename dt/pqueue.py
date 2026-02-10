class PQueue:
    def __init__(self):
        self._queues = [[] for _ in range(256)]
        self._dist = 0
        self._size = 0

    def push(self, p: tuple[int, int], d: int):
        i = (self._dist + d) % 256
        self._queues[i].append(p)
        self._size += 1

    def advance(self):
        assert not self.empty()
        while len(self._queues[self._dist % 256]) == 0:
            self._dist += 1

    def pop(self) -> tuple[int, int]:
        assert not self.empty()
        self.advance()
        self._size -= 1
        return self._queues[self._dist % 256].pop()

    def empty(self) -> bool:
        return self._size == 0

    @property
    def distance(self) -> int:
        return self._dist
