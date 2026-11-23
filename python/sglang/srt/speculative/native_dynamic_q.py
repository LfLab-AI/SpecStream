"""Online uniform proposal width for the native STANDALONE worker."""

from collections import defaultdict


class NativeDynamicWidth:
    widths = (8, 4, 6, 2)

    def __init__(self):
        self.buckets = defaultdict(dict)
        self.round_counts = defaultdict(int)
        self.selected = {}

    @staticmethod
    def bucket(batch_size):
        return 1 << (batch_size - 1).bit_length()

    def choose(self, batch_size):
        key = self.bucket(batch_size)
        samples = self.buckets[key]
        self.round_counts[key] += 1
        for width in self.widths:
            if width not in samples:
                return width
        count = self.round_counts[key]
        if count % 16 == 0:
            return self.widths[(count // 16) % len(self.widths)]
        current = self.selected.get(key, 8)
        best = min(self.widths, key=samples.__getitem__)
        if samples[best] <= 0.92 * samples[current]:
            current = best
        self.selected[key] = current
        return current

    def update(self, batch_size, width, elapsed_ms, committed_draft):
        samples = self.buckets[self.bucket(batch_size)]
        cost = elapsed_ms / (batch_size + committed_draft)
        prior = samples.get(width)
        samples[width] = cost if prior is None else 0.8 * prior + 0.2 * cost
