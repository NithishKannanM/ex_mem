from dataclasses import dataclass


@dataclass
class BenchmarkMetrics:
    predictions: list[dict]

    def accuracy(self) -> float:
        correct = sum(1 for p in self.predictions if p["predicted"] == p["expected"])
        return correct / len(self.predictions)

    def false_conflict_rate(self) -> float:
        # predicted UPDATE/DELETE when truth was ADD/NOOP
        false_conflicts = sum(
            1 for p in self.predictions
            if p["predicted"] in ("UPDATE", "DELETE") and p["expected"] in ("ADD", "NOOP")
        )
        return false_conflicts / len(self.predictions)

    def miss_rate(self) -> float:
        # predicted ADD/NOOP when truth was UPDATE/DELETE
        misses = sum(
            1 for p in self.predictions
            if p["predicted"] in ("ADD", "NOOP") and p["expected"] in ("UPDATE", "DELETE")
        )
        return misses / len(self.predictions)

    def p50_latency(self) -> float:
        latencies = sorted(p["latency_ms"] for p in self.predictions)
        return latencies[len(latencies) // 2]

    def p95_latency(self) -> float:
        latencies = sorted(p["latency_ms"] for p in self.predictions)
        return latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]

    def per_category_accuracy(self) -> dict:
        cats: dict[str, list[bool]] = {}
        for p in self.predictions:
            cats.setdefault(p.get("category", "unknown"), []).append(p["predicted"] == p["expected"])
        return {c: round(sum(v) / len(v), 3) for c, v in cats.items()}

    def summary(self) -> dict:
        return {
            "accuracy": round(self.accuracy(), 3),
            "false_conflict_rate": round(self.false_conflict_rate(), 3),
            "miss_rate": round(self.miss_rate(), 3),
            "p50_latency_ms": round(self.p50_latency(), 2),
            "p95_latency_ms": round(self.p95_latency(), 2),
            "per_category_accuracy": self.per_category_accuracy(),
        }
