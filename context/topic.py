from collections import defaultdict
import math
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict

class TopicManager():
    def __init__(self, context_manager):
        self.context_manager = context_manager
        self.current_topics = {}
        

    def add_current_topic(self, topic_points:Dict[str, int]):
        to_del = []
        for k in self.current_topics.keys():
            self.current_topics[k] -= 1
            if self.current_topics[k] <= 0:
                to_del.append(k)
                print(f"[Context Manager] Removing topic '{k}' from current topics due to low relevance.")
        for k in to_del:
            del self.current_topics[k]

        for topic, points in topic_points.items():
            self.current_topics[topic] = points

    def dump_chunks(self, chunks:List[str], topic_gravity:Dict[str, float]):
        print("\n[Context Manager] Dumping chunks and their topic gravity:")
        for i, chunk in enumerate(chunks):
            gravity_info = ", ".join(f"{topic}: {gravity:.3f}" for topic, gravity in topic_gravity.items())
            print(f"  Chunk {i+1}: '{chunk[:50]}...' | Gravity: {gravity_info}")

    def identify_topic(
        self,
        content: str,
        min_words: int = 5,
        chunk_size: int = 24,
        chunk_overlap: int = 12,
        similarity_k: int = 4,
        per_chunk_min_gravity: float = 1.5,
        temperature: float = 0.6,
    ) -> Dict[str, int]:
        """
        Identify topics via chunked retrieval with dominance-ratio voting.
        
        Core insight: real signals dominate runner-ups by 3x+; noise queries don't.
        No baseline, no cliff detection, no qualifying floors — just dominance.
        """

        # ---- Stage 1: Conditional chunking ----
        if len(content.split()) < min_words:
            chunks = [content]
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[" ", ""],
            )
            chunks = splitter.split_text(content)
            if content not in chunks:
                chunks.append(content)
        
        

        if not chunks:
            return {}

        # ---- Stage 2: Per-chunk retrieval with rank-weighted aggregation ----
        topic_gravity: Dict[str, float] = defaultdict(float)
        chunks_used = 0

        for chunk in chunks:
            vector = self.context_manager._get_embedding(chunk)
            results = self.context_manager.topic_vectors.similarity_search_with_score_by_vector(
                vector, k=similarity_k
            )

            chunk_gravity: Dict[str, float] = defaultdict(float)
            for doc, score in results:
                topic = doc.metadata.get("topic")
                chunk_gravity[topic] += 1 / (score + 1)

            if not chunk_gravity:
                continue

            max_gravity = max(chunk_gravity.values())
            if max_gravity < per_chunk_min_gravity:
                continue

            exps = {t: math.exp((g - max_gravity) / temperature)
                    for t, g in chunk_gravity.items()}
            z = sum(exps.values())  # partition function

            chunks_used += 1
            for topic, e in exps.items():
                topic_gravity[topic] += (e / z) * max_gravity
        
        CHUNKS_USED_MIN = 2
        HIGH_CONFIDENCE_GRAVITY = 2.5  # tune from your data
        PEER_FLOOR = 0.20            # topics above this are real peers
        ABSOLUTE_FLOOR = 1.0         # topics must also exceed this absolute gravity

        if chunks_used < CHUNKS_USED_MIN and len(chunks) > 1:
            # Trust a single chunk if it produced a strong signal
            if not topic_gravity or max(topic_gravity.values()) < HIGH_CONFIDENCE_GRAVITY:
                return self._print_election(content, [], len(chunks), chunks_used, {})
            # else fall through — high-confidence single chunk is allowed to speak
        if not topic_gravity:
            return self._print_election(content, [], 0, 0, {})

        sorted_topics = sorted(topic_gravity.items(), key=lambda x: x[1], reverse=True)
        top_gravity = sorted_topics[0][1]
        
        normalized = [(topic, g / top_gravity) for topic, g in sorted_topics]
        
        topic_points = {sorted_topics[0][0]: 3}
        points_remaining = 2
        
        for topic, fraction in normalized[1:]:
            if fraction >= PEER_FLOOR and topic_gravity[topic] >= ABSOLUTE_FLOOR:
                # Real peer — award decreasing points
                topic_points[topic] = points_remaining
                points_remaining -= 1
                if points_remaining <= 0:
                    break
            else:
                # Either gray zone (0.20-0.40) or noise (<0.20) — stop awarding
                break
        return self._print_election(content, sorted_topics, len(chunks), chunks_used, topic_points)

    def _print_election(
        self,
        content: str,
        sorted_topics: List[Tuple[str, float]],
        n_chunks: int,
        chunks_used: int,
        topic_points: Dict[str, int],
    ) -> Dict[str, int]:
        """Pretty-print the election result and return topic_points."""
        BOX_WIDTH = 66
        BAR_WIDTH = 20

        input_preview = content[:60].replace("\n", " ") + ("…" if len(content) > 60 else "")

        if sorted_topics:
            top = sorted_topics[0][1]
            runner_up = sorted_topics[1][1] if len(sorted_topics) >= 2 else 0
            ratio = top / (runner_up + 1e-9) if runner_up > 0 else float('inf')
            ratio_str = f"{ratio:.2f}x" if ratio != float('inf') else "∞"
        else:
            top = 0
            ratio_str = "n/a"

        print(f"\n┌─ identify_topic {'─' * (BOX_WIDTH - 18)}┐")
        print(f"│ input    : {input_preview:<{BOX_WIDTH - 13}} │")
        print(f"│ chunks   : {n_chunks} (used: {chunks_used})    "
            f"topics: {len(sorted_topics)}    "
            f"dominance: {ratio_str:<8}      │")
        print(f"├{'─' * BOX_WIDTH}┤")
        print(f"│ {'topic':<26} {'gravity':>8}  {'bar':<{BAR_WIDTH}}  {'pts':>3}    │")
        print(f"│ {'─' * 26} {'─' * 8}  {'─' * BAR_WIDTH}  {'─' * 3}    │")

        bar_max = top if top > 0 else 1
        for topic, gravity in sorted_topics[:8]:
            filled = int((gravity / bar_max) * BAR_WIDTH)
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)
            pts = topic_points.get(topic, 0)
            pts_str = str(pts) if pts else " "
            marker = "◄" if pts == max(topic_points.values(), default=0) and pts > 0 else " "
            topic_display = topic[:26]
            print(f"│ {topic_display:<26} {gravity:>8.3f}  {bar}  {pts_str:>3} {marker}  │")

        if len(sorted_topics) > 8:
            omitted = f"… {len(sorted_topics) - 8} more topics omitted"
            print(f"│ {omitted:<{BOX_WIDTH - 2}} │")

        result_str = str(topic_points) if topic_points else "(rejected — insufficient dominance)"
        print(f"├{'─' * BOX_WIDTH}┤")
        print(f"│ result   : {result_str[:BOX_WIDTH - 14]:<{BOX_WIDTH - 13}} │")
        print(f"└{'─' * BOX_WIDTH}┘")

        return topic_points
    