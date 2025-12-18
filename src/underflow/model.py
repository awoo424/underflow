import heapq
from typing import Callable, Protocol

import hnswlib
import numpy as np
import requests
from gensim.models import KeyedVectors  # type: ignore[import-untyped]
from platformdirs import user_cache_path
from stream_unzip import stream_unzip

GLOVE_ZIP_URL = "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.300d.zip"
GLOVE_TXT_FILE = (
    "wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
)
CACHE_DIR = user_cache_path(appname="underflow", ensure_exists=True)
MODEL_FILE = CACHE_DIR / "alpha_300.model"
INDEX_FILE = CACHE_DIR / "cosine_index.bin"


class ProgressCallback(Protocol):
    def __call__(
        self, *, total: int | None = ..., progress: int = ..., advance: int = ...
    ) -> None: ...


def ensure_model_ready(
    status_callback: Callable[[str], None],
    progress_callback: ProgressCallback,
) -> None:
    if MODEL_FILE.exists() and INDEX_FILE.exists():
        return

    status_callback("Downloading GloVe vectors...")
    response = requests.get(GLOVE_ZIP_URL, stream=True)
    response.raise_for_status()

    for file_name, file_size, unzipped_chunks in stream_unzip(
        response.iter_content(1024)
    ):
        fn = file_name.decode("utf-8")
        assert fn == GLOVE_TXT_FILE, f"Unexpected file: {fn}"
        progress_callback(total=file_size, progress=0)
        with open(CACHE_DIR / fn, "wb") as f:
            for chunk in unzipped_chunks:
                f.write(chunk)
                progress_callback(advance=len(chunk))

    txt_path = CACHE_DIR / GLOVE_TXT_FILE

    status_callback("Loading execution model...")
    progress_callback(total=None)
    wvs = KeyedVectors.load_word2vec_format(str(txt_path), binary=False, no_header=True)

    status_callback("Filtering non-alphabetical words...")
    new_index_to_key: list[str] = []
    new_key_to_index: dict[str, int] = {}
    new_vectors: list[np.ndarray] = []
    for ind, word in enumerate(wvs.index_to_key):
        if word.isalpha():
            new_key_to_index[word] = len(new_index_to_key)
            new_index_to_key.append(word)
            new_vectors.append(wvs.vectors[ind])

    wvs.index_to_key = new_index_to_key
    wvs.key_to_index = new_key_to_index
    wvs.vectors = np.array(new_vectors)

    status_callback("Saving optimised model...")
    wvs.save_word2vec_format(str(MODEL_FILE), binary=True)

    status_callback("Building HNSW index...")
    _get_hnsw_index(wvs, max_neighbours=Model.MAX_NEIGHBOURS)

    # Cleanup
    if txt_path.exists():
        txt_path.unlink()


def _get_hnsw_index(
    wvs: KeyedVectors, max_neighbours: int, ef: int = 50, ef_construction: int = 64
):
    index = hnswlib.Index(space="cosine", dim=wvs.vectors.shape[1])

    if INDEX_FILE.exists():
        index.load_index(str(INDEX_FILE))
    else:
        index.init_index(
            max_elements=wvs.vectors.shape[0],
            ef_construction=ef_construction,
            M=max_neighbours,
        )
        index.add_items(wvs.vectors)
        index.save_index(str(INDEX_FILE))

    index.set_ef(ef)
    return index


class PriorityQueue:
    """
    A priority queue that supports adding new items and updating
    the priority of existing items.

    Items are (key, priority) pairs. Tie-breaking is done
    alphabetically by key.
    """

    def __init__(self):
        self.heap = []  # The min-heap, stored as a list of entries
        self.entry_finder = {}  # Mapping from key -> entry
        self.REMOVED = -1  # Sentinel to mark removed tasks

    def add(self, key: int, priority: float):
        """
        Add a new key or update the priority of an existing key.

        Args:
            key (int): The unique key for the item.
            priority (float): The priority (lower numbers are higher priority).
        """
        if key in self.entry_finder:
            # If key already exists, mark the old entry as removed.
            # We can't remove it from the heap directly without O(n) cost.
            old_entry = self.entry_finder[key]
            old_entry[-1] = self.REMOVED  # Mark as removed

        # Add the new entry: [priority, key]
        # If priorities are equal, heapq will compare the keys
        entry = [priority, key]
        self.entry_finder[key] = entry
        heapq.heappush(self.heap, entry)

    def popmin(self) -> tuple[int, float]:
        """
        Remove and return the key with the minimum priority.

        Returns:
            tuple: A (key, priority) tuple.

        Raises:
            KeyError: if the priority queue is empty.
        """
        while self.heap:
            # Pop the smallest item from the heap
            priority, key = heapq.heappop(self.heap)

            if key is not self.REMOVED:
                # This is a valid entry
                del self.entry_finder[key]  # Remove from tracking dict
                return key, priority

        # If we get here, the heap was empty or only had removed tasks
        raise KeyError("pop from an empty priority queue")

    def peekmin(self) -> tuple[int, float]:
        """
        Return the key with the minimum priority without removing it.

        This will also clean any "removed" entries from the top
        of the heap.

        Returns:
            tuple: A (key, priority) tuple.

        Raises:
            KeyError: if the priority queue is empty.
        """
        # Clean up any <REMOVED> items from the top of the heap
        while self.heap:
            priority, key = self.heap[0]  # Look at the top item
            if key is self.REMOVED:
                heapq.heappop(self.heap)  # Remove stale entry
            else:
                return key, priority  # Found the valid minimum

        # If we get here, the queue is empty
        raise KeyError("peek at an empty priority queue")

    def is_empty(self) -> bool:
        """Check if the priority queue is empty."""
        # We must check entry_finder, as self.heap might contain <removed> tasks
        return len(self.entry_finder) == 0

    def __len__(self) -> int:
        """Return the number of valid items in the queue."""
        return len(self.entry_finder)


class Model:
    MAX_NEIGHBOURS = 64

    def __init__(self) -> None:
        if not MODEL_FILE.exists() or not INDEX_FILE.exists():
            raise RuntimeError("Model not found. Run ensure_model_ready() first.")

        self._wvs: KeyedVectors = KeyedVectors.load_word2vec_format(
            str(MODEL_FILE), binary=True
        )
        self._idx: hnswlib.Index = _get_hnsw_index(
            self._wvs, max_neighbours=self.MAX_NEIGHBOURS
        )

    def is_word(self, key: str) -> bool:
        return key in self._wvs

    def _reconstruct_path(self, came_from: dict[int, int], current: int) -> list[str]:
        if current not in came_from:
            return [self._wvs.index_to_key[current]]
        return self._reconstruct_path(came_from, came_from[current]) + [
            self._wvs.index_to_key[current]
        ]

    def bidirectional_search(
        self, start: str, goal: str, step_distance: float, shortest: bool
    ) -> list[str]:
        start_idx: int = self._wvs.key_to_index[start]
        goal_idx: int = self._wvs.key_to_index[goal]

        if start_idx == goal_idx:
            return [start]

        # --- Heuristics ---
        h_fwd = self._wvs.distances(goal) / step_distance  # Heuristic to GOAL
        h_bwd = self._wvs.distances(start) / step_distance  # Heuristic to START

        # --- Forward Search (start -> goal) ---
        open_set_fwd = PriorityQueue()
        open_set_fwd.add(start_idx, h_fwd[start_idx])
        came_from_fwd: dict[int, int] = {}
        steps_to_reach_fwd: dict[int, int] = {start_idx: 0}  # g_score

        # --- Backward Search (goal -> start) ---
        open_set_bwd = PriorityQueue()
        open_set_bwd.add(goal_idx, h_bwd[goal_idx])
        came_from_bwd: dict[int, int] = {}
        steps_to_reach_bwd: dict[int, int] = {goal_idx: 0}  # g_score

        # --- Shared Termination Data ---
        best_path_cost = float("inf")
        meeting_node: int | None = None

        # --- Helper Function for Expanding ---
        def expand_search_direction(
            open_set: PriorityQueue,
            came_from: dict[int, int],
            steps_to_reach: dict[int, int],
            other_steps_to_reach: dict[int, int],
            heuristic: np.ndarray,
        ) -> bool:
            """
            Expands one node from the given search direction.
            Returns True if the greedy (non-shortest) search should stop.
            """
            nonlocal best_path_cost, meeting_node

            current, f_score = open_set.popmin()

            if shortest and f_score > best_path_cost:
                return False  # This node is already worse than our best complete path

            # Check for intersection
            if current in other_steps_to_reach:
                cost = steps_to_reach[current] + other_steps_to_reach[current]
                if shortest:
                    if cost < best_path_cost:
                        best_path_cost = cost
                        meeting_node = current
                else:  # Greedy: first path is fine
                    meeting_node = current
                    return True  # Signal to stop immediately

            # Expand neighbors
            labels, distances = self._idx.knn_query(
                self._wvs[current],
                k=self.MAX_NEIGHBOURS,
            )
            neighbours = labels[0][
                (distances[0] < step_distance) & (labels[0] != current)
            ]

            for next_node in neighbours:
                steps_to_reach_next = steps_to_reach[current] + 1
                if next_node not in steps_to_reach or (
                    shortest and steps_to_reach_next < steps_to_reach[next_node]
                ):
                    came_from[next_node] = current
                    steps_to_reach[next_node] = steps_to_reach_next
                    f_score_new = steps_to_reach_next + heuristic[next_node]
                    priority = f_score_new if shortest else heuristic[next_node]
                    open_set.add(next_node, priority)

            return False  # Do not stop

        # --- Main Search Loop ---
        while not open_set_fwd.is_empty() and not open_set_bwd.is_empty():
            # A* Termination Check (only for shortest path)
            if shortest:
                _, min_f_fwd = open_set_fwd.peekmin()
                _, min_f_bwd = open_set_bwd.peekmin()

                # If the best nodes we *could* expand from both directions are
                # already worse than a complete path we've found, we can stop.
                if min_f_fwd >= best_path_cost and min_f_bwd >= best_path_cost:
                    break

            # Decide which direction to expand (expand from smaller f-score)
            if open_set_fwd.peekmin()[1] < open_set_bwd.peekmin()[1]:
                stop_greedy = expand_search_direction(
                    open_set_fwd,
                    came_from_fwd,
                    steps_to_reach_fwd,
                    steps_to_reach_bwd,
                    h_fwd,
                )
            else:
                stop_greedy = expand_search_direction(
                    open_set_bwd,
                    came_from_bwd,
                    steps_to_reach_bwd,
                    steps_to_reach_fwd,
                    h_bwd,
                )

            if stop_greedy:
                break

        # --- Loop finished, now reconstruct path ---

        if meeting_node is None:
            return []

        # Reconstruct both halves
        path_fwd = self._reconstruct_path(came_from_fwd, meeting_node)
        path_bwd = self._reconstruct_path(came_from_bwd, meeting_node)

        # Combine: [start, ..., meeting_node] + [..., goal]
        # (We slice [1:] to remove the duplicate meeting_node from the 2nd list)
        return path_fwd + list(reversed(path_bwd))[1:]

    def find_best_path(
        self, start: str, goal: str, step_distances: list[float]
    ) -> list[str]:
        # Binary search for the smallest step distance that connects start to goal
        sorted_distances = sorted(step_distances)
        low = 0
        high = len(sorted_distances) - 1
        best_path = None

        while low <= high:
            mid = (low + high) // 2
            path = self.bidirectional_search(
                start, goal, sorted_distances[mid], shortest=True
            )

            if path:
                best_path = path
                high = mid - 1  # Try to find a smaller valid step_distance
            else:
                low = mid + 1  # Increase step_distance to find a connection

        if best_path:
            return best_path

        raise RuntimeError(f"Got stuck traversing from {start} to {goal}")
