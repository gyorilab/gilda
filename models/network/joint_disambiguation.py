"""
Joint Disambiguation module.

Uses Gilda to generate candidates and DeepWalk embeddings to ground
entity mentions with maximal semantic coherence.
"""

import pickle
import numpy as np
from itertools import product
import gilda

DEFAULT_MODEL = 'deepwalk_model_v1.pkl'


def cosine_sim(v1, v2):
    """Cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def default_embedding_key(term):
    """Default mapping from a Gilda Term to its embedding lookup key.

    Uses ``term.id`` for CHEBI and GO namespaces, and ``term.entry_name``
    otherwise.
    """
    if term.db in ('CHEBI', 'GO'):
        return term.id
    return term.entry_name


class JointDisambiguator:
    """Jointly disambiguate entity mentions using Gilda scores and
    embedding coherence.

    Parameters
    ----------
    model_path : str
        Path to the pickled KeyedVectors model.
    alpha : float, optional
        Weight balancing coherence vs. Gilda score. 0 = pure Gilda score,
        1 = pure embedding coherence. Default: 0.5.
    embedding_key_fn : callable, optional
        Function that takes a :class:`gilda.term.Term` and returns the
        string key used to look up its embedding vector. Defaults to
        :func:`default_embedding_key`.
    """

    def __init__(self, model_path=DEFAULT_MODEL, alpha=0.5, embedding_key_fn=None):
        self.model_path = model_path
        self.alpha = alpha
        self.embedding_key_fn = embedding_key_fn or default_embedding_key
        self._vocab = None
        self._vectors = None

    def _load(self):
        """Load the model from disk on first use."""
        if self._vocab is not None:
            return
        with open(self.model_path, 'rb') as f:
            kv = pickle.load(f)
        self._vocab = kv.__dict__.get('vocab', {})
        self._vectors = kv.vectors

    def get_vector(self, key):
        """Get the embedding vector for a key, or None if not found."""
        self._load()
        if key in self._vocab:
            idx = self._vocab[key].index
            return self._vectors[idx]
        return None

    def _score_assignment(self, assignment, resolved):
        """Compute the combined objective for a candidate assignment.

        Score = alpha * (avg pairwise cosine) + (1 - alpha) * (avg Gilda score)

        Parameters
        ----------
        assignment : list[tuple]
            Each element is (emb_key, scored_match, vec) for an ambiguous
            mention's candidate choice.
        resolved : dict[str, tuple]
            Mapping from resolved mention to (emb_key, gilda_score) pairs.
        """
        all_vecs = []
        all_gilda_scores = []

        for _, (key, gscore) in resolved.items():
            vec = self.get_vector(key)
            if vec is not None:
                all_vecs.append(vec)
                all_gilda_scores.append(gscore)

        for _, sm, vec in assignment:
            all_vecs.append(vec)
            all_gilda_scores.append(sm.score)

        n = len(all_vecs)
        avg_cosine = 0.0
        if n >= 2:
            total_sim = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    total_sim += cosine_sim(all_vecs[i], all_vecs[j])
                    count += 1
            avg_cosine = total_sim / count

        avg_gilda = np.mean(all_gilda_scores) if all_gilda_scores else 0.0
        return float(self.alpha * avg_cosine + (1 - self.alpha) * avg_gilda)

    def disambiguate(self, mentions):
        """Jointly disambiguate a list of entity mentions.

        For each mention, Gilda generates grounding candidates. Mentions
        with a single embeddable candidate are resolved immediately. For
        ambiguous mentions (multiple embeddable candidates), all
        combinations are evaluated to find the assignment that maximizes
        a weighted sum of average pairwise embedding similarity and
        average Gilda score.

        Parameters
        ----------
        mentions : list[str]
            Entity mention strings to ground.

        Returns
        -------
        results : dict[str, dict]
            Mapping from each mention to a dict with keys:

            - ``match``: the :class:`gilda.grounder.ScoredMatch` for the
              chosen grounding, or None if no candidates were found
            - ``key``: the embedding lookup key for the chosen grounding
            - ``ambiguous``: whether the mention had multiple embeddable
              candidates
        best_score : float or None
            The combined objective score for the best assignment, or
            None if there were no ambiguous mentions to optimize over.
        """
        resolved = {}
        ambiguous = {}

        for mention in mentions:
            scored_matches = gilda.ground(mention)
            embeddable = []

            for sm in scored_matches:
                emb_key = self.embedding_key_fn(sm.term)
                vec = self.get_vector(emb_key)
                if vec is not None:
                    embeddable.append((emb_key, sm, vec))

            if len(embeddable) == 0:
                if scored_matches:
                    m = scored_matches[0]
                    resolved[mention] = {
                        'match': m,
                        'key': self.embedding_key_fn(m.term),
                        'ambiguous': False,
                    }
                else:
                    resolved[mention] = {
                        'match': None,
                        'key': None,
                        'ambiguous': False,
                    }
            elif len(embeddable) == 1:
                emb_key, sm, _ = embeddable[0]
                resolved[mention] = {
                    'match': sm,
                    'key': emb_key,
                    'ambiguous': False,
                }
            else:
                ambiguous[mention] = embeddable

        best_score = None
        if ambiguous:
            ambiguous_mentions = list(ambiguous.keys())
            candidate_lists = [ambiguous[m] for m in ambiguous_mentions]

            resolved_for_scoring = {
                m: (r['key'], r['match'].score if r['match'] else 0.0)
                for m, r in resolved.items()
            }

            best_score = -float('inf')
            best_assignment = None

            for combo in product(*candidate_lists):
                s = self._score_assignment(combo, resolved_for_scoring)
                if s > best_score:
                    best_score = s
                    best_assignment = combo

            for mention, (emb_key, sm, _) in zip(ambiguous_mentions,
                                                  best_assignment):
                resolved[mention] = {
                    'match': sm,
                    'key': emb_key,
                    'ambiguous': True,
                }

        return resolved, best_score


def main():
    """Demo: jointly disambiguate a small set of mentions."""
    terms = ['ER', 'tamoxifen', 'PR']
    jd = JointDisambiguator()
    results, best_score = jd.disambiguate(terms)

    if best_score is not None:
        print(f"Best combined score: {best_score:.4f}\n")

    print("=== Final Joint Groundings ===")
    header = (f"{'Mention':<15} | {'Key':<20} | {'Database':<10} | "
              f"{'ID':<15} | {'Score'}")
    print(header)
    print("-" * len(header))
    for mention in terms:
        r = results[mention]
        m = r['match']
        db = m.term.db if m else None
        eid = m.term.id if m else None
        score = m.score if m else 0.0
        print(f"{mention:<15} | {str(r['key']):<20} | {str(db):<10} | "
              f"{str(eid):<15} | {score:.4f}")


if __name__ == "__main__":
    main()
