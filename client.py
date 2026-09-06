"""
Agent Epistemic Aleatoric Uncertainty Decomposer Skill Client
Pure Python Standard Library implementation of uncertainty decomposition.
Decomposes Total Uncertainty = Aleatoric Uncertainty + Epistemic Uncertainty
using Shannon Entropy and Mutual Information over ensemble/temperature sample distributions.
"""

import math
from typing import List, Dict, Any, Tuple


class UncertaintyDecomposer:
    """
    Decomposes model predictive variance into:
    1. Aleatoric uncertainty: Expected conditional entropy (inherent noise).
    2. Epistemic uncertainty: Mutual information between predictions and model weights (reducible uncertainty).
    """

    @staticmethod
    def _entropy(probs: List[float]) -> float:
        """Compute Shannon entropy in nats."""
        ent = 0.0
        for p in probs:
            if p > 1e-12:
                ent -= p * math.log(p)
        return ent

    def decompose(self, sample_predictions: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Decompose ensemble or Monte Carlo sample distributions.
        :param sample_predictions: List of probability distributions across M samples/agents.
                                   e.g. [{"yes": 0.8, "no": 0.2}, {"yes": 0.6, "no": 0.4}]
        :return: Dict containing total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty.
        """
        if not sample_predictions:
            raise ValueError("Sample predictions list cannot be empty")

        num_samples = len(sample_predictions)
        all_classes = set()
        for sample in sample_predictions:
            all_classes.update(sample.keys())
        class_list = sorted(list(all_classes))

        # 1. Compute Mean Predictive Distribution: p_bar(c) = (1 / M) * sum(p_m(c))
        mean_distribution: Dict[str, float] = {}
        for c in class_list:
            mean_prob = sum(sample.get(c, 0.0) for sample in sample_predictions) / num_samples
            mean_distribution[c] = mean_prob

        # 2. Total Uncertainty: H(p_bar) = - sum(p_bar * log(p_bar))
        total_uncertainty = self._entropy(list(mean_distribution.values()))

        # 3. Aleatoric Uncertainty: E_m[H(p_m)] = (1 / M) * sum(H(p_m))
        sample_entropies = []
        for sample in sample_predictions:
            probs = [sample.get(c, 0.0) for c in class_list]
            sample_entropies.append(self._entropy(probs))
        aleatoric_uncertainty = sum(sample_entropies) / num_samples

        # 4. Epistemic Uncertainty: I(y; W) = H(p_bar) - E_m[H(p_m)]
        # Clip numerical precision drift at 0.0
        epistemic_uncertainty = max(0.0, total_uncertainty - aleatoric_uncertainty)

        # Recommendation based on dominant uncertainty type
        if epistemic_uncertainty > aleatoric_uncertainty and epistemic_uncertainty > 0.15:
            recommendation = "FETCH_MORE_CONTEXT"
            explanation = "High epistemic uncertainty detected. The agent lacks factual grounding or relevant knowledge."
        elif aleatoric_uncertainty > 0.3:
            recommendation = "ASK_CLARIFICATION"
            explanation = "High aleatoric uncertainty detected. The prompt or task contains inherent ambiguity."
        else:
            recommendation = "CONFIDENT_EXECUTION"
            explanation = "Low predictive variance. Safe to execute autonomously."

        return {
            "total_uncertainty": total_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "epistemic_ratio": (epistemic_uncertainty / total_uncertainty) if total_uncertainty > 1e-9 else 0.0,
            "mean_distribution": mean_distribution,
            "recommendation": recommendation,
            "explanation": explanation
        }
