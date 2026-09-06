"""
Example usage of Agent Epistemic Aleatoric Uncertainty Decomposer Skill.
"""

from client import UncertaintyDecomposer


def main():
    print("=== Agent Epistemic Aleatoric Uncertainty Decomposer Demonstration ===")
    decomposer = UncertaintyDecomposer()

    # Scenario 1: High Epistemic Uncertainty (Different models disagree completely on a niche fact)
    print("\n--- Scenario 1: Epistemic Uncertainty (Model Disagreement) ---")
    epistemic_samples = [
        {"option_a": 0.95, "option_b": 0.05},
        {"option_a": 0.05, "option_b": 0.95},
        {"option_a": 0.90, "option_b": 0.10},
        {"option_a": 0.10, "option_b": 0.90}
    ]
    res1 = decomposer.decompose(epistemic_samples)
    print("Total Uncertainty:", round(res1["total_uncertainty"], 4))
    print("Aleatoric (Noise):", round(res1["aleatoric_uncertainty"], 4))
    print("Epistemic (Knowledge Gap):", round(res1["epistemic_uncertainty"], 4))
    print("Epistemic Ratio:", f"{res1['epistemic_ratio'] * 100:.1f}%")
    print("Recommendation:", res1["recommendation"])

    # Scenario 2: High Aleatoric Uncertainty (Inherent coin-flip / Ambiguous prompt)
    print("\n--- Scenario 2: Aleatoric Uncertainty (Inherent Ambiguity) ---")
    aleatoric_samples = [
        {"option_a": 0.51, "option_b": 0.49},
        {"option_a": 0.49, "option_b": 0.51},
        {"option_a": 0.50, "option_b": 0.50},
        {"option_a": 0.52, "option_b": 0.48}
    ]
    res2 = decomposer.decompose(aleatoric_samples)
    print("Total Uncertainty:", round(res2["total_uncertainty"], 4))
    print("Aleatoric (Noise):", round(res2["aleatoric_uncertainty"], 4))
    print("Epistemic (Knowledge Gap):", round(res2["epistemic_uncertainty"], 4))
    print("Recommendation:", res2["recommendation"])

    # Scenario 3: High Confidence
    print("\n--- Scenario 3: Confident Prediction ---")
    confident_samples = [
        {"option_a": 0.98, "option_b": 0.02},
        {"option_a": 0.97, "option_b": 0.03},
        {"option_a": 0.99, "option_b": 0.01}
    ]
    res3 = decomposer.decompose(confident_samples)
    print("Total Uncertainty:", round(res3["total_uncertainty"], 4))
    print("Recommendation:", res3["recommendation"])


if __name__ == "__main__":
    main()
