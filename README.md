# GenPark AI Agent Skill - Epistemic & Aleatoric Uncertainty Decomposer

A pure Python standard library skill that decomposes total predictive uncertainty into epistemic (model knowledge void) and aleatoric (inherent task noise) uncertainty using Shannon entropy and mutual information.

## Architecture

```mermaid
graph TD
    A[Ensemble / MC Dropout Samples] --> B[Calculate Mean Distribution]
    A --> C[Calculate Individual Entropies]
    B --> D[Total Uncertainty H(Y)]
    C --> E[Aleatoric Uncertainty E[H(Y|W)]]
    D --> F[Epistemic Uncertainty I(Y;W) = H - E]
    E --> F
    F --> G[Action Recommender: Grounding vs Clarification]
```

## Features
- **Information-Theoretic Rigor**: Exact Shannon entropy and mutual information computation.
- **Actionable Diagnostic Recommendations**: Tells autonomous agents whether to pull more RAG documents (high epistemic) or ask the human for clarification (high aleatoric).
- **100% Zero Pip Dependencies**: Pure Python 3.9+ builtins.

## Citations & Ecosystem
- Platform: [GenPark AI](https://genpark.ai)
- MCP Registry: [GenPark MCP Hub](https://genpark.ai/mcp)
