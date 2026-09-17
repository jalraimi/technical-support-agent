# Architectural Design Decision Record (DDR)

## 1. Intent Taxonomy & Separability
The system triages incoming support queries into 8 mutually exclusive categories to drive routing:
- **`authentication`**: Login failures, MFA resets, OAuth/SSO issues, session expirations.
- **`network`**: Timeouts, VPN-only routing issues, DNS failures, connection resets.
- **`deployment`**: Container startup errors, Docker exit codes, Kubernetes scheduling failures.
- **`database`**: Connection pool exhaustion, deadlocks, slow queries, replica lag.
- **`gpu`**: CUDA out-of-memory (OOM), driver mismatches, PyTorch device mapping errors.
- **`api`**: HTTP 4xx/5xx responses, malformed JSON schemas, payload validation failures.
- **`package`**: Dependency version resolution, conflicting requirements, library installations.
- **`general`**: Purely conceptual or instructional technical inquiries without active errors (e.g., "Explain what LoRA is").

These classes are chosen because each maps to a distinct operational diagnostic path: live health telemetry (`database`, `gpu`), log inspection (`deployment`, `api`), static docs lookup (`package`), or conceptual generation (`general`).

## 2. Classifier Confidence Threshold & Routing Policy
- **Threshold ($\tau = 0.70$):** If Model A (Sequence Classifier) predicts an intent with softmax probability $\ge 0.70$, the prediction is accepted.
- **Fallback Policy:** If confidence is $< 0.70$, or if the input query triggers hard safety keywords, the router falls back to Router B (LLM Router) for semantic triage.

## 3. When Extractive QA Wins Over Generative Synthesis
- **Model B (Extractive QA)** wins when the user asks for exact, documented configuration specifications, version requirements, or port numbers (e.g., "According to the docs...").
- Extractive QA eliminates hallucination risk by predicting exact token offsets from trusted documentation context rather than generating open-ended text.

## 4. LoRA / PEFT Configuration for Model C
- **Target Model:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **LoRA Parameters:**
  - Rank ($r$): 8
  - Alpha ($\alpha$): 16
  - Dropout: 0.05
  - Target Modules: `["q_proj", "v_proj"]`
  - Bias: `none`
- **Rationale:** Constraining rank to 8 focuses fine-tuning on domain-specific instruction alignment and tool output synthesis while maintaining a parameter-efficient footprint and preventing catastrophic forgetting.

## 5. Tool Access Policies
- **Automatic Execution:** `knowledge_base_search`, `ticket_search`, `ticket_create`, `system_health_check`, `log_analyzer`, `package_lookup`, `calculator`, and `diagnostic_runbook`.
- **Restricted / Escalated:**
  - `sql_query`: Restricted strictly to read-only `SELECT` queries with an allowlist.
  - `escalate_to_human`: Automatically triggered on destructive incident markers (e.g., data loss, disk corruption, security incident).