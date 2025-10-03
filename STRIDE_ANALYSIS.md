# Threat Modeling Using the STRIDE Framework

## System Overview

System Components

- **Training**: Downloads MNIST, trains SimpleCNN, saves model.
- **Evaluation**: Loads model, computes accuracy/plots.
- **Inference**: Gradio web UI, users draw digits → prediction.
- **Dataset Loader**: Fetches MNIST from public source.


## Threat Analysis

### Spoofing

**Risk**: Medium

**Threats in AI Systems**:

- Fake Gradio front-end hosted elsewhere to steal user input.
- Adversarially drawn digits that trick classifier into mislabeling.
- Someone pretends to be the model server (if hosted remotely).

**Mitigation Strategies**:

- Keep inference local (or authenticate server if deployed remotely).
- Use CAPTCHA/rate-limit if exposed publicly.
- Adversarial input detection is overkill here, but could be noted.

### Tampering

**Risk**: High

**Threats in AI Systems**:

- Training set poisoning: attacker swaps MNIST source → corrupted data.
- Model file on disk replaced with malicious payload.
- Preprocessing code altered (e.g., normalize incorrectly).

**Mitigation Strategies**:

- Verify dataset checksums before training.
- Store models in read-only directory or sign them.
- Use version control for training pipeline.

### Repudiation
**Risk**: Low

**Threats in AI Systems**:

- A user claims “I never submitted that input” (hardly relevant in MNIST demo).
- Logs modified to hide tampering.

**Mitigation Strategies**:

- Simple append-only logs are fine.
- Blockchain etc. = overkill.

### Information Disclosure

**Risk**: Medium

**Threats in AI Systems**:

- Model inversion: attacker queries API to reconstruct digit shapes. (The dataset is public, so not sensitive here.)
- If reused with private data, risk escalates fast.
- Debug logs accidentally leak internal details.

**Mitigation Strategies**:

- Avoid exposing confidence scores in public API.
- Keep logs clean of sensitive info.
- If extended to real data → differential privacy or regularization.

### Denial of Service (DoS)

**Risk**: Medium-High if public

**Threats in AI Systems**:

- Flood Gradio with requests until system crashes.
- Large fake inputs cause memory exhaustion.

**Mitigation Strategies**:

- Add request size limits in Gradio.
- Deploy behind a reverse proxy with rate limiting if public.
- Auto-restart model server on crash.

### Elevation of Privilege

**Risk**: Low-Medium

**Threats in AI Systems**:

- Attacker injects Python code in model file → executes on load.
- Exploiting Gradio bugs to escape sandbox.

**Mitigation Strategies**:

- Never load untrusted models.
- Keep Gradio and dependencies patched.
- Run inference as a low-privilege user, not root.


# Conclusion

| STRIDE Category             | Example Threats                                                    | Likelihood  | Impact     | Risk Level | Notes / Mitigations                                                                                |
| --------------------------- | ------------------------------------------------------------------ | ----------- | ---------- | ---------- | -------------------------------------------------------------------------------------------------- |
| **Spoofing**                | Fake Gradio front-end to steal input; adversarially drawn digits   | Medium      | Low-Medium | **Medium** | Keep inference local; authenticate server if deployed; basic adversarial input checks if extended. |
| **Tampering**               | Poisoned MNIST dataset; model file replaced; preprocessing altered | High        | High       | **High**   | Verify dataset checksum; sign/lock models; version control pipeline.                               |
| **Repudiation**             | User denies sending input; logs modified                           | Low         | Low        | **Low**    | Append-only logs suffice; blockchain = overkill here.                                              |
| **Information Disclosure**  | Model inversion; logs leaking details                              | Medium      | Medium     | **Medium** | Don’t expose confidences publicly; sanitize logs; if real data, add differential privacy.          |
| **Denial of Service (DoS)** | Flood Gradio with requests; oversized inputs cause crash           | Medium-High | Medium     | **High**   | Rate limiting; request size caps; auto-restart service; deploy behind reverse proxy.               |
| **Elevation of Privilege**  | Malicious model payload executes on load; Gradio sandbox escape    | Low         | High       | **Medium** | Never load untrusted models; patch dependencies; run under restricted user.                        |


# Appendix: Disclaimer

This file is optimized for reading and used AI assistance to enhance clarity and structure.
