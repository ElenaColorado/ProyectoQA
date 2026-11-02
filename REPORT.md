# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `team-02` |
| Members | Elena Colorado (Member 1), David Aguilar (Member 2), Emilio Martinez (Member 3), Juan Pablo Quintanilla (Member 4) |
| Repository URL | https://github.com/ElenaColorado/ProyectoQA |
| `submission` commit hash | Tag `submission` (run `git rev-parse submission`; the SHA is also sent to the instructor with the link) |
| Public service URL used at the dry run | https://oblivious-latitude-luckless.ngrok-free.dev (ngrok; backup: Cloudflare Quick Tunnel, whose URL changes on every restart) |
| AI tools used (which, and for what) | Antigravity AI Assistant (code generation, debugging, and documentation) Claude (Anthropic): drafted and debugged the Member 3 code (src/inference/, src/evaluation/, tests/test_inference.py); Member 3 ran, tested and reviewed it. Claude (Anthropic) also helped Member 4 draft `app.py`, its tests and the latency script, and with troubleshooting; Member 4 ran and checked them on his own laptop. |

## 1. Model choice and justification

- **Base checkpoint** (exact Hugging Face id): `distilbert/distilbert-base-uncased`
- **Alternatives considered** (at least one) and why you did not choose them:
  - `bert-base-uncased`: Has 110M parameters (66% more than DistilBERT), requiring significantly higher GPU VRAM during fine-tuning and introducing higher inference latency (~2x) on standard laptop CPUs.
  - `roberta-base`: Employs byte-pair encoding (BPE) and larger intermediate representations, exceeding laptop memory limits during prolonged multi-worker serving.
  - Pre-fine-tuned models (`distilbert-base-cased-distilled-squad`): Explicitly prohibited by project rules requiring real student fine-tuning from a general base checkpoint.
- **Trade-off:** DistilBERT retains approximately 97% of BERT's language comprehension capabilities while being 40% smaller (66.36M parameters vs 110M) and 60% faster in inference. This trade-off is optimal for meeting the strict latency budget (<3.0s per request) when serving extractive QA on a CPU-only laptop.

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | Full SQuAD 2.0 train set, preprocessed into 131,754 sliding windows |
| Validation data (what you validated on, how many examples) | Full SQuAD 2.0 validation set, preprocessed into 12,134 sliding windows |
| Epochs | 2 |
| Learning rate and schedule (warmup?) | 3e-5 with AdamW optimizer and linear decay schedule |
| Batch size | 8 per device with gradient accumulation steps = 2 (effective batch size = 16) |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM), FP16 mixed precision, 3,434.73 seconds (~57 minutes) |

- **Loss curve:** [`logs/loss.png`](logs/loss.png) (Backup pilot curve: [`logs/pilot/loss_pilot.png`](logs/pilot/loss_pilot.png)).
- **Validation results:**
  - Initial untuned loss: ~5.07
  - Pilot training (20k subset) eval loss: 1.7067
  - Full training (131,754 windows) eval loss: **1.2619** (optimal checkpoint selected at step 8,235 with eval loss 1.2613)
  - Train loss final: **1.2074** (gap with eval loss is 0.0545, confirming healthy convergence without overfitting)
  - Exact Match (EM) and F1 validation: 1,000 held-out SQuAD 2.0 validation examples: EM 59.0, F1 63.25 (HasAns F1 70.13, NoAns F1 55.36). Source: `logs/inference_validation_metrics.json`.

## 3. No-answer threshold
Decision rule. Each question/context pair is split into windows (384 tokens, stride 128). For every window the model gives start and end logits. We compare two scores:

null_score = the minimum over windows of start_logit[CLS] + end_logit[CLS].
best_score = the maximum over all windows of start_logit[s] + end_logit[e], using only context tokens, e >= s, spans of at most 30 tokens and the top 20 start/end candidates.

We compute diff = null_score - best_score and return null when diff > threshold. A higher threshold therefore gives fewer null answers. The returned score is a heuristic confidence (a sigmoid of diff - threshold), not a calibrated probability.

How the threshold was chosen. We swept thresholds from -10 to 10 in steps of 0.25 on 2,000 random examples of the SQuAD 2.0 validation set (seed 42). The best F1 on that set was at -1.0 (EM 64.85, F1 67.55). However, the validation set has about 50% unanswerable questions and the hidden test about 25% (according to the course guide), so we re-weighted the HasAns and NoAns F1 of the same sweep 75/25. The weighted F1 is almost flat between thresholds 0.5 and 2.5 (69.03 to 69.73, a spread of less than one point), and the maximum (2.5) gets that small gain by letting NoAns F1 fall to 44.17. We therefore chose threshold = 1.0, inside the plateau on its conservative side: weighted F1 69.04 (0.69 below the maximum), HasAns F1 73.64 and NoAns F1 55.24. This is a judgment call, not the arg-max. We did not tune it on the five public samples.

Held-out check. On 1,000 other validation examples (not used for the sweep) with threshold 1.0: EM 59.0 and F1 63.25 overall; HasAns (534 examples) EM 62.17 and F1 70.13; NoAns (466 examples) EM and F1 55.36. These numbers are slightly optimistic because the trainer used the same validation set to select its best checkpoint.

Too low or too high. At -2.0 the model says null too often: HasAns F1 drops to 56.09 while NoAns F1 rises to 77.38. At 3.5 it answers almost everything: HasAns F1 rises to 79.45 but NoAns F1 falls to 34.48, which means it invents answers for impossible questions.
## 4. Serving architecture

- **Framework and server:** FastAPI served by Uvicorn, 1 worker, port 8000 (`app.py`).
- **Model loading:** once, when the process starts (FastAPI lifespan, in a background thread). `QAPredictor` (Member 3) loads the tokenizer and weights from `models/qa-final/` with `local_files_only=True`, then one warm-up prediction runs. Until then `GET /health` returns 503 `{"status":"loading"}`; afterwards 200 `{"status":"ok"}`. The `params` value of `/info` is computed from the loaded model.
- **Request flow:** the `POST /predict` body is validated by a Pydantic model (strict strings, `id` 1–64, `context` 1–5000, `question` 1–500 characters, extra fields forbidden), then the token is checked, then `predict(context, question)` runs. Long contexts are split into 384-token windows with stride 128; the best span over all windows is compared with the null score using threshold 1.0 (Section 3). Before returning 200, `app.py` checks that `context[start:start+len(answer)] == answer`, that `start` is `null` only when `answer` is `null`, and that `score` is in [0,1]; otherwise it returns 500 `internal_error` instead of hiding the bug. The request `id` is echoed unchanged. A lock allows one inference at a time.
- **Access control:** `/info` and `/predict` use a dependency that compares the whole `Authorization` header with `Bearer <QA_API_TOKEN>` (`hmac.compare_digest`, case-sensitive). The token comes from the `QA_API_TOKEN` environment variable; the app refuses to start without it, and it is never in the code or the repository. A wrong or missing token gives 401 `{"error":"unauthorized"}`. Invalid bodies give 400 `{"error":"bad_request","detail":...}` through a custom handler (not FastAPI's default format).
- **Tunnel:** ngrok, `ngrok http 127.0.0.1:8000`. Backup: `cloudflared tunnel --url http://localhost:8000`. The five checks of the guide were run through the public HTTPS URL and are saved in `logs/tunnel_checks.txt` (no secrets); latency through the tunnel is in `logs/latency_results.json`.
- **How to run** from a clean machine:

```powershell
python -m pip install -r requirements.txt
# put the model files in models/qa-final/ (shared folder, not in Git)
$env:QA_API_TOKEN = "<token from the instructor>"
$env:QA_TEAM_ID = "team-02"
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
ngrok http 127.0.0.1:8000
```


## 5. Results on the public sample set

| Item | Expected | Your answer | start | Correct? |
|---|---|---|---|---|
| sample-01 | 16 | 16 | 383 | Yes |
| sample-02 | 3.5 days | 3.5 days | 437 | Yes |
| sample-03 | null | null | null | Yes |
| sample-04 | null | 3 | 310 | No |
| sample-05 | 50,257 | 50,257 | 3157 | Yes |

**4 of 5 correct.** The start values match the reference offsets, and sample-05 (answer near the end of a context longer than 512 tokens) is found, so the sliding windows and offset mapping work.

**Sample-04 failure analysis:** The question asks how much LoRA reduces GPU memory when fine-tuning BERT-base, but the context only reports that figure for GPT-3 175B. The model extracted "3" (character 310), the memory-reduction figure in that sentence. We did not run an ablation, but a likely explanation is that the wording of the question matches that sentence and nothing in an extractive encoder checks that the model named in the question (BERT-base) is the one in the text (GPT-3 175B).

From the returned score (0.755) we infer that its null-minus-best-span gap was about -0.13 (this value is derived, it is not stored in the log), so it would only turn into null with a threshold below about -0.13. A threshold of -0.25 would have fixed this case, but according to our sweep it would cost about 1.6 points of weighted F1 on validation (69.04 to 67.39), so we kept 1.0 instead of overfitting to one public sample.


## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | 253.17 MB (`models/qa-final/model.safetensors`) |
| Mean latency per `/predict` request on your laptop (CPU) | 0.361 s through the ngrok tunnel (50 requests, 0 failures, p95 0.738 s, max 0.809 s, 18.04 s in total; `logs/latency_results.json`); 0.126 s for the model alone without network (Member 3, 25 calls, `logs/cpu_latency.json`) |
| Laptop CPU and RAM | Intel Core i7-9750H @ 2.60 GHz (6 cores, 12 threads), 15.8 GB RAM |

## 7. Problems encountered and how you fixed them

At least two concrete problems. For each: what happened, the cause, the fix, and the commit where it was fixed.

1. **Problem:** `Trainer.__init__()` raised `TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'`.  
   **Cause:** Upgraded `transformers==5.2.0` deprecated the `tokenizer` parameter in favor of `processing_class`.  
   **Fix:** Added adaptive instantiation in `src/training/train.py` checking for `processing_class` support first with a fallback to `tokenizer` for compatibility across transformer versions.  
   **Commit:** `f995d6b`

2. **Problem:** CUDA Out-Of-Memory (OOM) error on laptop NVIDIA RTX 3050 (4 GB VRAM) when attempting batch size 16 with sequence length 384.  
   **Cause:** 384 tokens with attention matrices and forward/backward activations exceeded the 4 GB VRAM limit at batch size 16.  
   **Fix:** Reduced per-device batch size to 8 and configured `gradient_accumulation_steps=2`, simulating an effective batch size of 16 without increasing peak VRAM footprint, along with FP16 mixed precision.  
   **Commit:** `f995d6b`

3. **Problem:** running the whole test suite (`python -m pytest -q`) made 5 tests of `tests/test_public_sample.py` fail, with answers such as `'2018.'` or `'3'` and a constant score of 0.9, while the same file passed 9 of 10 tests when run alone.
   **Cause:** `tests/test_api_contract.py` set `QA_USE_MOCK=1` with `os.environ.update(...)`, and that value stayed in the environment for the rest of the pytest process. When the public-sample tests started the API, it loaded the fake development predictor instead of the real model.
   **Fix:** the fixture now uses `pytest.MonkeyPatch` and restores the environment when its tests finish. After the fix the full suite shows only the known `sample-04` case (Section 5).
   **Commit:** `ae25c63`

## 8. Decoder investigation

Problem 1: false answer when the question names an entity the text does not support (sample-04). Our encoder scores spans by how well they match the question but has no step that checks every detail of the question against the text. A decoder-only or encoder-decoder model (GPT-style, T5-style) can read the whole question and the context and then generate "the text does not say this for BERT-base", which handles this kind of near-miss better. The cost is high: generation is token by token, so latency grows with answer length, and a model that does this reliably is much larger than our 66M-parameter DistilBERT (T5-base alone has about 220M, roughly 3 times more). On a CPU laptop with a limit of about 3 seconds per question, this is a serious risk.

Problem 2: the threshold trade-off (NoAns F1 55.36 vs HasAns F1 70.13 on held-out data). With one score gap we must trade answering against abstaining. A generative model could abstain from an explicit reading of the question, but it would also be more likely to invent a fluent answer that is not in the text.

Evaluation cost. The API contract requires context[start:start+len(answer)] == answer. Our extractive model satisfies this by construction. A generated answer can paraphrase, so it would need extra alignment against the context, and some answers would fail the contract even when they are semantically right. Evaluation with EM/F1 would also be less predictable.

Conclusion. For this contract (exact span, start offset, about 3 s per question on CPU), the extractive encoder is the better fit. A decoder would help most for answers that are not a contiguous span or for yes/no questions, which our system does not handle (see Section 9).

## 9. Known limitations

The service is extractive only: it returns a span copied from the given context or null, and it cannot synthesize, summarize or combine information across passages.

- Impossible questions: NoAns F1 is only 55.36 on held-out data, so the model sometimes answers questions the text does not support. On public sample-04 (an impossible question) it returns "3" with score 0.755 instead of null. We did not tune the threshold (1.0) to fix this single case, to avoid overfitting to five examples.
- Language and size: English only. Requests with context over 5000 characters, question over 500 characters, or id over 64 characters are rejected with a 400/422 error in the contract format.
- Concurrency: one worker and one lock, so inferences run one at a time. Under heavy parallel load requests would queue and could exceed the 5 s limit.
- Score: the 0..1 score is a heuristic confidence, not a calibrated probability.
- Availability: inference runs on one laptop (CPU) behind a public tunnel; if the laptop sleeps, loses network or the tunnel restarts, the service is unavailable and the free ngrok URL may change.

