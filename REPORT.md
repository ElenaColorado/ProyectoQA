# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `team-02` |
| Members | Elena Colorado (Member 1), Integrante 2 (Member 2), Emilio Martinez, Integrante 4 (Member 4) |
| Repository URL | https://github.com/ElenaColorado/ProyectoQA |
| `submission` commit hash | *To be filled by Member 4 upon final submission tag* |
| Public service URL used at the dry run | *To be filled by Member 4* |
| AI tools used (which, and for what) | Antigravity AI Assistant (code generation, debugging, and documentation) Claude (Anthropic): drafted and debugged the Member 3 code (src/inference/, src/evaluation/, tests/test_inference.py); Member 3 ran, tested and reviewed it.|

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
  - Exact Match (EM) and F1 validation: *To be measured in detail on answerable and unanswerable subsets by Member 3 during inference calibration*.

## 3. No-answer threshold
Decision rule. Each question/context pair is split into windows (384 tokens, stride 128). For every window the model gives start and end logits. We compare two scores:

null_score = the minimum over windows of start_logit[CLS] + end_logit[CLS].
best_score = the maximum over all windows of start_logit[s] + end_logit[e], using only context tokens, e >= s, spans of at most 30 tokens and the top 20 start/end candidates.

We compute diff = null_score - best_score and return null when diff > threshold. A higher threshold therefore gives fewer null answers. The returned score is a heuristic confidence (a sigmoid of diff - threshold), not a calibrated probability.

How the threshold was chosen. We swept thresholds from -10 to 10 in steps of 0.25 on 2,000 random examples of the SQuAD 2.0 validation set (seed 42). The best F1 on that set was at -1.0 (EM 64.85, F1 67.55). However, the validation set has about 50% unanswerable questions and the hidden test about 25% (according to the course guide), so we re-weighted the HasAns and NoAns F1 of the same sweep 75/25. The weighted F1 is almost flat between thresholds 0.5 and 2.5 (69.03 to 69.73, a spread of less than one point), and the maximum (2.5) gets that small gain by letting NoAns F1 fall to 44.17. We therefore chose threshold = 1.0, inside the plateau on its conservative side: weighted F1 69.04 (0.69 below the maximum), HasAns F1 73.64 and NoAns F1 55.24. This is a judgment call, not the arg-max. We did not tune it on the five public samples.

Held-out check. On 1,000 other validation examples (not used for the sweep) with threshold 1.0: EM 59.0 and F1 63.25 overall; HasAns (534 examples) EM 62.17 and F1 70.13; NoAns (466 examples) EM and F1 55.36. These numbers are slightly optimistic because the trainer used the same validation set to select its best checkpoint.

Too low or too high. At -2.0 the model says null too often: HasAns F1 drops to 56.09 while NoAns F1 rises to 77.38. At 3.5 it answers almost everything: HasAns F1 rises to 79.45 but NoAns F1 falls to 34.48, which means it invents answers for impossible questions.
## 4. Serving architecture

*(To be completed by Member 4 — FastAPI and Deployment)*

- **Framework and server** (e.g. FastAPI + uvicorn), number of workers.
- **Model loading:** when and where the model is loaded.
- **Request flow:** from the incoming `/predict` request to the returned span, including how long contexts are handled.
- **Access control:** where the token is checked and where the token value comes from.
- **Tunnel:** which tool you use and the command you run.
- **How to run** from a clean machine (commands and environment variables):

```bash
pip install -r requirements.txt
export QA_API_TOKEN=...
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 5. Results on the public sample set

Item	   | Expected|	Your answer|	start |	Correct?
sample-01|	16	   |   16	       |  383	  | Yes
sample-02| 3.5 days|	3.5 days	 |  437	  | Yes
sample-03| null	   |   null	     |  null	| Yes
sample-04| null	   |    3	       |  310	  | No
sample-05| 50,257  |	50,257	   |  3157	| Yes
4 of 5 correct. The start values match the reference offsets, and sample-05 (answer near the end of a context longer than 512 tokens) is found, so the sliding windows and offset mapping work.

sample-04 failed. The question asks how much LoRA reduces GPU memory when fine-tuning BERT-base, but the context only reports that figure for GPT-3 175B. The model extracted "3" (character 310), the memory-reduction figure in that sentence. We did not run an ablation, but a likely explanation is that the wording of the question matches that sentence and nothing in an extractive encoder checks that the model named in the question (BERT-base) is the one in the text (GPT-3 175B). From the returned score (0.755) we infer that its null-minus-best-span gap was about -0.13 (this value is derived, it is not stored in the log), so it would only turn into null with a threshold below about -0.13. A threshold of -0.25 would have fixed this case, but according to our sweep it would cost about 1.6 points of weighted F1 on validation (69.04 to 67.39), so we kept 1.0 instead of overfitting to one public sample.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | 253.17 MB (`models/qa-final/model.safetensors`) |
| Mean latency per `/predict` request on your laptop (CPU) | *(To be measured by Member 3/4 on target CPU)* |
| Laptop CPU and RAM | Intel Core i5 / AMD Ryzen, 16 GB RAM |

## 7. Problems encountered and how you fixed them

At least two concrete problems. For each: what happened, the cause, the fix, and the commit where it was fixed.

1. **Problem:** `Trainer.__init__()` raised `TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'`.  
   **Cause:** Upgraded `transformers==5.2.0` deprecated the `tokenizer` parameter in favor of `processing_class`.  
   **Fix:** Added adaptive instantiation in `src/training/train.py` checking for `processing_class` support first with a fallback to `tokenizer` for compatibility across transformer versions.  
   **Commit:** `[Stage 2 commit]`

2. **Problem:** CUDA Out-Of-Memory (OOM) error on laptop NVIDIA RTX 3050 (4 GB VRAM) when attempting batch size 16 with sequence length 384.  
   **Cause:** 384 tokens with attention matrices and forward/backward activations exceeded the 4 GB VRAM limit at batch size 16.  
   **Fix:** Reduced per-device batch size to 8 and configured `gradient_accumulation_steps=2`, simulating an effective batch size of 16 without increasing peak VRAM footprint, along with FP16 mixed precision.  
   **Commit:** `[Stage 2 commit]`
## 8. Decoder investigation

*(To be completed by Member 3 / Team)*

How could a decoder (GPT-style) or an encoder–decoder architecture address specific problems **your team** encountered (Section 7 or Section 5)? Examples: answers that are not a contiguous span, yes/no questions, long contexts.

For each point: the problem, how the alternative architecture would handle it, and what it would cost (latency, model size, evaluation).

## 9. Known limitations

*(To be completed by Team)*

What your service does **not** do or handle, and what would happen if it received such input.

