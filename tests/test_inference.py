"""Tests de la etapa 3. Correr desde la raiz:  python -m pytest -q

- Los tests "puros" (candidatos, metricas) NO necesitan modelo.
- Los tests con modelo se saltan solos si models/qa-final/ no existe.
"""
import pytest

from src.evaluation.evaluate import aggregate, compute_exact, compute_f1, score_example
from src.inference.qa_inference import DEFAULT_MODEL_DIR, extract_candidates, sigmoid


# ---- contrato (reutilizable por el Integrante 4) -------------------------
def assert_contract(context: str, out: dict):
    assert set(out) == {"answer", "start", "score"}
    answer, start, score = out["answer"], out["start"], out["score"]
    if answer is None:
        assert start is None
    else:
        assert isinstance(start, int) and start >= 0
        assert context[start : start + len(answer)] == answer
    assert isinstance(score, (float, int)) and 0.0 <= score <= 1.0


# ---- extract_candidates --------------------------------------------------
OFFS = [None, None, (0, 3), (4, 8), (9, 12), None]  # tokens 2-4 son contexto


def best(cands):
    return max(cands, key=lambda c: c[0])


def test_candidates_pick_best_span_chars():
    sl = [0, 0, 1, 5, 1, 0]
    el = [0, 0, 1, 1, 6, 0]
    assert best(extract_candidates(sl, el, OFFS)) == (11.0, 4, 12)


def test_candidates_ignore_non_context_tokens():
    sl = [9, 0, 0, 0, 0, 0]  # el mejor logit esta en [CLS]
    el = [9, 0, 0, 0, 0, 9]  # y en el ultimo token especial
    cands = extract_candidates(sl, el, OFFS)
    assert cands  # si hay candidatos de contexto
    for _, a, b in cands:
        assert a in {0, 4, 9} and b in {3, 8, 12}  # solo offsets de tokens del contexto


def test_candidates_reject_end_before_start():
    sl = [0, 0, 0, 0, 9, 0]
    el = [0, 0, 0, 9, 0, 0]
    # start=4, end=3 es invalido: no debe aparecer como (9,8)
    assert (18.0, 9, 8) not in extract_candidates(sl, el, OFFS)


def test_candidates_respect_max_answer_len():
    offs = [None] + [(i, i + 1) for i in range(10)]
    sl = [0, 9] + [0] * 9
    el = [0] * 10 + [9]  # el span ideal (token 1 a token 10) mide 10 tokens
    cands = extract_candidates(sl, el, offs, max_answer_len=3)
    assert cands
    assert all(b - a <= 3 for _, a, b in cands)
    assert (18.0, 0, 10) not in cands


# ---- metricas ------------------------------------------------------------
def test_metrics_normalization():
    assert compute_exact("The Eiffel Tower", "eiffel tower") == 1
    assert compute_f1("16 attention heads", "16") == pytest.approx(0.5)


def test_metrics_no_answer():
    assert score_example([], None) == (1, 1.0)  # imposible y predijo null
    assert score_example([], "algo") == (0, 0.0)  # imposible y inventó
    assert score_example(["16"], None) == (0, 0.0)  # posible y dijo null


def test_aggregate_splits_hasans_noans():
    rows = [(True, 1, 1.0), (True, 0, 0.0), (False, 1, 1.0)]
    m = aggregate(rows)
    assert m["HasAns"]["n"] == 2 and m["NoAns"]["n"] == 1
    assert m["NoAns"]["F1"] == 100.0


def test_sigmoid_bounds():
    assert 0.0 <= sigmoid(-1e6) <= sigmoid(0) <= sigmoid(1e6) <= 1.0


# ---- con modelo real -----------------------------------------------------
needs_model = pytest.mark.skipif(not DEFAULT_MODEL_DIR.exists(), reason="falta models/qa-final/")

LEMPA = (
    "The Lempa River is the longest river in El Salvador, flowing about 422 km "
    "from Guatemala to the Pacific Ocean."
)


@pytest.fixture(scope="module")
def predictor():
    from src.inference.qa_inference import QAPredictor

    return QAPredictor()


@needs_model
def test_answerable_gives_expected_span(predictor):
    out = predictor.predict(LEMPA, "How long is the Lempa River?")
    assert_contract(LEMPA, out)
    assert out["answer"] is not None and "422" in out["answer"]


@needs_model
def test_unanswerable_gives_null(predictor):
    ctx = "Cats are small carnivorous mammals. They are often kept as pets."
    out = predictor.predict(ctx, "Who painted the Mona Lisa?")
    assert_contract(ctx, out)
    assert out["answer"] is None and out["start"] is None


@needs_model
def test_long_context_answer_at_end(predictor):
    filler = "The committee reviewed the budget and postponed the vote. " * 70  # ~4000 chars
    ctx = filler + "The secret access code is 7421."
    assert len(ctx) <= 5000
    out = predictor.predict(ctx, "What is the secret access code?")
    assert_contract(ctx, out)
    assert out["answer"] is not None and "7421" in out["answer"]
    assert out["start"] > 3000  # la respuesta esta al final: no se perdio en otra ventana


@needs_model
def test_non_ascii_and_newlines(predictor):
    ctx = "Café Münster opened in 1998.\nIt is located in San Salvador, El Salvador."
    out = predictor.predict(ctx, "When did Café Münster open?")
    assert_contract(ctx, out)
    assert out["answer"] is not None and "1998" in out["answer"]
    out2 = predictor.predict(ctx, "Where is Café Münster located?")
    assert_contract(ctx, out2)  # el start debe seguir siendo correcto despues del salto de linea
    assert out2["answer"] is not None and "San Salvador" in out2["answer"]
