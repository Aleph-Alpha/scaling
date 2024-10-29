import pytest

from scaling.core.data.blended_dataset import weights_by_num_docs, weights_examples_proportional
from tests.core.utils import rounded_equal


@pytest.mark.short
@pytest.mark.parametrize(
    "num_docs",
    [
        [900, 1],
        [900, 900],
        [900, 50, 40, 100],
        [900, 50, 40, 100, 1],
        [25000, 200, 200],
    ],
)
def test_weight_by_num_docs_alpha_0_produces_equal_weights(num_docs):
    weights = weights_by_num_docs(num_docs, alpha=0.0)
    num_docs_weighted = [n * w for n, w in zip(num_docs, weights)]

    for i in range(1, len(num_docs_weighted)):
        assert rounded_equal(
            num_docs_weighted[0], num_docs_weighted[i], 4
        ), "alpha 0.0 does not produce equal weightings"


@pytest.mark.short
@pytest.mark.cpu
@pytest.mark.parametrize(
    "num_docs",
    [
        [900, 1],
        [900, 900],
        [900, 50, 40, 100],
        [900, 50, 40, 100, 1],
    ],
)
def test_weight_by_num_docs_alpha_1_produces_identy_weights(num_docs):
    unbiased_sample_probs = [i / sum(num_docs) for i in num_docs]

    weights = weights_by_num_docs(num_docs, alpha=1.0)
    num_docs_weighted = [n * w for n, w in zip(num_docs, weights)]
    biased_sample_probs = [i / sum(num_docs_weighted) for i in num_docs_weighted]

    for i in range(0, len(num_docs)):
        assert rounded_equal(
            unbiased_sample_probs[i], biased_sample_probs[i], 4
        ), "alpha 1.0 does not produce identity weightings"


@pytest.mark.short
@pytest.mark.parametrize(
    "num_docs",
    [
        [900, 1],
        [900, 50, 40, 100],
        [900, 50, 40, 100, 1],
    ],
)
@pytest.mark.parametrize("alpha_base", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
def test_weight_by_num_docs_alpha_produces_something_between_alpha0_and_alpha1(num_docs, alpha_base):
    alpha_compare = alpha_base - 0.1

    weights_1 = weights_by_num_docs(num_docs, alpha=1.0)

    weights_compare = weights_by_num_docs(num_docs, alpha=alpha_compare)
    weights_base = weights_by_num_docs(num_docs, alpha=alpha_base)

    for i in range(len(num_docs)):
        assert abs(weights_base[i] - weights_1[i]) < abs(
            weights_compare[i] - weights_1[i]
        ), "weights to not move in direction of unbiased distribution when alpha increases"


@pytest.mark.short
@pytest.mark.parametrize(
    "num_docs",
    [
        [900, 1],
        [900, 900],
        [900, 50, 40, 100],
        [900, 50, 40, 100, 1],
    ],
)
@pytest.mark.parametrize("alpha_base", [1.0, 2.0, 0.5])
def test_examples_proportional_mixing_vs_alpha(num_docs, alpha_base):
    # check both functions produce the same outputs when temperature is 1
    weights_ep = weights_examples_proportional(num_docs, temperature=1 / alpha_base, maximum=None)
    weights_alpha = weights_by_num_docs(num_docs, alpha_base)

    for i in range(0, len(num_docs)):
        assert rounded_equal(
            weights_ep[i], weights_alpha[i], 4
        ), "two sampling functions in this setting do not produce matching weights"


@pytest.mark.short
@pytest.mark.parametrize(
    "num_docs",
    [
        [900, 1],
        [900, 900],
        [900, 50, 40, 100],
        [900, 50, 40, 100, 10],
    ],
)
@pytest.mark.parametrize("maximum", [None, 0, 0.4, 4, 40, 400, 4000])
def test_examples_proportional_mixing_no_temperature_scaling(num_docs, maximum):
    weights = weights_examples_proportional(num_docs, temperature=1.0, maximum=maximum)
    num_docs_weighted = [n * w for n, w in zip(num_docs, (weights / weights.max()))]
    # check if datasets above maximum limit have equal weights to each other
    idx_datasets_above_max = []

    if maximum:
        for idx, doc_size in enumerate(num_docs):
            if doc_size > maximum:
                idx_datasets_above_max.append(idx)
        if len(idx_datasets_above_max) > 1:
            for i in range(0, len(idx_datasets_above_max)):
                assert rounded_equal(
                    num_docs_weighted[idx_datasets_above_max[0]],
                    num_docs_weighted[idx_datasets_above_max[i]],
                    4,
                ), "alpha 1.0 does not produce identity weightings"
    else:
        for i in range(0, len(num_docs)):
            assert rounded_equal(num_docs[i], num_docs_weighted[i], 4), "alpha 1.0 does not produce identity weightings"

    print(weights)


@pytest.mark.short
@pytest.mark.parametrize(
    "num_docs",
    [
        [900, 1],
        [900, 50, 40, 100],
        [900, 50, 40, 100, 1],
    ],
)
@pytest.mark.parametrize("maximum", [50, 400])
@pytest.mark.parametrize("temperature", [1.5, 2, 4])
def test_examples_proportional_mixing_with_temperature_scaling(num_docs, maximum, temperature):
    weights = weights_examples_proportional(num_docs, temperature=temperature, maximum=maximum)
    weights_compare = weights_examples_proportional(num_docs, temperature=temperature + 1, maximum=maximum)
    weights_1 = weights_examples_proportional(num_docs, temperature=1.0, maximum=maximum)
    # num_docs_weighted = [n * w for n, w in zip(num_docs, (weights / weights.max()))]

    for i in range(len(num_docs)):
        assert abs(weights[i] - weights_1[i]) < abs(
            weights_compare[i] - weights_1[i]
        ), "weights to not move in direction of unbiased distribution when temperature decreases"
    print(weights)
