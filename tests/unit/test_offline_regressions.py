"""
Regression tests for bugs that can be checked without calling any LLM.
"""
import pytest
import numpy as np

from tinytroupe.experimentation.randomization import ABRandomizer
from tinytroupe.experimentation.statistical_tests import StatisticalTester, cohen_d
from tinytroupe.validation.simulation_validator import (
    SimulationExperimentDataset,
    SimulationExperimentEmpiricalValidator,
)
from tinytroupe.validation.propositions import _build_precondition_function_for_action_types
from tinytroupe.utils.json import merge_dicts, JsonSerializableRegistry
from tinytroupe.utils.llm import extract_json, LLMChat
from tinytroupe.environment import TinyWorld


def _run(test_type, control, treatment):
    tester = StatisticalTester({"c": {"m": control}}, {"t": {"m": treatment}})
    return tester.run_test(test_type=test_type)["t"]["m"]


def test_ab_randomizer_actually_randomizes():
    r = ABRandomizer(random_seed=42)
    choices = {r.randomize(i, "a", "b") for i in range(50)}
    assert choices == {("a", "b"), ("b", "a")}

    # reproducible for the same seed, and reversible
    r2 = ABRandomizer(random_seed=42)
    for i in range(50):
        shown = r2.randomize(i, "a", "b")
        assert r2.choices[i] == r.choices[i]
        assert r2.derandomize(i, *shown) == ("a", "b")


def test_mann_whitney_cles_counts_ties_and_ci_has_the_sign_of_the_difference():
    same = _run("mann_whitney", [3, 3, 3, 4, 4], [3, 3, 3, 4, 4])
    assert same["effect_size"] == pytest.approx(0.5)

    higher = _run("mann_whitney", [1, 2, 3, 4, 5], [11, 12, 13, 14, 15])
    assert higher["median_difference"] > 0
    lo, hi = higher["confidence_interval"]
    assert lo > 0 and hi > 0


def test_mann_whitney_with_single_values_does_not_crash():
    result = _run("mann_whitney", [0.3], [0.9])
    assert result["confidence_interval"] is None


def test_cohen_d_zero_variance():
    assert cohen_d([1, 1, 1], [1, 1, 1]) == 0.0
    assert cohen_d([1, 1, 1], [0, 0, 0]) == -np.inf
    assert np.isnan(cohen_d([0.1], [0.9]))


def test_percent_change_sign_follows_difference():
    result = _run("welch_t_test", [-10, -11, -9], [-5, -6, -4])
    assert result["mean_difference"] > 0
    assert result["percent_change"] > 0


def test_check_assumptions_finds_metric():
    tester = StatisticalTester({"c": {"m": [1, 2, 3, 4, 5]}}, {"t": {"m": [2, 3, 4, 5, 7]}})
    assert "t" in tester.check_assumptions("m")


def test_validator_converts_effect_sizes_to_a_common_scale():
    v = SimulationExperimentEmpiricalValidator()
    # Mann-Whitney CLES: 0.5 means no effect, 0 or 1 mean maximal effect
    assert v._extract_effect_size({"test_type": "Mann-Whitney U test", "effect_size": 0.5}) == 0
    assert abs(v._extract_effect_size({"test_type": "Mann-Whitney U test", "effect_size": 0.0})) == 1
    assert v._extract_effect_size({"test_type": "Welch t-test (unequal variance)", "effect_size": float("nan")}) is None
    assert v._extract_effect_size({"test_type": "Welch t-test (unequal variance)", "effect_size": 0.7}) == 0.7


def test_validator_encodes_categories_consistently_across_datasets():
    control = SimulationExperimentDataset(key_results={"answer": ["yes", "no", "maybe", "yes"]},
                                          data_types={"answer": "categorical"})
    treatment = SimulationExperimentDataset(key_results={"answer": ["yes", "no", "no", "yes"]},
                                            data_types={"answer": "categorical"})
    v = SimulationExperimentEmpiricalValidator()
    c, t = v._align_category_codes("answer", control, treatment,
                                   control.key_results["answer"], treatment.key_results["answer"])
    # same category, same code on both sides
    assert c[0] == t[0]  # yes
    assert c[1] == t[1]  # no


def test_proportions_given_as_percentages_are_scaled_together():
    ds = SimulationExperimentDataset(key_results={"p": [0.5, 1, 25, 80]}, data_types={"p": "proportion"})
    assert ds.key_results["p"] == [0.005, 0.01, 0.25, 0.8]


def test_action_type_precondition_checks_all_actions():
    not_done = _build_precondition_function_for_action_types(["DONE"], check_for_presence=False)
    assert not_done(None, None, {"action": [{"type": "DONE"}, {"type": "TALK"}]}) is True
    assert not_done(None, None, {"action": [{"type": "DONE"}]}) is False
    assert not_done(None, None, {}) is True


def test_merge_dicts_does_not_share_or_mutate_lists():
    fragment = {"interests": ["chess"]}
    a = merge_dicts({}, fragment)
    b = merge_dicts({}, fragment)
    a = merge_dicts(a, {"interests": ["surfing"]})
    assert b["interests"] == ["chess"]
    assert fragment["interests"] == ["chess"]
    assert merge_dicts({"x": [{"k": [1]}]}, {"x": [{"k": [1]}]})["x"] == [{"k": [1]}]


def test_from_json_keeps_plain_dicts_with_a_type_key():
    class _Holder(JsonSerializableRegistry):
        serializable_attributes = ["items"]

    h = _Holder.from_json({"type": "_Holder", "items": [{"type": "action", "content": "hi"}]},
                          serialization_type_field_name="type")
    assert h.items == [{"type": "action", "content": "hi"}]


def test_extract_json_keeps_unicode_escapes():
    assert extract_json('{"name": "caf\\u00e9"}') == {"name": "café"}


def test_enumerable_coercion_with_special_characters():
    chat = LLMChat.__new__(LLMChat)
    assert chat._coerce_to_enumerable("I'd go with C++ here.", ["C", "C++", "Python"]) == "C++"
    assert chat._coerce_to_enumerable("Answer: N/A (x)", ["Yes", "N/A (x)"]) == "N/A (x)"


def test_worlds_do_not_share_interventions():
    w1 = TinyWorld("regression world 1")
    w2 = TinyWorld("regression world 2")
    w1._interventions.append("an intervention")
    assert w2._interventions == []
