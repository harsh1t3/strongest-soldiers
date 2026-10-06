"""
Offline (no LLM) regression tests for world stepping, the person factory sampling plan, stories,
profiling, the in-place experiment runner and tools.
"""
import logging
from unittest.mock import MagicMock, patch

import pytest

from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld
from tinytroupe.experimentation.in_place_experiment_runner import InPlaceExperimentRunner
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.factory.tiny_factory import TinyFactory
from tinytroupe.profiling import Profiler
from tinytroupe.steering import TinyStory
from tinytroupe.tools import TinyTool


@pytest.fixture(autouse=True)
def clean_registries():
    def clear():
        TinyPerson.clear_agents()
        TinyWorld.clear_environments()
        TinyFactory.clear_factories()
        TinyPersonFactory.clear_factories()

    clear()
    yield
    clear()


class FakeAgent:
    """Just enough of an agent for TinyWorld stepping, without any LLM."""

    def __init__(self, name, actions, fail=False):
        self.name = name
        self.environment = None
        self._actions = actions
        self._fail = fail
        self._buffer = []

    def act(self, return_actions=False):
        self._buffer.extend(self._actions)  # committed before a possible failure, as TinyPerson.act does
        if self._fail:
            raise RuntimeError("act failed")
        return list(self._actions)

    def pop_latest_actions(self):
        actions, self._buffer = self._buffer, []
        return actions


def _talk(content):
    return {"type": "TALK", "content": content, "target": ""}


def test_parallel_step_delivers_actions_buffered_before_an_act_failure():
    failing = FakeAgent("failing", [_talk("partial")], fail=True)
    ok = FakeAgent("ok", [_talk("hello")])
    world = TinyWorld("parallel world", [failing, ok])
    delivered = []
    world._handle_actions = lambda agent, actions: delivered.append((agent.name, actions))

    agents_actions = world._step_in_parallel()

    assert ("failing", [_talk("partial")]) in delivered  # this step, not one step late
    assert failing._buffer == []
    assert agents_actions == {"ok": [_talk("hello")]}


def test_parallel_step_isolates_delivery_failures_per_agent():
    first = FakeAgent("first", [_talk("a")])
    second = FakeAgent("second", [_talk("b")])
    world = TinyWorld("isolation world", [first, second])
    delivered = []

    def handle(agent, actions):
        if agent.name == "first":
            raise KeyError("type")
        delivered.append(agent.name)

    world._handle_actions = handle

    world._step_in_parallel()

    assert delivered == ["second"]
    assert first._buffer == [] and second._buffer == []


def test_world_name_is_not_orphaned_when_adding_agents_fails():
    duplicates = [FakeAgent("same", []), FakeAgent("same", [])]
    with pytest.raises(ValueError, match="Agent names must be unique"):
        TinyWorld("retry world", duplicates)

    assert TinyWorld.get_environment_by_name("retry world") is None
    world = TinyWorld("retry world", [FakeAgent("unique", [])])  # retry with the same name works
    assert TinyWorld.get_environment_by_name("retry world") is world


def _sampling_plan_mocks(plan):
    dims = MagicMock(return_value={"sampling_space_description": "x", "dimensions": []})
    sample_plan = MagicMock(return_value={"sample_plan": plan})
    names = MagicMock(side_effect=[f"Person Number{i}" for i in range(10)])
    return dims, sample_plan, names


def test_context_only_factory_samples_from_its_context():
    factory = TinyPersonFactory(context="Nurses in a rural hospital.", total_population_size=2)
    dims, sample_plan, names = _sampling_plan_mocks([{"sampled_values": {"age": 30}, "quantity": 2}])

    with patch.object(TinyPersonFactory, "_compute_sampling_dimensions", dims), \
         patch.object(TinyPersonFactory, "_compute_sample_plan", sample_plan), \
         patch.object(TinyPersonFactory, "_generate_name_for_sample", names):
        factory.initialize_sampling_plan()

    assert dims.call_args.kwargs == {
        "sampling_space_description": "Nurses in a rural hospital.",
        "context": "Nurses in a rural hospital.",
    }
    assert sample_plan.call_args.kwargs["context"] == "Nurses in a rural hospital."
    assert len(factory.remaining_characteristics_sample) == 2


def test_empty_sampling_plan_fails_the_postcondition():
    factory = TinyPersonFactory(sampling_space_description="Anyone.", total_population_size=2)
    dims, sample_plan, names = _sampling_plan_mocks([])

    with patch.object(TinyPersonFactory, "_compute_sampling_dimensions", dims), \
         patch.object(TinyPersonFactory, "_compute_sample_plan", sample_plan), \
         patch.object(TinyPersonFactory, "_generate_name_for_sample", names):
        with pytest.raises(ValueError, match="Postcondition not met"):
            factory.initialize_sampling_plan()

    assert factory.remaining_characteristics_sample is None


class _StoryAgent:
    def pretty_current_interactions(self, **kwargs):
        return ""


@pytest.mark.parametrize("method", ["start_story", "continue_story"])
def test_story_raises_clearly_when_the_llm_request_fails(method):
    story = TinyStory(agent=_StoryAgent(), context="Before.")
    failing_client = MagicMock()
    failing_client.send_message.return_value = None

    with patch("tinytroupe.steering.tiny_story.client", return_value=failing_client):
        with pytest.raises(RuntimeError, match="no story text"):
            getattr(story, method)()

    assert story.current_story == "Before."


def test_story_returns_the_llm_text():
    story = TinyStory(agent=_StoryAgent())
    ok_client = MagicMock()
    ok_client.send_message.return_value = {"role": "assistant", "content": "Once upon a time."}

    with patch("tinytroupe.steering.tiny_story.client", return_value=ok_client):
        assert story.start_story() == "Once upon a time."
    assert "Once upon a time." in story.current_story


def test_profiler_reads_relationships_of_real_agents():
    agent = TinyPerson("Ana Relationship")
    agent.define("relationships", [{"name": "Bob", "description": "Her colleague at the bank."}])
    profiler = Profiler()
    profiler.agents = [agent]

    with patch("tinytroupe.profiling.Normalizer", None):
        results = profiler._analyze_persona_composition()

    roles = results["relationship_roles"]
    assert "colleague" in roles["category"].tolist()


def test_compare_populations_top_3_is_by_frequency():
    nationalities = ["Argentina", "Brazil", "Chile"] + ["Zambia"] * 5
    agents = [{"nationality": n} for n in nationalities]
    profiler = Profiler(attributes=["nationality"])

    with patch("tinytroupe.profiling.Normalizer", None):
        profiler.profile(agents, plot=False, advanced_analysis=False)
        comparison = profiler.compare_populations(agents)

    top_3 = comparison["attribute_comparisons"]["nationality"]["current_top_3"]
    assert list(top_3) == ["Zambia", "Argentina", "Brazil"]
    assert top_3["Zambia"] == 5


def test_activate_next_experiment_runs_experiments_before_a_fixed_one(tmp_path):
    runner = InPlaceExperimentRunner(config_file_path=str(tmp_path / "experiments.json"))
    for name in ["A", "B", "C"]:
        runner.add_experiment(name)

    runner.fix_active_experiment("B")
    ran = [runner.get_active_experiment()]
    while True:
        runner.activate_next_experiment()
        if runner.has_finished_all_experiments():
            break
        ran.append(runner.get_active_experiment())

    assert ran == ["B", "C", "A"]
    assert runner.get_active_experiment() is None
    assert sorted(runner.experiment_config["finished_experiments"]) == ["A", "B", "C"]


def test_statistical_tests_skip_experiments_without_results(tmp_path, caplog):
    runner = InPlaceExperimentRunner(config_file_path=str(tmp_path / "experiments.json"))
    for name in ["control", "treatment", "pending"]:
        runner.add_experiment(name)
    runner.add_experiment_results({"score": [1.0, 2.0, 3.0, 4.0, 5.0]}, experiment_name="control")
    runner.add_experiment_results({"score": [3.0, 4.0, 5.0, 6.0, 7.0]}, experiment_name="treatment")

    with caplog.at_level(logging.WARNING, logger="tinytroupe"):
        results = runner.run_statistical_tests("control")

    assert set(results) == {"treatment"}
    assert any("'pending' has no results" in r.message for r in caplog.records)

    with pytest.raises(ValueError, match="has no results"):
        runner.run_statistical_tests("pending")
    with pytest.raises(ValueError, match="does not exist"):
        runner.run_statistical_tests("missing")


class _FailingTool(TinyTool):
    def __init__(self):
        super().__init__("failing", "Fails after a side effect.")
        self.side_effects = 0

    def _process_action(self, agent, action):
        self.side_effects += 1  # e.g. an enrichment call or an export
        raise IOError("export failed")


def test_tool_failure_does_not_repeat_side_effects():
    tool = _FailingTool()

    with pytest.raises(IOError):
        tool.process_action(MagicMock(name="agent"), {"type": "WRITE_DOCUMENT", "content": "{}"})

    assert tool.side_effects == 1
