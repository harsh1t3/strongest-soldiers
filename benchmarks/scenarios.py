"""
Library-level scenarios used to compare this fork with upstream TinyTroupe.

Every scenario exercises documented public API, needs no LLM and no API key, and is deterministic, so the
same file can run against any checkout: `python benchmarks/scenarios.py` prints one JSON result per scenario.
Use benchmarks/compare_with_upstream.py to run it against two checkouts and diff the results.
"""
import json
import os
import sys
import tempfile
import traceback

SCENARIOS = []


def scenario(what_it_checks):
    """Registers a scenario. It returns evidence on success and raises on failure."""
    def register(func):
        SCENARIOS.append((func.__name__, what_it_checks, func))
        return func
    return register


def _reset_registries():
    from tinytroupe.agent import TinyPerson
    from tinytroupe.environment import TinyWorld

    TinyPerson.clear_agents()
    TinyWorld.clear_environments()


def _scripted_generator(*actions):
    """Stands in for the LLM: every turn produces the given actions."""
    return type("ScriptedGenerator", (), {"generate_next_actions": staticmethod(
        lambda agent, messages: ([dict(a) for a in actions], "assistant", {}, [])
    )})()


DONE = {"type": "DONE", "content": "", "target": ""}


# ---------------------------------------------------------------- simulation mechanics

@scenario("a social network runs its simulation steps")
def social_network_runs():
    from tinytroupe.environment import TinySocialNetwork

    network = TinySocialNetwork("benchmark network")
    network.run(1)
    return "run(1) completed"


@scenario("agents added through a relation are reachable by name")
def relations_register_agents():
    from tinytroupe.agent import TinyPerson
    from tinytroupe.environment import TinySocialNetwork

    alice, bob = TinyPerson("Alice"), TinyPerson("Bob")
    network = TinySocialNetwork("benchmark network")
    network.add_relation(alice, bob, "friends")

    assert network.get_agent_by_name("Bob") is bob, "Bob is not reachable by name"
    assert bob.environment is network, "Bob's environment was not set"
    return "both agents registered"


@scenario("interventions are not shared between worlds")
def interventions_are_per_world():
    from tinytroupe.environment import TinyWorld

    first, second = TinyWorld("first world"), TinyWorld("second world")
    first.add_intervention(object())

    assert second._interventions == [], f"second world also holds {len(second._interventions)} intervention(s)"
    return "intervention stayed in its own world"


@scenario("time can be skipped without an explicit timedelta")
def skip_without_timedelta():
    from tinytroupe.environment import TinyWorld

    TinyWorld("benchmark world").skip(3)
    return "skip(3) completed"


@scenario("agents acting in parallel share the step's simulation transaction")
def parallel_steps_are_one_transaction():
    import tinytroupe.control as control
    from tinytroupe.agent import TinyPerson
    from tinytroupe.environment import TinyWorld

    agents = [TinyPerson(f"Agent {i}") for i in range(3)]
    for agent in agents:
        agent.action_generator = _scripted_generator(DONE)
    world = TinyWorld("benchmark world", agents)

    control.reset()
    control.begin(os.path.join(tempfile.mkdtemp(), "trace.cache.json"))
    try:
        world.run(2, parallelize=True)
        entries = len(control.current_simulation().execution_trace)
    finally:
        control.end()
        control.reset()

    # one top-level transaction for run(); worker threads must nest inside it, not start their own
    assert entries == 1, f"{entries} top-level transactions recorded instead of 1, so the cache cannot replay"
    return "1 cache entry per run"


@scenario("a failing tool does not abort the agent's turn")
def failing_tool_does_not_abort_turn():
    from tinytroupe.agent import TinyPerson
    from tinytroupe.agent.mental_faculty import CustomMentalFaculty

    def broken_tool(agent, action):
        raise ValueError("malformed tool input")

    agent = TinyPerson("Agent")
    agent.add_mental_faculty(CustomMentalFaculty(
        "Tool", actions_configs={"USE_TOOL": {"description": "Uses a tool.", "function": broken_tool}}))
    agent.action_generator = _scripted_generator(
        {"type": "USE_TOOL", "content": "x", "target": ""}, {"type": "TALK", "content": "done", "target": ""}, DONE)

    agent.act(communication_display=False)
    committed = [a["type"] for a in agent.pop_latest_actions()]

    assert committed == ["USE_TOOL", "TALK", "DONE"], f"turn ended early: {committed}"
    return "turn continued after the tool failed"


# ---------------------------------------------------------------- agents and personas

@scenario("an agent's memory survives save_specification/load_specification")
def memory_survives_a_save_load_round_trip():
    from tinytroupe.agent import TinyPerson

    agent = TinyPerson("Agent")
    for i in range(3):
        agent.episodic_memory.store({"role": "user", "type": "stimulus", "simulation_timestamp": None,
                                     "content": {"stimuli": [{"type": "CONVERSATION", "content": f"message {i}",
                                                              "source": "someone"}]}})

    path = os.path.join(tempfile.mkdtemp(), "agent.json")
    agent.save_specification(path, include_memory=True)
    reloaded = TinyPerson.load_specification(path, new_agent_name="Agent reloaded")

    items = reloaded.episodic_memory.retrieve_all()
    broken = [i for i in items if not isinstance(i, dict)]
    assert not broken, f"{len(broken)} of {len(items)} memory items came back as {type(broken[0]).__name__}"
    assert len(items) == 3, f"{len(items)} memory items instead of 3"
    return "3 memory items restored as dicts"


@scenario("auto_rename_agent lets the same specification be loaded twice")
def auto_rename_agent_is_honored():
    from tinytroupe.agent import TinyPerson

    path = os.path.join(tempfile.mkdtemp(), "agent.json")
    TinyPerson("Agent").save_specification(path)

    first = TinyPerson.load_specification(path, auto_rename_agent=True)
    second = TinyPerson.load_specification(path, auto_rename_agent=True)

    assert len({first.name, second.name, "Agent"}) == 3, f"names collided: {first.name!r}, {second.name!r}"
    return f"renamed to {first.name!r} and {second.name!r}"


@scenario("a persona fragment shared by two agents is not mutated by either")
def persona_fragments_are_isolated():
    from tinytroupe.agent import TinyPerson

    fragment = {"interests": ["chess"]}
    first, second = TinyPerson("First"), TinyPerson("Second")
    first.include_persona_definitions(fragment)
    second.include_persona_definitions(fragment)
    first.define("interests", ["surfing"], merge=True)

    assert second._persona["interests"] == ["chess"], f"the other agent's interests became {second._persona['interests']}"
    assert fragment["interests"] == ["chess"], f"the fragment itself became {fragment['interests']}"
    return "fragment and the other agent unchanged"


@scenario("retrieve_last(0) returns no memories")
def retrieve_last_zero_returns_nothing():
    from tinytroupe.agent import TinyPerson

    agent = TinyPerson("Agent")
    for i in range(3):
        agent.episodic_memory.store({"role": "user", "type": "stimulus", "content": f"message {i}",
                                     "simulation_timestamp": None})

    retrieved = agent.episodic_memory.retrieve_last(0, include_omission_info=False)
    assert retrieved == [], f"returned {len(retrieved)} memories instead of none"
    return "returned nothing"


@scenario("repetition prevention can be turned off")
def repetition_prevention_can_be_disabled():
    from tinytroupe.agent import TinyPerson

    agent = TinyPerson("Agent", enable_basic_action_repetition_prevention=False)
    assert agent.enable_basic_action_repetition_prevention is False, "the setting was ignored"
    return "setting honored"


@scenario("relationships can be declared on a fresh agent")
def relationships_work_on_a_fresh_agent():
    from tinytroupe.agent import TinyPerson

    TinyPerson("First").related_to(TinyPerson("Second"), "colleague")
    return "related_to() worked"


# ---------------------------------------------------------------- tools and documents

@scenario("an agent can create a calendar event")
def calendar_event_can_be_created():
    from tinytroupe.agent import TinyPerson
    from tinytroupe.tools.tiny_calendar import TinyCalendar

    calendar = TinyCalendar()
    handled = calendar._process_action(TinyPerson("Agent"), {
        "type": "CREATE_EVENT", "content": json.dumps({"title": "Lunch", "date": "2026-10-07"})})

    assert handled, "the action was not handled"
    assert calendar.calendar["2026-10-07"][0]["title"] == "Lunch", "the event was not stored"
    return "event stored"


@scenario("documents with text needing sanitization can be indexed")
def documents_can_be_sanitized():
    from llama_index.core import Document
    from tinytroupe.agent.grounding import BaseSemanticGroundingConnector

    original = Document(text="old text", id_="doc-1", metadata={"semantic_memory_id": "notes.txt"})
    clone = BaseSemanticGroundingConnector("benchmark").clone_document_with_new_text(original, "new text")

    assert clone.text == "new text" and clone.id_ == "doc-1", "the clone lost its text or id"
    assert clone.metadata["semantic_memory_id"] == "notes.txt", "the clone lost its metadata"
    return "document cloned"


# ---------------------------------------------------------------- experiment analysis

@scenario("A/B randomization actually varies between items")
def ab_randomization_varies():
    from tinytroupe.experimentation.randomization import ABRandomizer

    randomizer = ABRandomizer(random_seed=42)
    orders = {randomizer.randomize(i, "control", "treatment") for i in range(200)}

    assert len(orders) == 2, f"all 200 items got the same order: {orders}"
    return "both orders occurred"


@scenario("clearly different populations are not scored as equivalent")
def different_populations_score_low():
    from tinytroupe.validation.simulation_validator import validate_simulation_experiment_empirically

    result = validate_simulation_experiment_empirically(
        control_data={"name": "control", "key_results": {"satisfaction": [5, 5, 6, 6, 7]}},
        treatment_data={"name": "treatment", "key_results": {"satisfaction": [1, 1, 2, 2, 3]}},
        validation_types=["statistical"], statistical_test_type="mann_whitney")

    assert result.overall_score < 0.5, (
        f"scored {result.overall_score:.2f} similarity for opposite data (1.0 means identical)")
    return f"similarity {result.overall_score:.2f}"


@scenario("a confidence interval has the sign of the difference it describes")
def confidence_interval_sign_matches():
    from tinytroupe.experimentation.statistical_tests import StatisticalTester

    result = StatisticalTester({"control": {"score": [1, 2, 3, 4, 5]}},
                               {"treatment": {"score": [11, 12, 13, 14, 15]}}).run_test("mann_whitney")["treatment"]["score"]
    low, high = result["confidence_interval"]

    assert result["median_difference"] > 0, "the treatment median should be higher"
    assert low > 0 and high > 0, f"the difference is +{result['median_difference']} but its CI is ({low}, {high})"
    return f"difference +{result['median_difference']}, CI ({low:.1f}, {high:.1f})"


@scenario("statistical assumptions can be checked for a metric")
def assumptions_can_be_checked():
    from tinytroupe.experimentation.statistical_tests import StatisticalTester

    tester = StatisticalTester({"control": {"score": [1, 2, 3, 4, 5]}}, {"treatment": {"score": [2, 3, 4, 5, 7]}})
    assert "treatment" in tester.check_assumptions("score"), "no assumptions reported"
    return "assumptions reported"


# ---------------------------------------------------------------- model output handling

@scenario("JSON from the model keeps its accented characters")
def model_json_keeps_unicode():
    from tinytroupe.utils.llm import extract_json

    extracted = extract_json(r'{"city": "São Paulo"}')
    assert extracted.get("city") == "São Paulo", f"got {extracted.get('city')!r}"
    return "accents preserved"


@scenario("results survive a model that writes one object per item")
def results_survive_object_per_item():
    from tinytroupe.utils.llm import extract_json

    # asked for several participants, models often emit one object per participant instead of an array,
    # which is not valid JSON as a whole; losing it means a whole simulation's findings come back empty
    extracted = extract_json('{"participant": "Riya", "would_pay": "no"}\n'
                             '{"participant": "Gurpreet", "would_pay": "maybe"}')

    assert extracted, "the whole extraction came back empty"
    assert len(extracted) == 2, f"kept {len(extracted)} of the 2 participants"
    return "both participants kept"


@scenario("a model choice is matched even when it contains punctuation")
def model_choices_with_punctuation_match():
    from tinytroupe.utils.llm import LLMChat

    chat = LLMChat.__new__(LLMChat)
    choice = chat._coerce_to_enumerable("I would go with C++ here.", ["C", "C++", "Python"])
    assert choice == "C++", f"matched {choice!r}"
    return "matched C++"


def main():
    results = []
    for name, what_it_checks, func in SCENARIOS:
        _reset_registries()
        try:
            results.append({"scenario": name, "checks": what_it_checks, "ok": True, "detail": func()})
        except Exception as e:
            detail = f"{type(e).__name__}: {e}".strip().splitlines()[0][:300]
            results.append({"scenario": name, "checks": what_it_checks, "ok": False, "detail": detail,
                            "traceback": traceback.format_exc()[-800:]})

    print("<<<BENCHMARK_RESULTS>>>")
    print(json.dumps(results))


if __name__ == "__main__":
    sys.exit(main())
