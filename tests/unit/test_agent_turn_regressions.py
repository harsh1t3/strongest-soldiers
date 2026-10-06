"""
Offline regression tests for how TinyPerson.act() builds, generates and commits a turn (no LLM calls).
"""
import pytest

import tinytroupe.utils as utils
from tinytroupe.agent import TinyPerson
from tinytroupe.agent.action_generator import ActionGenerator
from tinytroupe.agent.mental_faculty import CustomMentalFaculty

from testing_utils import *

DONE = {"type": "DONE", "content": "", "target": ""}


def _talk(content):
    return {"type": "TALK", "content": content, "target": ""}


class _ScriptedGenerator:
    """Stands in for the LLM: returns (or raises) the scripted results in order, repeating the last one."""

    def __init__(self, *results):
        self.results = list(results)
        self.prompts = []

    def generate_next_actions(self, agent, current_messages):
        self.prompts.append(list(current_messages))
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return [dict(a) for a in result], "assistant", {}, []


def _stimuli_sent(prompt):
    # prompt = [system message, "produce your next actions" instruction, *stimulus payloads]
    return [s["content"] for msg in prompt[2:] for s in msg["content"]["stimuli"]]


def test_failing_commit_does_not_regenerate_and_duplicate_actions(monkeypatch):
    agent = TinyPerson("Ann")
    agent.action_generator = generator = _ScriptedGenerator([_talk("first"), _talk("second"), DONE])

    real_similarity = utils.next_action_jaccard_similarity

    def similarity(agent, action):
        if action.get("content") == "second":
            raise TypeError("failure while committing")
        return real_similarity(agent, action)

    monkeypatch.setattr(utils, "next_action_jaccard_similarity", similarity)

    with pytest.raises(TypeError):
        agent.act(communication_display=False)

    assert len(generator.prompts) == 1
    assert [a["content"] for a in agent.pop_latest_actions()] == ["first"]


def test_failing_generation_is_still_retried():
    agent = TinyPerson("Ann")
    agent.action_generator = generator = _ScriptedGenerator(KeyError("malformed"), [_talk("hello"), DONE])

    actions = agent.act(return_actions=True, communication_display=False)

    assert len(generator.prompts) == 2
    assert [a["action"]["type"] for a in actions] == ["TALK", "DONE"]
    assert [a["type"] for a in agent.pop_latest_actions()] == ["TALK", "DONE"]


def test_latest_stimulus_is_presented_only_until_acted_upon():
    agent = TinyPerson("Ann")
    agent.action_generator = generator = _ScriptedGenerator([_talk("reply"), DONE])

    agent.listen("Hi Ann!", communication_display=False)
    agent.act(communication_display=False)
    agent.act(communication_display=False)  # e.g., the next TinyWorld step, without new input
    agent.listen("Are you there?", communication_display=False)
    agent.act(communication_display=False)

    assert [_stimuli_sent(p) for p in generator.prompts] == [["Hi Ann!"], [], ["Are you there?"]]


def test_stimulus_produced_while_committing_is_presented_next_turn():
    agent = TinyPerson("Ann")
    agent.add_mental_faculty(
        CustomMentalFaculty(
            "Recall",
            actions_configs={
                "RECALL": {
                    "description": "Recalls something.",
                    "function": lambda agent, action: agent.think("The key is under the mat."),
                }
            },
        )
    )
    agent.action_generator = generator = _ScriptedGenerator(
        [{"type": "RECALL", "content": "key", "target": ""}, DONE], [_talk("It's under the mat."), DONE]
    )

    agent.listen("Where is the key?", communication_display=False)
    agent.act(communication_display=False)
    agent.act(communication_display=False)
    agent.act(communication_display=False)

    assert [_stimuli_sent(p) for p in generator.prompts] == [
        ["Where is the key?"],
        ["The key is under the mat."],
        [],
    ]


def test_direct_correction_handles_multi_action_output(monkeypatch):
    generator = ActionGenerator(
        max_attempts=1,
        enable_regeneration=False,
        enable_direct_correction=True,
        enable_multi_action_output=True,
    )

    tentative = [_talk("bad one"), {"type": "THINK", "content": "bad two", "target": ""}, DONE]
    monkeypatch.setattr(
        generator,
        "_generate_tentative_action",
        lambda *args, **kwargs: ([dict(a) for a in tentative], "assistant", {"actions": tentative}),
    )
    monkeypatch.setattr(
        generator,
        "_check_action_quality",
        lambda stage, agent, tentative_action: (
            (False, 1, "too bad") if stage == "Original Action" else (True, 9, "fine")
        ),
    )
    monkeypatch.setattr(utils, "extract_observed_vs_expected_rules", lambda situation: "be good")
    monkeypatch.setattr(utils, "correct_according_to_rule", lambda observation, rules: observation.replace("bad", "good"))

    actions, role, content, _ = generator.generate_next_actions(
        TinyPerson("Bob"), [{"role": "user", "content": "hi"}]
    )

    assert [a["content"] for a in actions] == ["good one", "good two", ""]
    assert actions[-1]["type"] == "DONE"
    assert content["actions"] == actions


def test_failing_faculty_does_not_abort_the_turn():
    def broken_tool(agent, action):
        raise ValueError("malformed tool input")

    agent = TinyPerson("Ann")
    agent.add_mental_faculty(
        CustomMentalFaculty(
            "Tool",
            actions_configs={"USE_TOOL": {"description": "Uses a tool.", "function": broken_tool}},
        )
    )
    agent.action_generator = generator = _ScriptedGenerator(
        [{"type": "USE_TOOL", "content": "x", "target": ""}, _talk("done with it"), DONE], [DONE]
    )

    agent.act(communication_display=False)
    agent.act(communication_display=False)

    # the rest of the turn was still committed, and the agent was told about the failure
    assert [a["type"] for a in agent.pop_latest_actions()] == ["USE_TOOL", "TALK", "DONE", "DONE"]
    assert "malformed tool input" in _stimuli_sent(generator.prompts[1])[0]
