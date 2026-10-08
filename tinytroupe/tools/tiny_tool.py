from tinytroupe.tools import logger
from tinytroupe.utils import JsonSerializableRegistry
from tinytroupe.utils import repeat_on_error


class TinyTool(JsonSerializableRegistry):

    # Define what attributes should be serialized
    serializable_attributes = ["name", "description", "real_world_side_effects"]

    def __init__(self, name, description, owner=None, real_world_side_effects=False, exporter=None, enricher=None):
        """
        Initialize a new tool.

        Args:
            name (str): The name of the tool.
            description (str): A brief description of the tool.
            owner (str): The agent that owns the tool. If None, the tool can be used by anyone.
            real_world_side_effects (bool): Whether the tool has real-world side effects. That is to say, if it has the potential to change the 
                state of the world outside of the simulation. If it does, it should be used with caution.
            exporter (ArtifactExporter): An exporter that can be used to export the results of the tool's actions. If None, the tool will not be able to export results.
            enricher (Enricher): An enricher that can be used to enrich the results of the tool's actions. If None, the tool will not be able to enrich results.
        
        """
        self.name = name
        self.description = description
        self.owner = owner
        self.real_world_side_effects = real_world_side_effects
        self.exporter = exporter
        self.enricher = enricher

    def _process_action(self, agent, action: dict) -> bool:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _protect_real_world(self):
        if self.real_world_side_effects:
            logger.warning(f" !!!!!!!!!! Tool {self.name} has REAL-WORLD SIDE EFFECTS. This is NOT just a simulation. Use with caution. !!!!!!!!!!")
        
    def _enforce_ownership(self, agent):
        if self.owner is not None and agent.name != self.owner.name:
            raise ValueError(f"Agent {agent.name} does not own tool {self.name}, which is owned by {self.owner.name}.")
    
    def set_owner(self, owner):
        self.owner = owner

    def actions_definitions_prompt(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def actions_constraints_prompt(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def process_action(self, agent, action: dict) -> bool:
        """
        Processes an action by delegating to the subclass implementation.

        If ``_process_action`` raises an exception, the tool-level retry
        mechanism re-invokes it (up to 3 times) after invalidating the
        last API cache entry.  This keeps retries granular — only the
        tool execution is repeated, avoiding side-effects that a full
        turn-level retry would cause.
        """
        self._protect_real_world()
        self._enforce_ownership(agent)

        @repeat_on_error(retries=3, exceptions=[Exception])
        def _try_process():
            return self._process_action(agent, action)

        return _try_process()
