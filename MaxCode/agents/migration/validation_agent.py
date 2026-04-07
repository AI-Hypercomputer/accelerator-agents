"""Agent for validating faithfulness of PyTorch-to-JAX conversions."""

import json
import re
from typing import Any

from agents import base
from agents import utils


VALIDATION_PROMPT = """You are an expert code reviewer specializing in PyTorch-to-JAX
conversions. Your task is to compare the ORIGINAL PyTorch source code with the
CONVERTED JAX/Flax output and identify every FAITHFULNESS DEVIATION.

A faithfulness deviation is any place where the JAX output CHANGES the behavior,
defaults, structure, or semantics of the original PyTorch code. You should NOT
flag intentional JAX idiom changes (e.g., torch.Tensor -> jnp.ndarray,
nn.Module -> nn.Module with @nn.compact, self.training -> deterministic flag).

## Original PyTorch Source:
```python
{pytorch_code}
```

## Converted JAX Output:
```python
{jax_code}
```

## Check each of the following categories:

### 1. Default Values
Compare every constructor parameter default in the source vs the output.
Flag any changed numeric value (e.g., capacity_factor=1.0 changed to 1.25).

### 2. Weight Initialization
For each nn.Linear/nn.Dense in the source:
- If the source uses bare `nn.Linear(...)` with NO explicit init call
  (no nn.init.zeros_, nn.init.normal_, etc.), the JAX output should use
  the Flax default initializer (no kernel_init argument).
- If the source EXPLICITLY calls an init (e.g., nn.init.zeros_), the JAX
  output should use the matching Flax initializer.
Flag any case where an initializer was added or changed.

### 3. Missing Components
List every class, function, method, or constant in the source that has
NO equivalent in the JAX output. Include:
- Base classes that were merged into subclasses
- get_config() or serialization methods
- Utility functions (metrics, logging helpers, etc.)
- Utility classes (e.g., metrics aggregation classes)
- Lambda attributes or property methods

### 4. Reduction Operations
Flag any place where .mean() was changed to .sum() or vice versa,
or where a reduction axis was changed.

### 5. Method Placement
Flag any method/attribute that was moved from one class to another,
or converted from an instance method to a standalone function when
the source has it as a method.

### 6. Dropped Features
Flag any feature present in the source that was removed in the output
(e.g., TensorBoard logging, checkpoint saving, progress bars, etc.)

## IMPORTANT: Use Exact Code Snippets
When reporting deviations, copy the relevant lines VERBATIM from the code
above. Do NOT paraphrase or describe the code in English. Use the actual
source and output lines so that a repair tool can find-and-replace them.

## Output Format

Return a JSON array of deviations. Each deviation must have:
- "category": one of "default_value", "initialization", "missing_component",
  "reduction_op", "method_placement", "dropped_feature"
- "severity": "high" (changes model output), "medium" (changes training behavior),
  or "low" (cosmetic or minor)
- "source_snippet": copy the exact line(s) verbatim from the PyTorch source
  (max 3 lines). For missing components, show the class/function signature.
- "output_snippet": copy the exact line(s) verbatim from the JAX output
  (max 3 lines). Use "MISSING" if the component does not exist.
- "corrected_snippet": the exact replacement code that should replace
  output_snippet to fix the deviation. Use "ADD" for missing components
  (and put the new code in the fix field).
- "fix": specific instruction for how to fix the deviation

If there are NO deviations, return an empty array: []

Return ONLY the JSON array, no markdown formatting, no explanation.
"""


REPAIR_PROMPT = """You are an expert JAX/Flax developer. You have been given a
JAX/Flax code file that was converted from PyTorch, along with a list of
faithfulness deviations that need to be fixed.

## Original PyTorch Source (for reference):
```python
{pytorch_code}
```

## Current JAX Code:
```python
{jax_code}
```
{rag_section}
## Deviations to Fix:
{deviations_text}

## CRITICAL RULES:
1. For each deviation, find the EXACT output_snippet in the JAX code and
   replace it with the corrected_snippet. If the snippets are not exact
   matches (whitespace differences, etc.), locate the closest match and
   apply the fix described in the instruction.
2. NEVER remove an existing class, function, method, or import -- even if it
   seems unused or redundant. If the current JAX code has a class (e.g.,
   MoETrainer, MoEMetrics), it MUST remain in the output.
3. NEVER convert a class into standalone functions or vice versa.
4. NEVER remove a training loop, epoch loop, or any training utility code.
5. If a deviation's instruction says the current behavior is acceptable,
   desirable, or "not recommended" to change, SKIP that deviation entirely.
6. Preserve ALL existing code structure -- only change what the deviation
   specifically asks you to change.
7. The output must have the SAME number of classes and functions (or more)
   as the input JAX code.

Return ONLY the complete fixed Python code. No markdown formatting, no
explanation, no ```python blocks.
"""


_CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\n?(.*?)\n?```", re.DOTALL)


def _strip_markdown_formatting(text: str) -> str:
    """Strips markdown and returns only the first Python code block."""
    code_block_match = _CODE_BLOCK_PATTERN.search(text)
    if code_block_match:
        return code_block_match.group(1).strip()
    return text


def _parse_json_response(text: str) -> list:
    """Parse JSON from LLM response, handling markdown wrapping."""
    text = text.strip()
    # Strip markdown code blocks if present
    json_match = re.search(r"```(?:json)?\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON array in the text
        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass
    return []


class ValidationAgent(base.Agent):
    """Agent for validating faithfulness of PyTorch-to-JAX conversions.

    This agent takes the original PyTorch source and the converted JAX output,
    identifies faithfulness deviations (changed defaults, wrong init, missing
    components, altered semantics), and optionally repairs them.
    """

    def __init__(self, model: Any, rag_agent_instance=None):
        """Initializes the agent.

        Args:
            model: The LLM model to use for generation.
            rag_agent_instance: Optional RAGAgent for retrieving context
                during repair. If None, repair runs without RAG context.
        """
        super().__init__(
            model=model,
            agent_domain=utils.AgentDomain.MIGRATION,
            agent_type=utils.AgentType.PRIMARY,
        )
        self._rag_agent = rag_agent_instance

    def validate(self, pytorch_code: str, jax_code: str) -> list:
        """Validates the JAX output against the PyTorch source.

        Args:
            pytorch_code: The original PyTorch source code.
            jax_code: The converted JAX/Flax code.

        Returns:
            A list of deviation dicts, each with category, severity,
            source_line, output_line, and fix fields.
        """
        response = self.generate(
            VALIDATION_PROMPT,
            {"pytorch_code": pytorch_code, "jax_code": jax_code},
        )
        return _parse_json_response(response)

    @staticmethod
    def _filter_actionable(deviations: list) -> list:
        """Filter out deviations that explicitly say not to fix."""
        skip_phrases = [
            "not recommended",
            "desirable deviation",
            "correct and desirable",
            "overly complex",
            "acceptable deviation",
        ]
        actionable = []
        for d in deviations:
            fix_text = d.get("fix", "").lower()
            if any(phrase in fix_text for phrase in skip_phrases):
                continue
            actionable.append(d)
        return actionable

    @staticmethod
    def _format_deviations_for_repair(deviations: list) -> str:
        """Formats deviations as numbered find/replace blocks for repair.

        Falls back to old source_line/output_line fields if the new
        source_snippet/output_snippet fields are absent.

        Args:
            deviations: List of deviation dicts from validate().

        Returns:
            A formatted string with numbered find/replace blocks.
        """
        blocks = []
        for i, d in enumerate(deviations, 1):
            severity = d.get("severity", "medium")
            category = d.get("category", "unknown")
            source = d.get("source_snippet", d.get("source_line", "N/A"))
            output = d.get("output_snippet", d.get("output_line", "N/A"))
            corrected = d.get("corrected_snippet", "")
            fix = d.get("fix", "")

            block = f"### Deviation {i} [{severity}] - {category}\n"
            block += f"Source (PyTorch):   {source}\n"
            block += f"Find in JAX:        {output}\n"
            if corrected and corrected not in ("ADD", "MISSING"):
                block += f"Replace with:       {corrected}\n"
            block += f"Instruction:        {fix}"
            blocks.append(block)
        return "\n\n".join(blocks)

    def _get_repair_rag_context(self, deviations: list) -> str:
        """Retrieves RAG context relevant to the repair deviations.

        Builds a query from deviation categories and fix text, retrieves
        top-k documents, and returns a formatted string for the prompt.

        Args:
            deviations: List of deviation dicts from validate().

        Returns:
            A formatted RAG context string, or "" if no RAG agent.
        """
        if not self._rag_agent:
            return ""

        # Build query from deviation categories and fix descriptions
        query_parts = []
        for d in deviations:
            category = d.get("category", "")
            fix = d.get("fix", "")
            if category:
                query_parts.append(category.replace("_", " "))
            if fix:
                query_parts.append(fix)
        query = " ".join(query_parts)
        if not query.strip():
            return ""

        try:
            docs = self._rag_agent.retrieve_context(query, top_k=3)
        except Exception:
            return ""

        if not docs:
            return ""

        section = "\n## Reference Patterns (from RAG):\n"
        for doc in docs:
            name = doc.get("name", "unknown")
            text = doc.get("text", "")
            section += f"\n### {name}\n{text}\n"
        return section

    def repair(self, jax_code: str, deviations: list,
               pytorch_code: str = "") -> str:
        """Repairs the JAX code based on identified deviations.

        Args:
            jax_code: The converted JAX/Flax code to repair.
            deviations: List of deviation dicts from validate().
            pytorch_code: The original PyTorch source for reference.

        Returns:
            The repaired JAX code.
        """
        # Filter to only actionable deviations
        actionable = self._filter_actionable(deviations)
        if not actionable:
            return jax_code

        deviations_text = self._format_deviations_for_repair(actionable)
        rag_section = self._get_repair_rag_context(actionable)
        response = self.generate(
            REPAIR_PROMPT,
            {
                "jax_code": jax_code,
                "deviations_text": deviations_text,
                "rag_section": rag_section,
                "pytorch_code": pytorch_code,
            },
        )
        repaired = _strip_markdown_formatting(response)
        # If the repair returned empty or very short, return original
        if len(repaired) < len(jax_code) * 0.5:
            return jax_code
        return repaired

    def run(self, pytorch_code: str, jax_code: str) -> tuple:
        """Validates and optionally repairs the conversion.

        Args:
            pytorch_code: The original PyTorch source code.
            jax_code: The converted JAX/Flax code.

        Returns:
            Tuple of (repaired_code, deviations_list).
        """
        deviations = self.validate(pytorch_code, jax_code)
        if deviations:
            repaired_code = self.repair(
                jax_code, deviations, pytorch_code=pytorch_code
            )
            return repaired_code, deviations
        return jax_code, []
