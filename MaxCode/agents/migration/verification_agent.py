"""Verification agent for scoring PyTorch-to-JAX conversion quality.

Produces a scorecard with two metrics:
  - Completeness (AST-based, no LLM): compares classes, methods, and
    standalone functions by name.
  - Correctness (LLM-based, requires API key): runs ValidationAgent to
    detect deviations and scores them with weighted penalties.
"""

import ast
from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    """Result of verifying a conversion."""
    completeness: dict = field(default_factory=dict)   # score, total, found, classes, methods, functions
    correctness: dict | None = None                    # score, deviations, by_category, by_severity (None if no api_key)
    overall: float = 0.0


# Standard PyTorch -> JAX/Flax method renames.
METHOD_RENAMES = {
    "__init__": {"setup", "__call__"},
    "forward": {"__call__"},
}

# Methods always inlined during conversion.
ALWAYS_INLINED = {
    "reset_parameters",
}

# Severity weights for correctness scoring.
SEVERITY_WEIGHTS = {"high": 5, "medium": 3, "low": 1}

# Known false-positive (category, severity) pairs.
FALSE_POSITIVE_RULES = {
    ("method_placement", "low"),
    ("missing_component", "low"),
    ("dropped_feature", "low"),
}


class VerificationAgent:
    """Scores the quality of a PyTorch-to-JAX conversion.

    The completeness check is pure AST (no LLM). The correctness check
    delegates to ValidationAgent for deviation detection and applies
    weighted scoring.
    """

    def __init__(self, model=None, target: str = "jax"):
        """Initialize the verification agent.

        Args:
            model: Optional LLM model instance for correctness checks.
                If None, correctness scoring is skipped.
            target: Conversion target ("jax" or "maxtext"). Threaded
                through to the inner ValidationAgent so the correctness
                check uses the right validation prompt.
        """
        self._model = model
        self._target = target

    @staticmethod
    def extract_components(code):
        """Parse Python code and return its classes, methods, and functions.

        Args:
            code: Python source code string.

        Returns:
            dict with keys "classes" (name -> [methods]) and "functions" (list).
        """
        tree = ast.parse(code)
        classes = {}
        functions = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    n.name
                    for n in ast.iter_child_nodes(node)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                classes[node.name] = methods
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)

        return {"classes": classes, "functions": functions}

    @staticmethod
    def compute_completeness(source_components, output_components):
        """Compare source and output components and return a completeness report.

        Returns:
            dict with score, total, found, classes, methods, functions breakdown.
        """
        src_classes = source_components["classes"]
        out_classes = output_components["classes"]

        src_class_names = set(src_classes.keys())
        out_class_names = set(out_classes.keys())
        matched_classes = src_class_names & out_class_names
        missing_classes = sorted(src_class_names - out_class_names)

        total_methods = 0
        found_methods = 0
        missing_methods = []

        for cls in src_classes:
            src_methods = set(src_classes[cls])
            total_methods += len(src_methods)
            if cls in out_classes:
                out_methods = set(out_classes[cls])
                has_call = "__call__" in out_methods
                for m in sorted(src_methods):
                    if m in out_methods:
                        found_methods += 1
                    elif m in METHOD_RENAMES and METHOD_RENAMES[m] & out_methods:
                        found_methods += 1
                    elif m in ALWAYS_INLINED:
                        found_methods += 1
                    elif has_call and m not in ("__init__", "forward"):
                        found_methods += 1
                    else:
                        missing_methods.append(f"{cls}.{m}")
            else:
                for m in sorted(src_methods):
                    missing_methods.append(f"{cls}.{m}")

        src_funcs = set(source_components["functions"])
        out_funcs = set(output_components["functions"])
        matched_funcs = src_funcs & out_funcs
        for f in src_funcs - matched_funcs:
            if f in out_class_names:
                matched_funcs = matched_funcs | {f}
        missing_funcs = sorted(src_funcs - matched_funcs)

        total = len(src_class_names) + total_methods + len(src_funcs)
        found = len(matched_classes) + found_methods + len(matched_funcs)
        score = (found / total * 100) if total > 0 else 100.0

        return {
            "score": round(score, 1),
            "total": total,
            "found": found,
            "classes": {
                "total": len(src_class_names),
                "found": len(matched_classes),
                "missing": missing_classes,
            },
            "methods": {
                "total": total_methods,
                "found": found_methods,
                "missing": missing_methods,
            },
            "functions": {
                "total": len(src_funcs),
                "found": len(matched_funcs),
                "missing": missing_funcs,
            },
        }

    @staticmethod
    def compute_correctness(source_code, output_code, api_key,
                            total_components=0, model=None, target: str = "jax"):
        """Run ValidationAgent and score the output.

        Args:
            source_code: The PyTorch source code.
            output_code: The converted output code (JAX or MaxText).
            api_key: Google API key for the LLM.
            total_components: Number of source components for budget scaling.
            model: Optional pre-configured LLM model instance. If None,
                creates a new GeminiTool with the given api_key.
            target: Conversion target ("jax" or "maxtext"). Selects which
                validation prompt the inner ValidationAgent uses.

        Returns:
            dict with score, deviation_count, deviations, filtered_deviations,
            by_category, by_severity.
        """
        import models
        from agents.migration.validation_agent import ValidationAgent

        if model is None:
            model = models.GeminiTool(
                model_name=models.GeminiModel.GEMINI_3_1_PRO_PREVIEW,
                api_key=api_key,
            )
        validator = ValidationAgent(model=model, target=target)
        all_deviations = validator.validate(source_code, output_code)

        if not isinstance(all_deviations, list):
            all_deviations = []

        real = []
        filtered = []
        for d in all_deviations:
            sev = d.get("severity", "low").lower()
            cat = d.get("category", "unknown")
            if (cat, sev) in FALSE_POSITIVE_RULES:
                filtered.append(d)
            else:
                real.append(d)

        by_severity = {}
        by_category = {}
        penalty = 0

        for d in real:
            sev = d.get("severity", "low").lower()
            cat = d.get("category", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            by_category[cat] = by_category.get(cat, 0) + 1
            penalty += SEVERITY_WEIGHTS.get(sev, 1)

        if total_components <= 0:
            try:
                tree = ast.parse(source_code)
                total_components = sum(
                    1 for n in ast.iter_child_nodes(tree)
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                )
            except SyntaxError:
                total_components = 0

        budget = total_components * SEVERITY_WEIGHTS["medium"]
        if budget > 0:
            score = max(0.0, (1.0 - penalty / budget) * 100.0)
        else:
            score = 100.0 if penalty == 0 else 0.0

        return {
            "score": round(score, 1),
            "deviation_count": len(real),
            "deviations": real,
            "filtered_deviations": filtered,
            "by_category": by_category,
            "by_severity": by_severity,
        }

    def verify(self, source_code, output_code, api_key=None):
        """Run full verification (completeness + optional correctness).

        Args:
            source_code: The PyTorch source code string.
            output_code: The converted JAX output code string.
            api_key: Optional Google API key. If provided (or if self._model
                is set), runs correctness check.

        Returns:
            VerificationResult with completeness, correctness, and overall score.
        """
        src_components = self.extract_components(source_code)
        out_components = self.extract_components(output_code)
        completeness = self.compute_completeness(src_components, out_components)

        correctness = None
        if api_key or self._model:
            correctness = self.compute_correctness(
                source_code, output_code,
                api_key=api_key,
                total_components=completeness["total"],
                model=self._model,
                target=self._target,
            )

        if correctness is not None:
            overall = round((completeness["score"] + correctness["score"]) / 2, 1)
        else:
            overall = completeness["score"]

        return VerificationResult(
            completeness=completeness,
            correctness=correctness,
            overall=overall,
        )
