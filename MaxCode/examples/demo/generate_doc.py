"""Generate the MaxCode Pipeline Technical Reference as a Word document."""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import os

doc = Document()

style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)

# ── Title ──
title = doc.add_heading("MaxCode Migration Pipeline", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run("Technical Reference — PyTorch to JAX/Flax Conversion")
run.font.size = Pt(14)
run.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

doc.add_paragraph()

# ── 1. Overview ──
doc.add_heading("1. Pipeline Overview", level=1)
doc.add_paragraph(
    "MaxCode converts PyTorch repositories to JAX/Flax through a five-step "
    "pipeline. Each step is an independent script that reads the output of "
    "the previous step, allowing re-runs without restarting from scratch."
)

# Steps table
table = doc.add_table(rows=6, cols=3, style="Light Shading Accent 1")
table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ["Step", "Script", "Purpose"]
for i, h in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

steps = [
    ("1 — Clone", "step1_clone_repo.py", "Fetch the PyTorch repository from GitHub"),
    ("2 — Index", "step2_populate_rag.py", "Build the RAG vector database from reference JAX/Flax sources"),
    ("3 — Merge", "step3_merge.py", "Auto-detect model files, resolve dependencies, merge into one file"),
    ("4 — Convert", "step4_convert.py", "Run conversion with RAG context, fill gaps, validate, and repair"),
    ("5 — Verify", "step5_verify.py", "Score completeness (AST) and correctness (LLM) of the output"),
]
for row_idx, (step, script, purpose) in enumerate(steps, 1):
    table.rows[row_idx].cells[0].text = step
    table.rows[row_idx].cells[1].text = script
    table.rows[row_idx].cells[2].text = purpose

doc.add_paragraph()
doc.add_paragraph(
    "The pipeline produces a single JAX/Flax output file from potentially "
    "many input PyTorch files. This single-file approach gives the LLM full "
    "context during conversion and simplifies validation."
)

# ── 2. RAG Indexing ──
doc.add_heading("2. RAG Indexing Strategy (Step 2)", level=1)

doc.add_heading("2.1 Document Corpus", level=2)
doc.add_paragraph(
    "The RAG database contains 48 reference documents stored under "
    "MaxCode/rag/sources/, split into two categories:"
)
bullets = doc.add_paragraph(style="List Bullet")
bullets.text = (
    "Generic references (24 files) — JAX/Flax API documentation, MaxText "
    "model implementations, Flash-linear-attention examples, Flax attention patterns."
)
doc.add_paragraph(
    "Targeted patterns (24 files) — WRONG/CORRECT/WHY triplets covering "
    "common conversion mistakes: incorrect cosine similarity, wrong einsum "
    "dimensions, missing weight initialisation, broken MoE routing, etc.",
    style="List Bullet",
)

doc.add_heading("2.2 Embedding Flow", level=2)
doc.add_paragraph(
    "Each .py file in the source directory goes through the following pipeline:"
)
items = [
    "Read the file content.",
    "Generate a structured description using Gemini (CODE_DESCRIPTION prompt) "
    "that captures the file's functionality and usage in JSON format.",
    "Embed the description (not the raw code) using Google's embedding-001 "
    "model. This produces a dense vector in float32.",
    "Store the document in a SQLite database (rag_store.db) with columns: "
    "id, name, text (full source), desc (generated description), file (path), "
    "embedding (pickled numpy array).",
]
for item in items:
    doc.add_paragraph(item, style="List Number")

doc.add_paragraph(
    "A 2-second sleep is enforced between embedding API calls to respect "
    "rate limits. Results are cached in-memory to avoid redundant calls "
    "within the same session."
)

doc.add_heading("2.3 Vector Index", level=2)
doc.add_paragraph(
    "At query time, all stored embeddings are loaded into a NumPy array "
    "(shape: num_docs x embedding_dim). Search uses squared L2 (Euclidean) "
    "distance with np.argsort to find the top-k nearest neighbours. There "
    "is no approximate nearest-neighbour index (FAISS, Annoy, etc.) — the "
    "corpus is small enough (~48 docs) for exact brute-force search."
)

# Key params table
doc.add_heading("2.4 Key Parameters", level=2)
t2 = doc.add_table(rows=7, cols=3, style="Light Shading Accent 1")
t2.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(["Parameter", "Value", "Location"]):
    t2.rows[0].cells[i].text = h
    for r in t2.rows[0].cells[i].paragraphs[0].runs:
        r.bold = True
params = [
    ("Embedding model", "models/embedding-001 (Google)", "embedding.py"),
    ("Description model", "Gemini 2.5 Flash", "step2_populate_rag.py"),
    ("Distance metric", "Squared L2 (Euclidean)", "vector_db.py"),
    ("Storage format", "SQLite + pickled float32 arrays", "vector_db.py"),
    ("API sleep", "2 seconds between calls", "embedding.py"),
    ("Max context length", "100,000 characters", "rag_agent.py"),
]
for row_idx, (p, v, loc) in enumerate(params, 1):
    t2.rows[row_idx].cells[0].text = p
    t2.rows[row_idx].cells[1].text = v
    t2.rows[row_idx].cells[2].text = loc

# ── 3. Merge ──
doc.add_heading("3. Merge Strategy (Step 3)", level=1)

doc.add_heading("3.1 Model File Detection", level=2)
doc.add_paragraph(
    "The merge script scans every .py file in the repository and identifies "
    "model files by parsing the AST looking for class definitions that "
    "subclass nn.Module (matching torch.nn.Module, nn.Module, or bare Module). "
    "Files are opened with utf-8-sig encoding to handle BOM characters."
)

doc.add_heading("3.2 File-Level Filtering", level=2)
doc.add_paragraph("Before merging, several file-level filters are applied:")
filters = [
    "Config exclude patterns — path globs defined in config.py (e.g. tests/*, setup.py).",
    "Fused kernel heuristic — files matching fused_*.py are skipped.",
    "Infrastructure files — files where every class subclasses an infrastructure "
    "base (autograd.Function, PipelineModule, TransformerEngine wrappers, Enum) "
    "AND the file imports infrastructure packages (apex, deepspeed, transformer_engine).",
]
for f in filters:
    doc.add_paragraph(f, style="List Bullet")

doc.add_heading("3.3 Dependency Resolution", level=2)
doc.add_paragraph(
    "An import graph is built between the remaining model files by parsing "
    "ImportFrom AST nodes and resolving them to file paths (both relative "
    "and absolute-style imports). Entry points are identified as files that "
    "are not imported by any other model file but do import at least one. "
    "A BFS + DFS post-order traversal produces a topological ordering: "
    "dependencies first, entry points last."
)

doc.add_heading("3.4 Merge Process", level=2)
items = [
    "Standard-library imports are de-duplicated and collected at the top.",
    "Local cross-file imports are removed (no longer needed in a single file).",
    "Empty blocks left behind by import removal get a 'pass' statement inserted.",
    "Code sections are concatenated with file-boundary comments.",
    "A second pass removes infrastructure classes from the merged output "
    "(autograd.Function subclasses, PipelineModule, TransformerEngine wrappers, "
    "Enum subclasses, *Pipe-suffixed classes).",
]
for item in items:
    doc.add_paragraph(item, style="List Number")

doc.add_paragraph(
    "The result is a single merged_model.py file with all model definitions "
    "in dependency order, ready for conversion."
)

# ── 4. Retrieval ──
doc.add_heading("4. Retrieval Strategy", level=1)

doc.add_heading("4.1 Hybrid Per-Component Retrieval", level=2)
doc.add_paragraph(
    "All three conversion agents (SingleFileAgent, ModelConversionAgent, "
    "RepoAgent) use the retrieve_per_component_context() method, which "
    "combines two strategies:"
)

doc.add_heading("Full-File Query (Broad Context)", level=3)
doc.add_paragraph(
    "The entire PyTorch source code is embedded as a single query and "
    "the top 15 results are retrieved. This captures the overall domain "
    "(transformer architecture, attention patterns, etc.) and provides "
    "broad reference material."
)

doc.add_heading("Per-Component Queries (Targeted Context)", level=3)
doc.add_paragraph(
    "The source code is parsed with Python's ast module to extract each "
    "top-level class and function. A focused query string is built for each:"
)
doc.add_paragraph(
    'Classes: "JAX Flax {ClassName} {base_classes} {method_names} {init_params}"',
    style="List Bullet",
)
doc.add_paragraph(
    'Functions: "JAX Flax {func_name} {param_names}"',
    style="List Bullet",
)
doc.add_paragraph(
    "If there are more than 12 components, signatures are batched in groups "
    "of 4 to cap the number of embedding API calls at roughly 3-5."
)

doc.add_heading("Deduplication and Ranking", level=3)
doc.add_paragraph(
    "Results from both the full-file query and all per-component queries "
    "are merged into a single set, deduplicated by file path (keeping the "
    "entry with the best distance for each file). The final list is sorted "
    "by distance and truncated to max_total (default 15). If AST parsing "
    "fails, the method falls back to a single full-file query."
)

t3 = doc.add_table(rows=4, cols=2, style="Light Shading Accent 1")
t3.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(["Parameter", "Default"]):
    t3.rows[0].cells[i].text = h
    for r in t3.rows[0].cells[i].paragraphs[0].runs:
        r.bold = True
for row_idx, (p, v) in enumerate([
    ("top_k_per_component", "3"),
    ("max_total", "15"),
    ("Batch threshold", ">12 components"),
], 1):
    t3.rows[row_idx].cells[0].text = p
    t3.rows[row_idx].cells[1].text = v

# ── 5. Conversion ──
doc.add_heading("5. Conversion Pipeline (Step 4)", level=1)

doc.add_heading("5.1 Agent Routing", level=2)
doc.add_paragraph(
    "The PrimaryAgent receives the merged file path and orchestrates "
    "the conversion. For each file (or the single merged file), it "
    "decides which specialised agent to use:"
)
doc.add_paragraph(
    "ModelConversionAgent — if the file contains nn.Module subclasses "
    "(detected by is_model_file()). Uses MODEL_CONVERSION_PROMPT with "
    "16 conversion rules covering @nn.compact, KV caches, MoE dispatch, "
    "fused QKV projections, float32 softmax upcast, etc.",
    style="List Bullet",
)
doc.add_paragraph(
    "SingleFileAgent — for utility code, training loops, and data loading. "
    "Uses MIGRATE_MODULE_TO_JAX_PROMPT with general JAX best practices.",
    style="List Bullet",
)
doc.add_paragraph(
    "Both agents inject RAG context (retrieved via the hybrid strategy above) "
    "directly into the prompt alongside the PyTorch source code."
)

doc.add_heading("5.2 Gap-Filling (Two Phases)", level=2)
doc.add_paragraph(
    "After the initial conversion, _fill_missing_components() runs two "
    "phases to catch what the LLM missed:"
)

doc.add_heading("Phase 1 — Missing Top-Level Components", level=3)
doc.add_paragraph(
    "An AST diff compares class and function names between the PyTorch "
    "source and the JAX output. Any top-level component present in the "
    "source but absent in the output is extracted, sent to the LLM with "
    "RAG context, and the converted result is appended to the JAX file."
)

doc.add_heading("Phase 2 — Stub Detection and Missing Methods", level=3)
doc.add_paragraph("Two checks run on the JAX output:")
doc.add_paragraph(
    "Stub detection — walks the AST looking for functions/methods with "
    "placeholder bodies: pass, return None, ... (Ellipsis), docstring-only, "
    "or raise NotImplementedError.",
    style="List Bullet",
)
doc.add_paragraph(
    "Missing-method detection — for each class that exists in both source "
    "and output, compares method sets and identifies methods present in "
    "the PyTorch class but absent from the JAX class.",
    style="List Bullet",
)
doc.add_paragraph(
    "The PyTorch source for all identified stubs and missing methods is "
    "collected and sent in a single LLM call (FILL_STUBS_PROMPT) that "
    "receives the complete JAX file and returns the complete file with "
    "stubs replaced by real implementations. The result is accepted only "
    "if it passes ast.parse() and is at least 50% the length of the original."
)

doc.add_heading("5.3 Markdown Stripping", level=2)
doc.add_paragraph(
    "All LLM responses pass through _strip_markdown_formatting() which "
    "extracts the first Python code block from markdown-formatted output. "
    "It handles three cases: (1) properly fenced ```python...``` blocks, "
    "(2) truncated responses where the opening ``` is present but the "
    "closing ``` is missing (common with long outputs), and "
    "(3) triple-quote wrappers."
)

# ── 6. Validate & Repair ──
doc.add_heading("6. Validation and Repair Loop", level=1)

doc.add_heading("6.1 Validation Agent", level=2)
doc.add_paragraph(
    "The ValidationAgent performs an LLM-based comparison between the "
    "original PyTorch source and the converted JAX output. It checks "
    "six categories of deviations:"
)

t4 = doc.add_table(rows=7, cols=3, style="Light Shading Accent 1")
t4.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(["Category", "What It Catches", "Example"]):
    t4.rows[0].cells[i].text = h
    for r in t4.rows[0].cells[i].paragraphs[0].runs:
        r.bold = True
cats = [
    ("default_value", "Constructor parameter defaults changed",
     "init_method changed from xavier_normal to normal(0.02)"),
    ("initialization", "Weight initialisation added or changed",
     "zeros_init added where PyTorch uses default"),
    ("missing_component", "Classes, functions, methods, constants absent",
     "mup_reinitialize_weights method missing from class"),
    ("reduction_op", ".mean() vs .sum() or axis changes",
     "loss.mean() changed to loss.sum()"),
    ("method_placement", "Methods moved between classes or inlined",
     "helper moved from ClassA to ClassB"),
    ("dropped_feature", "Features removed entirely",
     "Sinkhorn error tracking loop removed"),
]
for row_idx, (cat, what, ex) in enumerate(cats, 1):
    t4.rows[row_idx].cells[0].text = cat
    t4.rows[row_idx].cells[1].text = what
    t4.rows[row_idx].cells[2].text = ex

doc.add_paragraph()
doc.add_paragraph(
    "Each deviation is assigned a severity (high, medium, or low) and "
    "includes source_snippet, output_snippet, corrected_snippet, and a "
    "fix instruction. The output is a JSON array."
)

doc.add_heading("6.2 Repair Loop", level=2)
doc.add_paragraph(
    "The PrimaryAgent runs up to 3 iterations of validate-then-repair:"
)

items = [
    "Validate: run the ValidationAgent to produce a list of deviations.",
    "Exit early if zero deviations remain (clean).",
    "Exit early if deviation count did not decrease from the previous "
    "iteration (no progress — avoid infinite loops).",
    "Filter actionable deviations: skip any whose fix text contains "
    "phrases like 'not recommended', 'desirable deviation', or 'acceptable'.",
    "Build repair prompt: inject the original PyTorch source, current JAX "
    "code, formatted deviation blocks, and RAG context (top 15 results "
    "queried from deviation categories and fix descriptions).",
    "The LLM returns the complete repaired JAX file. Accept only if the "
    "result is at least 50% the length of the input.",
]
for item in items:
    doc.add_paragraph(item, style="List Number")

doc.add_paragraph(
    "After the loop completes, validation results are stored per file "
    "with full iteration history (deviation counts per iteration, "
    "initial and remaining deviations)."
)

# ── 7. Verification ──
doc.add_heading("7. Verification Scorecard (Step 5)", level=1)

doc.add_heading("7.1 Completeness Score (AST-Based, No LLM)", level=2)
doc.add_paragraph(
    "Both the source and output files are parsed with Python's ast module. "
    "Three component types are compared by name:"
)
doc.add_paragraph("Classes — exact name match.", style="List Bullet")
doc.add_paragraph(
    "Methods — within matched classes, checked with rename awareness: "
    "__init__ may map to setup or __call__, forward maps to __call__. "
    "Methods like reset_parameters are treated as always-inlined (Flax "
    "handles them via initialiser arguments). Private/helper methods "
    "within a class that has __call__ are treated as legitimately inlined.",
    style="List Bullet",
)
doc.add_paragraph(
    "Functions — a PyTorch function is also considered matched if it was "
    "promoted to a class in the output.",
    style="List Bullet",
)
doc.add_paragraph()
p = doc.add_paragraph()
run = p.add_run("Formula:  ")
run.bold = True
p.add_run("score = (matched_classes + matched_methods + matched_functions) "
          "/ (total_classes + total_methods + total_functions) * 100")

doc.add_heading("7.2 Correctness Score (LLM-Based)", level=2)
doc.add_paragraph(
    "The ValidationAgent is run against the source and output. Deviations "
    "are filtered for known false positives (low-severity method_placement, "
    "missing_component, and dropped_feature are excluded as they represent "
    "legitimate Flax idioms)."
)
doc.add_paragraph(
    "Each remaining deviation contributes a penalty based on severity:"
)

t5 = doc.add_table(rows=4, cols=2, style="Light Shading Accent 1")
t5.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(["Severity", "Penalty"]):
    t5.rows[0].cells[i].text = h
    for r in t5.rows[0].cells[i].paragraphs[0].runs:
        r.bold = True
for row_idx, (s, p_val) in enumerate([("High", "5"), ("Medium", "3"), ("Low", "1")], 1):
    t5.rows[row_idx].cells[0].text = s
    t5.rows[row_idx].cells[1].text = p_val

doc.add_paragraph()
p = doc.add_paragraph()
run = p.add_run("Formula:  ")
run.bold = True
p.add_run("budget = total_components * 3  (medium severity weight)")
doc.add_paragraph()
p2 = doc.add_paragraph()
p2.add_run("             score = max(0,  (1 - penalty / budget) * 100)")
doc.add_paragraph()
doc.add_paragraph(
    "The budget scales with codebase size, so a large repository with "
    "150+ components is not unfairly penalised compared to a small one. "
    "A medium-severity deviation on every single component yields 0%. "
    "A high-severity deviation costs more than one component's budget "
    "(5 > 3), appropriately penalising severe issues."
)

doc.add_heading("7.3 Overall Score", level=2)
p = doc.add_paragraph()
run = p.add_run("Formula:  ")
run.bold = True
p.add_run("overall = (completeness + correctness) / 2")

doc.add_paragraph()
doc.add_paragraph(
    "Results are saved as verification_scorecard.json in the output "
    "directory, including full deviation details for post-mortem analysis."
)

# ── 8. Architecture Diagram ──
doc.add_heading("8. Architecture Diagram", level=1)

diagram = doc.add_paragraph()
diagram.paragraph_format.space_before = Pt(6)
diagram.paragraph_format.space_after = Pt(6)
run = diagram.add_run(
    "PyTorch Repository\n"
    "       |\n"
    "       v\n"
    " [Step 1: Clone]\n"
    "       |\n"
    "       v\n"
    " [Step 2: Index] -----> RAG Vector DB (48 docs, embedding-001)\n"
    "       |                        |\n"
    "       v                        |\n"
    " [Step 3: Merge]                |   (hybrid per-component retrieval)\n"
    "       |                        |\n"
    "       v                        v\n"
    " merged_model.py --------> [Step 4: Convert]\n"
    "                           |    |\n"
    "                  route--->|    |\n"
    "                 /         |    |\n"
    "    ModelConversion   SingleFile |\n"
    "         Agent         Agent    |\n"
    "                \\       /       |\n"
    "                 v     v        |\n"
    "           Fill Missing Components\n"
    "           (Phase 1: top-level gaps)\n"
    "           (Phase 2: stubs + methods)\n"
    "                    |           |\n"
    "                    v           |\n"
    "             Validate & Repair  |\n"
    "             (up to 3 iters)    |\n"
    "                    |           |\n"
    "                    v           v\n"
    "              repo_name_jax.py\n"
    "                    |\n"
    "                    v\n"
    "             [Step 5: Verify]\n"
    "                    |\n"
    "                    v\n"
    "             Scorecard (JSON)\n"
    "        Completeness | Correctness | Overall"
)
run.font.name = "Consolas"
run.font.size = Pt(9)

# ── Save ──
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "MaxCode_Pipeline_Reference.docx")
doc.save(out_path)
print(f"Saved: {out_path}")
