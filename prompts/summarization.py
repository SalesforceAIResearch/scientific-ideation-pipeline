"""Summarization prompts + the SummaryType registry for the Part-B dataset
generator (`dataset_creation/part_b/summarize_document.py`).

Two SCOPES of summary live here:

  DOCUMENT-scoped  — one LLM call per single document (title + body):
    TARGET_IDEA          target paper      -> research-idea card
    SOURCE_TO_IDEA       each source paper -> research-idea card
    SOURCE_TO_FULL_SUMM  each source paper -> six-field full summary

  RECORD-scoped    — one LLM call per record, consuming the WHOLE source set:
    IDEA_GROUNDED          source full summaries + target paper -> ONE idea in
                           v0 format, with the insight grounded in the provided
                           Document N set (same coordinate system as inference).
    SOURCE_MULTIPLE_OUTPUT the v0 inference prompt itself, run over the source
                           summaries to emit K ideas (<think>/<answer> list).
                           This is the SFT-data generator: its input is byte-
                           identical to live inference input.

`SummaryType` (an Enum) is the single key for every summary type — it is the
registry key, the --types CLI vocabulary, and what callers branch on. Each
member's spec (`SUMMARY_SPECS[st]`) carries: scope, expected output fields,
system prompt, and user template.
"""

from enum import Enum
from typing import Dict, List, Optional

from .ideation import IDEATION_V0_TEMPLATE, DEFAULT_NUM_IDEAS


# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #
class Scope(str, Enum):
    """What a single LLM call consumes."""
    TARGET_DOC = "target_doc"   # the target paper's own body
    SOURCE_DOC = "source_doc"   # one source paper's own body
    RECORD = "record"           # the whole record (source set [+ target])


# --------------------------------------------------------------------------- #
# DOCUMENT-scoped prompts (one document per call)
# --------------------------------------------------------------------------- #
# Shared by TARGET_IDEA and SOURCE_TO_IDEA: the IDEA -- conceptual approach --
# not its empirical validation.
IDEA_SYSTEM = (
    "You are an expert research scientist. You read a paper (title plus full "
    "text or abstract) and distill it into a compact, structured research-idea "
    "card. The card captures the IDEA -- the conceptual approach -- not its "
    "empirical validation. Be faithful to the source: do not invent "
    "contributions it does not claim. Return only a single JSON object."
)

IDEA_TEMPLATE = """Summarize the paper below into a structured research-idea card.
The card has three top-level fields -- `motivation`, `problem`, `insight` -- that
play DIFFERENT roles; a card whose fields collapse into restatements of each
other is not useful. `insight` is itself a structured object. Read the
definitions carefully.

  motivation:        The outward-facing stakes -- why anyone beyond the authors
                     should care if this progresses. Who is affected, what becomes
                     possible downstream, what limitation it would lift or cost it
                     would reduce. Answer "if this worked, what would change in the
                     world, and for whom?" -- not merely the gap in prior work
                     (1-2 sentences).
  problem:           The field-level research question -- the broad challenge the
                     work takes on, of the kind many researchers in the area might
                     independently arrive at. Name the object of study and the
                     difficulty or unknown about it, WITHOUT committing to the
                     specific angle, mechanism, or approach this paper uses (that
                     belongs in `insight`). It is the *what*, distinct from the
                     *why* (motivation) and the *how* (method) (1-2 sentences).
  insight:           The non-obvious observation, mechanism, or reframing -- the
                     load-bearing belief that gives the problem a specific, newly
                     tractable angle and distinguishes this work from others on the
                     same problem. It is a claim about how the world *is* (an
                     asymmetry, a hidden dependency, a failed assumption, a
                     reinterpretation), stated as the idea itself, independent of
                     how well it turned out. WHAT approach/mechanism/hypothesis does
                     the work introduce, and WHY would it work? `insight` is an
                     OBJECT with three sub-fields:
                       claim:          The insight itself (1-2 sentences), stated as
                                       a claim about how the world is.
                       grounding:      A list of the specific prior works THIS paper
                                       builds on for the claim -- the references it
                                       cites / the established findings it leans on,
                                       as named in the paper's own text (its intro,
                                       related work, or inline citations). Each entry
                                       is {{"source": "<cited work: title, author+year,
                                       or id as the paper names it>", "supports":
                                       "<what that work establishes that the claim
                                       builds on>"}}. Use one or more entries when the
                                       paper names the prior work it builds on; if the
                                       source text (e.g. a bare abstract) does not name
                                       specific prior work, return an empty list rather
                                       than inventing citations.
                       grounding_type: one of
                         - "in_domain_extension"      -- builds directly on findings in
                           the problem's home domain (does X+1 where the cited work does X)
                         - "cross_domain_transfer"    -- imports a mechanism established
                           in a different domain and applies it to this problem
                         - "cross_document_synthesis" -- combines claims across multiple
                           cited works that weren't previously connected
                         - "contradiction"            -- resolves or exploits a conflict
                           among the cited works
                         - "scope_extension"          -- relaxes a specific assumption the
                           cited work depends on
                         - "reframing"                -- proposes a materially different
                           framing of the phenomenon the cited work studies

Write the `insight.claim` so it reads the same whether or not the idea has been
tested yet. Describe the mechanism/claim and its underlying rationale, NOT the
outcome. Specifically:
  - Do NOT report results, metrics, benchmark numbers, or comparative claims
    ("improves accuracy by X%", "outperforms baselines", "achieves SOTA",
    "we show that ...").
  - Do NOT use evaluation verbs ("demonstrate", "show", "prove", "achieve").
  - DO state the mechanism / design / hypothesis ("treat X as Y", "use A to
    do B so that C", "the key idea is to ...").
  - The `insight` is NOT a prediction awaiting test (that's a hypothesis), NOT a
    plan of action (that's the method), and NOT a restatement of stakes (that's
    motivation).

Keep the `problem` broad enough that the `insight` is doing real
narrowing work: if the problem already contains the insight's commitments, state
the problem more broadly. Be concise and specific. If the abstract does not state
the motivation explicitly, infer the most likely stakes from the problem and insight.

=== PAPER ===
Title: {title}

Abstract / Description:
{body}

Return ONLY a JSON object of the form:
{{
  "motivation": "...",
  "problem": "...",
  "insight": {{
    "claim": "...",
    "grounding": [
      {{"source": "<cited work as the paper names it>",
       "supports": "<what this work establishes that the claim builds on>"}}
    ],
    "grounding_type": "in_domain_extension"
  }}
}}
Do not output anything outside the JSON object."""


# TARGET_IDEA template: target-paper idea recovery with `insight` as a FLAT
# string (no grounding object, no grounding_type). Use when you want a clean
# target-side idea card whose insight isn't anchored to the target paper's own
# citations (those don't share a coordinate system with the record's source
# set). `target_idea_with_grounding` uses IDEA_TEMPLATE instead, which keeps
# the nested {claim, grounding[], grounding_type} object.
TARGET_IDEA_TEMPLATE = """Summarize the paper below into a structured
research-idea card. The card has three top-level fields -- `motivation`,
`problem`, `insight` -- that play DIFFERENT roles; a card whose fields collapse
into restatements of each other is not useful. Read the definitions carefully.

  motivation:  The outward-facing stakes -- why anyone beyond the authors should
               care if this progresses. Who is affected, what becomes possible
               downstream, what limitation it would lift or cost it would reduce.
               Answer "if this worked, what would change in the world, and for
               whom?" -- not merely the gap in prior work (1-2 sentences).
  problem:     The field-level research question -- the broad challenge the work
               takes on, of the kind many researchers in the area might
               independently arrive at. Name the object of study and the
               difficulty or unknown about it, WITHOUT committing to the specific
               angle, mechanism, or approach this paper uses (that belongs in
               `insight`). It is the *what*, distinct from the *why* (motivation)
               and the *how* (method) (1-2 sentences).
  insight:     The non-obvious observation, mechanism, or reframing -- the
               load-bearing belief that gives the problem a specific, newly
               tractable angle and distinguishes this work from others on the
               same problem. A claim about how the world *is* (an asymmetry, a
               hidden dependency, a failed assumption, a reinterpretation),
               stated as the idea itself, independent of how well it turned out.
               WHAT approach/mechanism/hypothesis does the work introduce, and
               WHY would it work? Plain string (1-3 sentences).

Write `insight` so it reads the same whether or not the idea has been tested.
Describe the mechanism/claim and its underlying rationale, NOT the outcome.
Specifically:
  - Do NOT report results, metrics, benchmark numbers, or comparative claims
    ("improves accuracy by X%", "outperforms baselines", "achieves SOTA",
    "we show that ...").
  - Do NOT use evaluation verbs ("demonstrate", "show", "prove", "achieve").
  - DO state the mechanism / design / hypothesis ("treat X as Y", "use A to do
    B so that C", "the key idea is to ...").
  - The `insight` is NOT a prediction awaiting test (that's a hypothesis), NOT a
    plan of action (that's the method), and NOT a restatement of stakes (that's
    motivation).

Keep the `problem` broad enough that the `insight` is doing real narrowing
work: if the problem already contains the insight's commitments, state the
problem more broadly. Be concise and specific. If the abstract does not state
the motivation explicitly, infer the most likely stakes from the problem and
insight.

=== PAPER ===
Title: {title}

Abstract / Description:
{body}

Return ONLY a JSON object of the form:
{{
  "motivation": "...",
  "problem": "...",
  "insight": "..."
}}
Do not output anything outside the JSON object."""


# SOURCE_TO_FULL_SUMM: the idea-then-outcome split. problem/motivation/insight
# stay outcome-independent (insight identical in spirit to the idea card);
# method/results/takeaways carry the empirical story, with the numbers.
FULL_SUMM_SYSTEM = (
    "You are an expert research scientist. You read a paper (title plus full "
    "text or abstract) and distill it into a structured six-field summary that "
    "covers both the IDEA and what the work actually did and found. Be faithful "
    "to the source: report only what the paper claims, and use the numbers it "
    "states. Return only a single JSON object."
)

FULL_SUMM_TEMPLATE = """Summarize the paper below into a six-field summary. The
motivation / problem / insight fields play DIFFERENT roles -- do not
let them collapse into restatements of each other. Read the definitions carefully.

  motivation:        The outward-facing stakes -- why anyone beyond the authors
                     should care if this progresses. Who is affected, what becomes
                     possible downstream, what limitation it would lift or cost it
                     would reduce. Answer "if this worked, what would change in the
                     world, and for whom?" -- not merely the gap in prior work
                     (1-2 sentences).
  problem:           The field-level research question -- the broad challenge the
                     work takes on, of the kind many researchers in the area might
                     independently arrive at. Name the object of study and the
                     difficulty or unknown about it, WITHOUT committing to the
                     specific angle, mechanism, or approach this paper uses (that
                     belongs in `insight`). It is the *what*, distinct from the
                     *why* (motivation) and the *how* (method) (1-2 sentences).
  insight:           The non-obvious observation, mechanism, or reframing -- the
                     load-bearing belief that gives the problem a specific, newly
                     tractable angle and distinguishes this work from others on the
                     same problem. It is a claim about how the world *is* (an
                     asymmetry, a hidden dependency, a failed assumption, a
                     reinterpretation), stated as the idea itself, independent of
                     how well it turned out. WHAT approach/mechanism/hypothesis does
                     the work introduce, and WHY would it work? (1-2 sentences).
  method:            HOW the idea is realized and evaluated -- the approach, system,
                     model, data, and experimental setup (2-4 sentences).
  results:           The main findings -- key quantitative/qualitative outcomes,
                     comparisons against baselines, notable ablations. Report the
                     concrete numbers the paper states where available (2-4 sentences).
  takeaways:         Implications -- what worked and what did not, limitations, and
                     what the result enables or rules out going forward (1-3 sentences).

IMPORTANT split between `insight` and the rest:
  - `insight` must read the SAME whether or not the idea has been tested. State
    the mechanism / claim / hypothesis ("the key idea is to ...", "treat X as
    Y"); do NOT put results, metrics, or evaluation verbs ("show", "achieve",
    "outperform") in it. It is NOT a prediction awaiting test, NOT a plan of
    action (that's the method), and NOT a restatement of stakes (that's motivation).
  - Keep `problem` broad enough that the `insight` is doing real
    narrowing work; if the problem already contains the insight's commitments,
    state the problem more broadly.
  - `method`, `results`, and `takeaways` are where the empirical story belongs.
    Put the experimental setup in `method`, the outcomes and numbers in
    `results`, and the implications in `takeaways`.

Be concise and specific. If a field is not stated in the source, infer the most
likely answer from the rest of the paper; if the source is only an abstract,
fill `results` with whatever outcomes it reports (it may be brief).

=== PAPER ===
Title: {title}

Abstract / Description:
{body}

Return ONLY a JSON object of the form:
{{
  "motivation": "...",
  "problem": "...",
  "insight": "...",
  "method": "...",
  "results": "...",
  "takeaways": "..."
}}
Do not output anything outside the JSON object."""


# SI_SOURCE_SUMM: the SI-format source card. This is the SI analogue of
# SOURCE_TO_FULL_SUMM -- one card per source document -- but in the SI field
# vocabulary (problem_gap / core_claim / method) plus the empirical outcomes
# (results / takeaways). The SI spine fields are drawn from diff_data_templates
# (_F_PROBLEM_GAP / _F_METHOD), with `core_claim` rewritten to the INSIGHT-style
# definition (a paper's load-bearing claim about how the world *is*, stated
# outcome-independently) because we are SUMMARIZING an existing paper, not
# generating an idea. results/takeaways reuse FULL_SUMM's exact definitions. No
# `inspiration` on a source card: a source's own citations do not share the
# record's Document N coordinate system.
SI_SOURCE_SUMM_SYSTEM = (
    "You are an expert research scientist. You read a paper (title plus full "
    "text or abstract) and distill it into a compact, structured card in a fixed "
    "SI schema (problem_gap / core_claim / method) plus its empirical outcomes "
    "(results / takeaways). The spine (problem_gap / core_claim / method) captures "
    "the IDEA -- the conceptual approach -- stated independent of how well it "
    "turned out; results / takeaways carry what the work actually found. Be "
    "faithful to the source: report only what the paper claims, and use the "
    "numbers it states. Return only a single JSON object."
)

SI_SOURCE_SUMM_TEMPLATE = """Summarize the paper below into a structured card
with five fields -- `problem_gap`, `core_claim`, `method`, `results`,
`takeaways`. The spine fields (problem_gap / core_claim / method) play DIFFERENT
roles; a card whose fields collapse into restatements of each other is not
useful. Read the definitions carefully.

  problem_gap:  The gap this paper addresses -- the open question, tension, or
                unresolved difficulty in the area that motivates the work. State
                it as one clear gap: a contradiction, a missing explanation, a
                scope mismatch, an untested assumption, a disconnect between
                lines of work, or a bottleneck holding the area back. Name the
                gap itself, NOT the paper's specific solution to it (1-2
                sentences).
  core_claim:   The paper's load-bearing, non-obvious claim -- the observation,
                mechanism, or reframing that gives the problem a specific, newly
                tractable angle and distinguishes this work from others on the
                same problem. It is a claim about how the world *is* (an
                asymmetry, a hidden dependency, a failed assumption, a
                reinterpretation), stated as the idea itself, independent of how
                well it turned out. WHAT approach/mechanism/hypothesis does the
                work introduce, and WHY would it work? (1-2 sentences).
  method:       At a high level, HOW the paper turns the claim into research --
                the *kind* of move it makes (combining ideas, extending a result
                to a new setting, making a fragile result robust, deriving
                something formally, mapping a space empirically, building an
                artifact) and how that move addresses the gap. Keep it at the
                level of an approach and setup, not a step-by-step protocol (2-3
                sentences).
  results:      The main findings -- key quantitative/qualitative outcomes,
                comparisons against baselines, notable ablations. Include
                negative or null results and approaches the paper reports did
                NOT work, not only the wins. Report the concrete numbers the
                paper states where available (2-4 sentences). Do NOT report the
                paper's own untested conjectures as if they were findings --
                those belong in takeaways.
  takeaways:    Implications -- what worked and what did not, limitations, and
                what the result enables or rules out going forward. If the paper
                makes speculative claims or conjectures it does not test, note
                them here and mark them as speculative (1-3 sentences).

IMPORTANT split between the spine and the outcomes:
  - `problem_gap`, `core_claim`, and `method` must read the SAME whether or not
    the idea has been tested. State the gap, the mechanism/claim, and the
    approach; do NOT put results, metrics, benchmark numbers, or evaluation verbs
    ("show", "achieve", "outperform", "improves accuracy by X%") in them.
  - `core_claim` is NOT a prediction awaiting test (that's a hypothesis), NOT a
    plan of action (that's the method), and NOT a restatement of the gap.
  - Keep `problem_gap` broad enough that the `core_claim` is doing real narrowing
    work; if the gap already contains the claim's commitments, state the gap more
    broadly.
  - `results` and `takeaways` are where the empirical story and the numbers live.

Be concise and specific. If a field is not stated in the source, infer the most
likely answer from the rest of the paper; if the source is only an abstract, fill
`results` with whatever outcomes it reports (it may be brief).

=== PAPER ===
Title: {title}

Abstract / Description:
{body}

Return ONLY a JSON object of the form:
{{
  "problem_gap": "...",
  "core_claim": "...",
  "method": "...",
  "results": "...",
  "takeaways": "..."
}}
Do not output anything outside the JSON object."""


# --------------------------------------------------------------------------- #
# RECORD-scoped prompts (whole source set per call)
# --------------------------------------------------------------------------- #
# IDEA_GROUNDED: the GOLD target-idea card that finally mirrors v0 grounding.
# Unlike TARGET_IDEA (which sees only the target's own body and grounds in the
# target's own citation notation), this is handed the SAME inspiration set the
# inference model sees -- the source FULL SUMMARIES, numbered Document 1..N --
# plus the target paper. (It uses the full summaries, not the idea cards, so its
# "Document N" numbering is byte-identical to what live inference / the SFT
# prompt show -- see input_full_summaries_str in inference/dataloader.py. This
# alignment is what lets the gold idea be grafted into a SOURCE_MULTIPLE_OUTPUT
# answer without its grounding refs pointing at the wrong source.) It distils
# the target's single core idea and grounds the insight IN that provided set,
# using "Document N" references. Output is ONE idea object in the exact v0
# schema, so it is directly comparable, field by field, to what inference emits.
IDEA_GROUNDED_SYSTEM = (
    "You are an expert research scientist. You are given (a) a numbered set of "
    "summarized inspiration documents and (b) the full text of a target paper "
    "that was written building on those inspirations. Your job is to recover the "
    "single core research idea of the target paper and express it in a fixed "
    "schema, grounding its insight in the numbered inspiration documents you were "
    "given -- NOT in the target paper's own bibliography. Be faithful to the "
    "target paper: do not invent contributions it does not make. Return only a "
    "single JSON object."
)

IDEA_GROUNDED_TEMPLATE = """You are given a set of inspiration documents (each an
already-summarized paper) and the full text of a TARGET PAPER that builds on
them. Recover the target paper's SINGLE core research idea and express it as one
idea object in the schema below. The three fields play DIFFERENT roles; a card
whose fields collapse into restatements of each other is not useful.

  motivation:  The outward-facing stakes of the target paper's idea -- why anyone
               beyond the authors should care if it progresses. Name a concrete
               beneficiary and a concrete consequence, not generic importance
               (1-2 sentences).
  problem:     The field-level research question the target paper takes on -- broad
               enough that the insight is doing real narrowing work, stated WITHOUT
               committing to the paper's specific angle or mechanism (1-2 sentences).
  insight:     The target paper's load-bearing idea, expressed as a definite
               proposition positioned against the numbered inspiration documents.
               Stated independent of how well it turned out (no results, metrics,
               or evaluation verbs).

               The `claim` text should make three things visible in its prose:
                 (1) WHAT THE DOCS DO — the prior position the target paper's
                     move stands against, drawn from the cited inspiration
                     documents.
                 (2) THE CLAIM — the target paper's proposition (asymmetry,
                     mechanism, missed assumption, alternative framing). Not
                     hedging.
                 (3) THE BRIDGE — why what the inspiration docs do implies,
                     fails to address, or can be reinterpreted to yield the
                     target paper's claim. A "therefore" with no real reasoning
                     step is not a bridge.

               A reader should be able to answer "what's the claim?", "is this
               in the inspiration docs already?", and "how did the target paper
               get from the docs to here?" from the `claim` text alone.

               `insight` is an OBJECT with three sub-fields:
                 claim:          The insight itself, showing the three parts
                                 above in its prose.
                 grounding:      A list of one or more references INTO THE NUMBERED
                                 INSPIRATION DOCUMENTS BELOW that the target paper's
                                 insight builds on. Each entry is
                                 {{"source": "Document <N>", "supports": "<what that
                                 document establishes that the insight builds on>"}}.
                                 Cite by the "Document N" label, NOT by the target
                                 paper's own citations. If the target's insight
                                 genuinely builds on none of the provided documents,
                                 return an empty list rather than inventing a link.
                 grounding_type: one of
                   - "in_domain_extension"      -- builds directly on a provided
                     document in the problem's home domain (does X+1 where it does X)
                   - "cross_domain_transfer"    -- imports a mechanism from a provided
                     document in a different domain
                   - "cross_document_synthesis" -- combines claims across MULTIPLE
                     provided documents that weren't previously connected
                   - "contradiction"            -- resolves or exploits a conflict
                     among the provided documents
                   - "scope_extension"          -- relaxes a specific assumption a
                     provided document depends on
                   - "reframing"                -- proposes a materially different
                     framing of a phenomenon a provided document studies

Write the `insight.claim` so it reads the same whether or not the idea has been
tested: state the mechanism/claim/hypothesis, NOT the outcome. Do NOT report
results, metrics, or use evaluation verbs ("show", "achieve", "outperform").

The insight is NOT a hypothesis (prediction awaiting test), NOT a method (plan
of action), NOT a motivation (stakes), and NOT a category broadened to include
obvious cases ("X is also caused by Y").

═══════════════════════════════════════
INSPIRATION DOCUMENTS
═══════════════════════════════════════

{documents}

═══════════════════════════════════════
TARGET PAPER
═══════════════════════════════════════
Title: {title}

Full text:
{body}

Return ONLY a JSON object of the form:
{{
  "motivation": "...",
  "problem": "...",
  "insight": {{
    "claim": "...",
    "grounding": [
      {{"source": "Document <N>",
       "supports": "<what this document establishes that the insight builds on>"}}
    ],
    "grounding_type": "in_domain_extension"
  }}
}}
Do not output anything outside the JSON object."""


# SI_IDEA_GROUNDED: the GOLD SI-format target card, the SI analogue of
# IDEA_GROUNDED. Handed the SAME inspiration set the inference model sees -- but
# numbered from the SI SOURCE cards (input_si_summaries_str), NOT the v0 full
# summaries -- plus the target paper. It recovers the target's single core idea
# and expresses it in the 4 SI fields [problem_gap, core_claim, method,
# inspiration]. Unlike the source card, here `core_claim` keeps the IDEATION-style
# "we claim that X" bet wording (from diff_data_templates _F_CORE_CLAIM), because
# this card is the TRAINING TARGET -- its core_claim should read as the generative
# idea the model learns to produce. Grounding lives in the SI `inspiration`
# field, now PROSE and deliberately CRITICAL: it names each cited Document's
# limitation (what it got wrong / stopped short of) and argues why connecting them
# is non-obvious, so the idea reads as a response to a concrete shortcoming rather
# than a credit ledger, referencing documents by their exact "Document N" labels;
# any genuine external influence is named explicitly in the prose and carries no
# Document N.
SI_IDEA_GROUNDED_SYSTEM = (
    "You are an expert research scientist. You are given (a) a numbered set of "
    "summarized inspiration documents and (b) the full text of a target paper "
    "that was written building on those inspirations. Your job is to recover the "
    "single core research idea of the target paper and express it in a fixed SI "
    "schema (problem_gap / core_claim / method / inspiration), grounding it in the "
    "numbered inspiration documents you were given -- NOT in the target paper's "
    "own bibliography. Be faithful to the target paper: do not invent "
    "contributions it does not make. Return only a single JSON object."
)

SI_IDEA_GROUNDED_TEMPLATE = """You are given a set of inspiration documents (each
an already-summarized paper) and the full text of a TARGET PAPER that builds on
them. Recover the target paper's SINGLE core research idea and express it as one
idea object in the SI schema below. The fields play DIFFERENT roles; a card whose
fields collapse into restatements of each other is not useful.

  problem_gap:  The gap the target paper addresses, drawn from across the
                inspiration documents -- the open question, tension, or
                unresolved difficulty that connects them and that none resolves
                on its own. State it as one clear gap, NOT as the target paper's
                specific solution to it. BEGIN this field with the single
                best-fitting OPPORTUNITY PATTERN label in square brackets, chosen
                from the taxonomy below, then the prose gap (1-2 sentences). Pick
                the ONE dominant pattern even if several apply. If NONE genuinely
                fits, use "[Other]" rather than forcing a poor match -- a wrong
                label is worse than "[Other]". Example:
                "[Bridge Opportunity] Work on X and on Y each ... but neither
                connects ...".
  core_claim:   The couple of sentences the whole idea lives or dies on -- "we
                claim that X." It must be something that could turn out to be
                wrong: a definite, testable assertion/hypothesis/insight, not a
                topic or a direction. Do not just restate the gap, and do not
                describe what you would do about it. From these sentences alone a
                reader should know exactly what the target paper is betting on
                (1-2 sentences).
  method:       At a high level, HOW the claim WOULD BE turned into research --
                the *kind* of move (combining ideas, extending to a new setting,
                making a fragile result robust, deriving formally, mapping a space
                empirically, building an artifact) and how that move addresses the
                gap. Write it as a PROPOSED approach, forward-looking, as if the
                idea has not yet been carried out: use present/infinitive framing
                ("to test this, X"; "the approach is to X"), NOT past tense and
                NOT "the authors/the paper do X". Do not narrate what was done or
                built. Concrete specifics (a setting, a target system, a dataset)
                may be named, but state them as part of the plan, never as
                accomplished facts. Keep it at the level of an approach, NOT a
                step-by-step protocol. BEGIN this field with the single
                best-fitting METHOD PARADIGM label in square brackets, chosen from
                the taxonomy below, then the prose approach (2-3 sentences). Pick
                the ONE dominant paradigm even if several apply. If NONE genuinely
                fits, use "[Other]" rather than forcing a poor match -- a wrong
                label is worse than "[Other]". Example:
                "[Synthesis / Unification] To test this, the approach is to ...".
  inspiration:  Prose (three to five sentences) that ARGUES how the idea arises
                FROM the inspiration documents -- the relationship BETWEEN them,
                not a per-document restatement. This field must be CRITICAL, not a
                credit ledger. Two things it must do:
                (1) NAME EACH CITED SOURCE'S LIMITATION -- for every Document you
                    lean on, say what it got wrong, missed, stopped short of, or
                    left unresolved, so the idea reads as a RESPONSE to a concrete
                    shortcoming, not a recombination of things that already worked.
                    ("Document 3 measured X but only under assumption A"; "Document
                    5 proposed Y yet never addressed Z".)
                (2) JUSTIFY THE NON-OBVIOUS LEAP -- argue WHY connecting these
                    documents is non-trivial: the tension, friction, or mismatch
                    that had to be resolved for the pieces to fit, so the idea does
                    not read as the obvious next step anyone would take. If the
                    connection IS routine, say that plainly rather than inflating
                    it.
                Then show how the pieces combine. The idea may (a) COMBINE sources
                -- name which supplied the gap vs. the method vs. the analogy; (b)
                resolve a TENSION or contradiction between them; (c) reframe or
                invert what a source assumed; or (d) rest mainly on an idea
                imported from OUTSIDE the provided documents, with the documents in
                support. Refer to the numbered documents by their exact "Document
                <N>" labels wherever the idea genuinely builds on them, and say
                which move it is; if you drew on any outside idea or analogy, name
                that external source explicitly (it will carry no Document <N>). Do
                not invent a document link; if the idea rests on a single document
                with no combination, say so plainly -- and still name that
                document's limitation and why the idea's step beyond it is not
                obvious -- rather than inventing a connection.

Write `problem_gap`, `core_claim`, and `method` so they read the same whether or
not the idea has been tested: state the gap, the claim/bet, and the approach, NOT
the outcome. Do NOT report results, metrics, or use evaluation verbs ("show",
"achieve", "outperform"). Do NOT narrate the work as already completed: avoid
past tense and completed-action verbs ("designed", "built", "ran", "evaluated",
"conducted") and do not attribute actions to "the authors" or "the paper" -- the
card describes a forward-looking idea, not a recounting of what a paper did. Keep
`problem_gap` broad enough that the `core_claim` is doing real narrowing work.

═══════════════════════════════════════
TAXONOMY (for the required leading [labels])
═══════════════════════════════════════
`problem_gap` MUST begin with exactly ONE of these OPPORTUNITY PATTERNS, in
square brackets and written verbatim:
  [Puzzle / Contradiction]  -- a paradox, tradeoff, surprising failure, or
        conflicting evidence.
  [Explanation Gap]         -- a missing causal, mechanistic, theoretical, or
        explanatory account.
  [Scope Mismatch]          -- unrealistic assumptions, narrow regimes, unclear
        transferability, or boundary conditions.
  [Evidence Gap]            -- missing ways to observe, measure, benchmark,
        audit, diagnose, validate, compare, or accumulate evidence.
  [Bridge Opportunity]      -- disconnected literatures, theories, evidence
        streams, communities, or methods that could be connected.
  [Failure / Risk Gap]      -- brittleness, unreliability, bias, uncertainty,
        safety/privacy/security risk, or reproducibility concerns.
  [Resource Bottleneck]     -- cost, compute, time, data access, sample scarcity,
        experimental burden, deployment friction, usability, or scalability.

`method` MUST begin with exactly ONE of these METHOD PARADIGMS, in square
brackets and written verbatim:
  [Synthesis / Unification] -- bridges, integrates, reconciles, or unifies
        separate literatures, theories, evidence streams, mechanisms, or methods.
  [Relax / Extend Scope]    -- makes prior work function under weaker assumptions,
        broader scope, new regimes, noisier conditions, or realistic settings.
  [Robustification]         -- reduces failures, brittleness, risk, uncertainty,
        bias, unreliability, or trustworthiness problems.
  [Formal Derivation]       -- introduces a formal model, theorem, bound,
        objective, derivation, proof, ontology, taxonomy, conceptual distinction,
        or explanatory formulation.
  [Empirical Mapping]       -- builds or applies systematic measurement,
        benchmarks, diagnostics, datasets, empirical maps, comparative studies,
        or pattern analyses.
  [Artifact / System]       -- builds a concrete artifact, software system,
        platform, device, material, prototype, or deployment workflow as the
        central contribution.
  [Optimization / Search]   -- uses optimization, search, screening, tuning,
        active/adaptive design, scaling, resource allocation, or efficiency
        strategies to discover or improve a solution.

If, and only if, none of the seven listed labels on an axis genuinely fits, you
may use "[Other]" for that axis -- do not stretch a listed label to force a fit,
and do not invent any label outside this list plus "[Other]".

Use the bracketed label ONLY at the very start of the field; do not repeat it in
the prose or add labels to any other field.

═══════════════════════════════════════
INSPIRATION DOCUMENTS
═══════════════════════════════════════

{documents}

═══════════════════════════════════════
TARGET PAPER
═══════════════════════════════════════
Title: {title}

Full text:
{body}

Return ONLY a JSON object of the form:
{{
  "problem_gap": "[<one opportunity pattern>] ...",
  "core_claim": "...",
  "method": "[<one method paradigm>] ...",
  "inspiration": "... Document <N> did X but stopped short of / assumed / never addressed ...; Document <M> offered the method yet ...; connecting them is non-obvious because ... (name any external source explicitly) ..."
}}
Do not output anything outside the JSON object."""


# SOURCE_MULTIPLE_OUTPUT: the SFT-data generator. This IS the live inference
# prompt (IDEATION_V0_TEMPLATE) -- imported, not copied -- run over a record's
# source summaries to emit K ideas in the <think>/<answer> shape. Its `system`
# is None (the v0 template is self-contained) and it has no fixed-field card:
# the output is a JSON *list* parsed by inference/parsing.py, not _parse_card.
SOURCE_MULTIPLE_OUTPUT_TEMPLATE = IDEATION_V0_TEMPLATE


# --------------------------------------------------------------------------- #
# SummaryType enum + spec registry
# --------------------------------------------------------------------------- #
class SummaryType(str, Enum):
    """The single key for every summary type: registry key, --types CLI choice,
    and the value callers branch on. `str`-valued so `st.value` is the stable
    on-disk / CLI string and `SummaryType("target_idea")` round-trips."""
    TARGET_IDEA = "target_idea"                              # FLAT insight (no grounding)
    TARGET_IDEA_WITH_GROUNDING = "target_idea_with_grounding" # NESTED insight (with grounding)
    SOURCE_TO_IDEA = "source_to_idea"
    SOURCE_TO_FULL_SUMM = "source_to_full_summ"
    IDEA_GROUNDED = "idea_grounded"
    SOURCE_MULTIPLE_OUTPUT = "source_multiple_output"
    SI_SOURCE_SUMM = "si_source_summ"                        # SI source card (spine + outcomes)
    SI_IDEA_GROUNDED = "si_idea_grounded"                    # SI gold target card (grounded)


# Idea-card output fields (the nested-insight card shape).
_IDEA_FIELDS = ["motivation", "problem", "insight"]
_FULL_SUMM_FIELDS = ["motivation", "problem", "insight",
                     "method", "results", "takeaways"]
# SI-format card fields. The source card carries the SI spine + empirical
# outcomes; the gold target card carries the SI spine + the `inspiration`
# grounding field (prose on how the idea arises from / combines the numbered
# documents, naming any external source explicitly).
_SI_SOURCE_FIELDS = ["problem_gap", "core_claim", "method", "results", "takeaways"]
_SI_IDEA_FIELDS = ["problem_gap", "core_claim", "method", "inspiration"]

# Each spec carries everything summarize_document.py needs to run a type:
#   scope     -- Scope: what one LLM call consumes (drives task collection)
#   fields    -- card fields parsed out (None => not a fixed-field card, e.g.
#                the v0 list output, which is parsed by inference/parsing.py)
#   system    -- system prompt (None when the user template is self-contained)
#   template  -- user prompt; .format placeholders depend on scope:
#                  TARGET_DOC / SOURCE_DOC  -> {title}, {body}
#                  RECORD (IDEA_GROUNDED)   -> {documents}, {title}, {body}
#                  RECORD (SOURCE_MULTIPLE) -> v0 .replace placeholders
#                                              {num_ideas}, {documents}
#   list_output -- True when the model emits a JSON list of ideas (v0), not a card
#
# RECORD-scoped specs carry two extra keys consumed by collect_tasks:
#   source_str_field -- the record field holding the per-source summary strings
#                       that become the numbered Document block (these must
#                       already be present, i.e. a SECOND pass over a
#                       *_with_summaries.jsonl). BOTH record types use the
#                       source full summaries (input_full_summaries_str) -- the
#                       same set, numbered identically -- so the IDEA_GROUNDED
#                       gold card can be grafted into a SOURCE_MULTIPLE_OUTPUT
#                       answer with its "Document N" refs still valid. This set
#                       is byte-identical to live inference input.
#   needs_target     -- whether the target paper's full text is also supplied
#                       (IDEA_GROUNDED: yes; SOURCE_MULTIPLE_OUTPUT: no).
SUMMARY_SPECS: Dict[SummaryType, dict] = {
    # target_idea = the FLAT (no-grounding) version: insight is a plain string.
    SummaryType.TARGET_IDEA: {
        "scope": Scope.TARGET_DOC,
        "fields": list(_IDEA_FIELDS),
        "system": IDEA_SYSTEM,
        "template": TARGET_IDEA_TEMPLATE,
        "list_output": False,
    },
    # target_idea_with_grounding = the NESTED version (insight is an object with
    # claim/grounding[]/grounding_type). Grounding is in the target paper's OWN
    # citation notation, which does not share a coordinate system with the
    # record's source set -- prefer idea_grounded for inference-aligned gold.
    SummaryType.TARGET_IDEA_WITH_GROUNDING: {
        "scope": Scope.TARGET_DOC,
        "fields": list(_IDEA_FIELDS),
        "system": IDEA_SYSTEM,
        "template": IDEA_TEMPLATE,
        "list_output": False,
    },
    SummaryType.SOURCE_TO_IDEA: { # maybe i do not need this
        "scope": Scope.SOURCE_DOC, 
        "fields": list(_IDEA_FIELDS),
        "system": IDEA_SYSTEM,
        "template": IDEA_TEMPLATE,
        "list_output": False,
    },
    SummaryType.SOURCE_TO_FULL_SUMM: {
        "scope": Scope.SOURCE_DOC, # this is input 
        "fields": list(_FULL_SUMM_FIELDS),
        "system": FULL_SUMM_SYSTEM,
        "template": FULL_SUMM_TEMPLATE,
        "list_output": False,
    },
    SummaryType.IDEA_GROUNDED: {
        "scope": Scope.RECORD, # This can be incorporated as one of the outputs in the SFT data...
        "fields": list(_IDEA_FIELDS),
        "system": IDEA_GROUNDED_SYSTEM,
        "template": IDEA_GROUNDED_TEMPLATE,
        "list_output": False,
        # Full summaries (NOT idea cards): the same set live inference shows, so
        # "Document N" numbering matches the SFT prompt and the gold card can be
        # grafted into a SOURCE_MULTIPLE_OUTPUT answer without ref drift.
        "source_str_field": "input_full_summaries_str",
        "needs_target": True,
    },
    SummaryType.SOURCE_MULTIPLE_OUTPUT: {
        "scope": Scope.RECORD, ## This is SFT data
        "fields": None,
        "system": None,
        "template": SOURCE_MULTIPLE_OUTPUT_TEMPLATE,
        "list_output": True,
        "source_str_field": "input_full_summaries_str",
        "needs_target": False,
    },
    # SI PASS 1: per-source SI card (spine + empirical outcomes). SOURCE_DOC-
    # scoped, like source_to_full_summ, but emits the SI field vocabulary. Feeds
    # the input_si_summaries[_str] block that si_idea_grounded numbers over.
    SummaryType.SI_SOURCE_SUMM: {
        "scope": Scope.SOURCE_DOC,
        "fields": list(_SI_SOURCE_FIELDS),
        "system": SI_SOURCE_SUMM_SYSTEM,
        "template": SI_SOURCE_SUMM_TEMPLATE,
        "list_output": False,
    },
    # SI PASS 2: gold SI target card grounded into the SI source set. RECORD-
    # scoped. Numbers its Document N block from the SI source cards
    # (input_si_summaries_str, the NEW field), NOT the v0 full summaries, so the
    # gold card's "Document N" refs point at SI source cards.
    SummaryType.SI_IDEA_GROUNDED: {
        "scope": Scope.RECORD,
        "fields": list(_SI_IDEA_FIELDS),
        "system": SI_IDEA_GROUNDED_SYSTEM,
        "template": SI_IDEA_GROUNDED_TEMPLATE,
        "list_output": False,
        "source_str_field": "input_si_summaries_str",
        "needs_target": True,
    },
}

ALL_SUMMARY_TYPES = tuple(SummaryType)


def summary_type(value) -> SummaryType:
    """Coerce a str / SummaryType to a SummaryType (raises ValueError if bad)."""
    return value if isinstance(value, SummaryType) else SummaryType(value)
