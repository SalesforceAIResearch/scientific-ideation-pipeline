"""The v0 ideation (inference) prompt and its document-block helpers.

Moved here from the former `inference/prompts.py` so that both inference and the
dataset-generation pipeline (`dataset_creation/part_b/summarize_document.py`)
share ONE copy of the prompt text. The record-level `SOURCE_MULTIPLE_OUTPUT`
summary type reuses `IDEATION_V0_TEMPLATE` verbatim, so SFT-generation input is
identical to live inference input.

The active template is **v0** (see thinking/reward_formulations.md ->
"Prompt v0 — initial draft for training"): the model is given a set of
summarized inspiration documents and asked to generate `num_ideas` distinct
research ideas. Each idea is a `{motivation, problem, insight}` triple, where
`insight` is itself a structured object carrying `claim`, `grounding`, and
`grounding_type`. Reasoning goes in <think></think>; the final answer is a JSON
*list* of idea objects inside <answer></answer>.

The number of ideas is a parameter (`num_ideas`) threaded through build_prompt /
load_examples / the --num-ideas CLI flag, so the same template serves K=8 (the
reward-doc default) or any other count.
"""

from typing import List

# NOTE: plain string + .replace (NOT an f-string / .format): the JSON schema
# below contains literal { } braces that an f-string would treat as fields.
# Placeholders: {num_ideas} and {documents}.
IDEATION_V0_TEMPLATE = """You are a research ideation assistant. Given a set of source documents, your job is to generate {num_ideas} distinct research ideas that could inspire a scientist working in or adjacent to the area covered by these documents.

Each idea must follow the schema below. Read the definitions carefully — the three fields play different roles, and ideas that collapse them into restatements of each other are not useful.

═══════════════════════════════════════
SCHEMA
═══════════════════════════════════════

1. MOTIVATION
The outward-facing stakes of the work — the case for why anyone beyond the
authors should care if progress is made. Situate the work in a broader
context: who is affected, what becomes possible downstream, what current
limitation it would lift, or what cost it would reduce. Name a concrete
beneficiary (a community, a discipline, a class of applications) and a
concrete consequence (capability unlocked, harm reduced, understanding
gained), rather than gesturing at general importance. Answer: "if this
worked, what would change in the world, and for whom?"

2. PROBLEM
The field-level research question — the broad challenge being posed, of the
kind many researchers in the area might independently arrive at. Name the
object of study and the difficulty or unknown about it, but do NOT commit to
a specific angle, mechanism, or constraint on how it should be approached. A
well-posed problem is intelligible to anyone in the field, concrete enough
that two researchers would agree on what counts as progress, and broad
enough that multiple distinct research directions could attack it. It is
the *what*, distinct from both the *why* (motivation) and the *how*
(methodology). State it without committing to any particular insight or
approach.

3. INSIGHT
A definite proposition positioned against the cited documents. The `claim`
text should make three things visible in its prose:

  (1) WHAT THE DOCS DO — the prior position your move stands against,
      drawn from the cited documents.
  (2) THE CLAIM — your proposition (asymmetry, mechanism, missed
      assumption, alternative framing). Not hedging.
  (3) THE BRIDGE — why what the docs do implies, fails to address, or
      can be reinterpreted to yield the claim. This is the move that
      makes the claim follow from the docs rather than appear from
      nowhere. A "therefore" with no real reasoning step is not a bridge.

A reader should be able to answer "what's the claim?", "is this in the
docs already?", and "how did you get from the docs to here?" from the
`claim` text alone.

The insight is NOT a hypothesis (prediction awaiting test), NOT a method
(plan of action), NOT a motivation (stakes), and NOT a category broadened
to include obvious cases ("X is also caused by Y").

Each insight must additionally include:
  • GROUNDING: one or more specific references to source material
    (document IDs, papers, or established findings) that the claim builds
    on. Quote or paraphrase the supporting finding briefly.
  • GROUNDING_TYPE: one of
      - "in_domain_extension"      — builds directly on findings in the
        problem's home domain (does X+1 where the docs do X)
      - "cross_domain_transfer"    — imports a mechanism established in a
        different domain and applies it to this problem
      - "cross_document_synthesis" — combines claims across multiple
        sources that weren't previously connected; the insight requires
        the combination to be supported
      - "contradiction"            — resolves or exploits a conflict among
        the cited docs; names the disagreement and what it surfaces
      - "scope_extension"          — relaxes a specific assumption the
        cited docs depend on, and describes what the relaxation opens up
      - "reframing"                — proposes a materially different
        framing of the phenomenon the cited docs study, and describes
        what the new framing makes visible

═══════════════════════════════════════
QUALITY BAR (per idea)
═══════════════════════════════════════

i
• Problem is broad enough that the insight is doing real narrowing work. If the problem already contains the insight's commitments, broaden the problem.
• Insight `claim` text makes three things visible: what the cited docs do, the new claim, and the bridge from the former to the latter. A reader should answer "what's the claim?", "is this already in the docs?", and "how did you get here?" from the text alone.
• Insight is a definite proposition, not hedging or a category broadened to include obvious cases.
• Grounding points to specific prior evidence. Do not fabricate sources.

═══════════════════════════════════════
DIVERSITY BAR (across the {num_ideas} ideas)
═══════════════════════════════════════

The {num_ideas} ideas together should span:
• Multiple distinct problems (not variants of the same question).
• A mix of grounding types — ideally hitting at least 4 of the 6 types
  (in_domain_extension, cross_domain_transfer, cross_document_synthesis,
  contradiction, scope_extension, reframing). Do not produce ideas all of
  the same type.
• Different insights — even when two ideas share a problem, they must offer
  materially different lenses on it.

Avoid: paraphrasing the source documents, restating consensus findings as
"insights," and ideas whose insight collapses into the motivation or the
problem.

═══════════════════════════════════════
ORDERING
═══════════════════════════════════════

Put your strongest single idea first — most impactful, best-grounded, most defensible. Each subsequent idea should be as distinct as possible from the ones already listed.

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════

Each idea object must have exactly this structure:

{
  "motivation": "...",
  "problem": "...",
  "insight": {
    "claim": "...",
    "grounding": [
      {"source": "<doc id or citation>",
       "supports": "<what this source establishes that the claim builds on>"}
    ],
    "grounding_type": "in_domain_extension" | "cross_domain_transfer" | "cross_document_synthesis" | "contradiction" | "scope_extension" | "reframing"
  }
}

═══════════════════════════════════════
SOURCE DOCUMENTS
═══════════════════════════════════════

{documents}

═══════════════════════════════════════
TASK
═══════════════════════════════════════

Generate {num_ideas} research ideas conforming to the schema, quality bar,
diversity bar, and ordering rule above. Think before writing — sketch the
space of possible problems and insights internally, then commit to {num_ideas}
that together cover that space well.

Structure your response in two parts:

1. <think> ... </think>
   Use this block to sketch the space of possible problems and insights
   before committing. Inside, work through:
   • What are the main threads or open questions running through the
     source documents?
   • What candidate problems could be posed at the field level? Which are
     broad enough to admit multiple insights?
   • For each promising problem, what insights — drawn from the sources
     or from related domains — could give it a distinctive angle? Note
     the grounding type of each.
   • Which {num_ideas} (problem, insight) pairs together cover the space
     well, hit the diversity bar, and meet the quality bar?
   This block is for your reasoning only and will not be parsed as output.

2. <answer> ... </answer>
   Place the final JSON list of {num_ideas} idea objects inside this block,
   and nothing else. The content of <answer> must be a valid JSON list
   parseable on its own — no prose, no markdown code fences, no commentary
   before or after the list, no enclosing object around the list.

Do not produce any text outside of these two blocks.
"""

# --------------------------------------------------------------------------- #
# K=1 (single-idea) variant of the v0 prompt.
#
# Used by the DARLING baseline (data_prep/darling_k1), where each rollout emits
# exactly ONE idea and DIVERSITY is enforced ACROSS rollouts by the reward, NOT
# within a single response. So this template drops the within-response DIVERSITY
# BAR and the multi-idea ORDERING rule (both meaningless / contradictory for one
# idea) and switches all counts to the singular. Everything else -- SCHEMA,
# per-idea QUALITY BAR, the <think>/<answer> two-block structure, and the JSON
# *list* output (a one-element list, so the parser + reward are unchanged) -- is
# identical to IDEATION_V0_TEMPLATE. Answer format stays "v0".
# NOTE: no {num_ideas} placeholder (this template is single-idea by construction);
# {documents} is still substituted.
IDEATION_V0_K1_TEMPLATE = """You are a research ideation assistant. Given a set of source documents, your job is to generate a single research idea that could inspire a scientist working in or adjacent to the area covered by these documents.

The idea must follow the schema below. Read the definitions carefully — the three fields play different roles, and an idea that collapses them into restatements of each other is not useful.

═══════════════════════════════════════
SCHEMA
═══════════════════════════════════════

1. MOTIVATION
The outward-facing stakes of the work — the case for why anyone beyond the
authors should care if progress is made. Situate the work in a broader
context: who is affected, what becomes possible downstream, what current
limitation it would lift, or what cost it would reduce. Name a concrete
beneficiary (a community, a discipline, a class of applications) and a
concrete consequence (capability unlocked, harm reduced, understanding
gained), rather than gesturing at general importance. Answer: "if this
worked, what would change in the world, and for whom?"

2. PROBLEM
The field-level research question — the broad challenge being posed, of the
kind many researchers in the area might independently arrive at. Name the
object of study and the difficulty or unknown about it, but do NOT commit to
a specific angle, mechanism, or constraint on how it should be approached. A
well-posed problem is intelligible to anyone in the field, concrete enough
that two researchers would agree on what counts as progress, and broad
enough that multiple distinct research directions could attack it. It is
the *what*, distinct from both the *why* (motivation) and the *how*
(methodology). State it without committing to any particular insight or
approach.

3. INSIGHT
A definite proposition positioned against the cited documents. The `claim`
text should make three things visible in its prose:

  (1) WHAT THE DOCS DO — the prior position your move stands against,
      drawn from the cited documents.
  (2) THE CLAIM — your proposition (asymmetry, mechanism, missed
      assumption, alternative framing). Not hedging.
  (3) THE BRIDGE — why what the docs do implies, fails to address, or
      can be reinterpreted to yield the claim. This is the move that
      makes the claim follow from the docs rather than appear from
      nowhere. A "therefore" with no real reasoning step is not a bridge.

A reader should be able to answer "what's the claim?", "is this in the
docs already?", and "how did you get from the docs to here?" from the
`claim` text alone.

The insight is NOT a hypothesis (prediction awaiting test), NOT a method
(plan of action), NOT a motivation (stakes), and NOT a category broadened
to include obvious cases ("X is also caused by Y").

Each insight must additionally include:
  • GROUNDING: one or more specific references to source material
    (document IDs, papers, or established findings) that the claim builds
    on. Quote or paraphrase the supporting finding briefly.
  • GROUNDING_TYPE: one of
      - "in_domain_extension"      — builds directly on findings in the
        problem's home domain (does X+1 where the docs do X)
      - "cross_domain_transfer"    — imports a mechanism established in a
        different domain and applies it to this problem
      - "cross_document_synthesis" — combines claims across multiple
        sources that weren't previously connected; the insight requires
        the combination to be supported
      - "contradiction"            — resolves or exploits a conflict among
        the cited docs; names the disagreement and what it surfaces
      - "scope_extension"          — relaxes a specific assumption the
        cited docs depend on, and describes what the relaxation opens up
      - "reframing"                — proposes a materially different
        framing of the phenomenon the cited docs study, and describes
        what the new framing makes visible

═══════════════════════════════════════
QUALITY BAR
═══════════════════════════════════════

• Problem is broad enough that the insight is doing real narrowing work. If the problem already contains the insight's commitments, broaden the problem.
• Insight `claim` text makes three things visible: what the cited docs do, the new claim, and the bridge from the former to the latter. A reader should answer "what's the claim?", "is this already in the docs?", and "how did you get here?" from the text alone.
• Insight is a definite proposition, not hedging or a category broadened to include obvious cases.
• Grounding points to specific prior evidence. Do not fabricate sources.

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════

The idea object must have exactly this structure:

{
  "motivation": "...",
  "problem": "...",
  "insight": {
    "claim": "...",
    "grounding": [
      {"source": "<doc id or citation>",
       "supports": "<what this source establishes that the claim builds on>"}
    ],
    "grounding_type": "in_domain_extension" | "cross_domain_transfer" | "cross_document_synthesis" | "contradiction" | "scope_extension" | "reframing"
  }
}

═══════════════════════════════════════
SOURCE DOCUMENTS
═══════════════════════════════════════

{documents}

═══════════════════════════════════════
TASK
═══════════════════════════════════════

Generate ONE research idea conforming to the schema and quality bar above.
Think before writing — sketch the space of possible problems and insights
internally, then commit to the single strongest, best-grounded idea.

Structure your response in two parts:

1. <think> ... </think>
   Use this block to sketch the space of possible problems and insights
   before committing. Inside, work through:
   • What are the main threads or open questions running through the
     source documents?
   • What candidate problems could be posed at the field level? Which are
     broad enough to admit multiple insights?
   • For each promising problem, what insight — drawn from the sources
     or from related domains — could give it a distinctive angle? Note
     the grounding type.
   • Which single (problem, insight) pair is the strongest, best-grounded,
     and most defensible, and best meets the quality bar?
   This block is for your reasoning only and will not be parsed as output.

2. <answer> ... </answer>
   Place the final JSON list containing exactly one idea object inside this
   block, and nothing else. The content of <answer> must be a valid JSON list
   parseable on its own — no prose, no markdown code fences, no commentary
   before or after the list, no enclosing object around the list.

Do not produce any text outside of these two blocks.
"""

# --------------------------------------------------------------------------- #
# SI-format ideation prompt.
#
# Same task shape as v0 (K distinct ideas, <think>/<answer>, JSON *list*), but
# each idea is an SI card [problem_gap, core_claim, method, inspiration] with the
# leading [taxonomy] labels -- identical field vocabulary + taxonomy to the gold
# data generator SI_IDEA_GROUNDED_TEMPLATE (prompts/summarization.py), so the
# model is asked to PRODUCE ideas in exactly the format the gold cards are in.
#
# Two adaptations from the gold prompt: (1) it is GENERATIVE and multi-idea --
# the model invents `num_ideas` novel ideas rather than recovering one target
# paper's idea; (2) `inspiration` argues how each NEW idea arises from the
# numbered SI source cards it was shown (there is no target paper at inference).
# The `{documents}` block is the numbered SI source cards (input_si_summaries_str);
# `inspiration`'s "Document N" references point at that block.
# NOTE: plain string + .replace (literal { } braces in the JSON schema).
IDEATION_SI_TEMPLATE = """You are a research ideation assistant. Given a set of source documents, your job is to generate {num_ideas} distinct research ideas that could inspire a scientist working in or adjacent to the area covered by these documents.

Each idea is a single card in the schema below. The four fields play DIFFERENT roles; a card whose fields collapse into restatements of each other is not useful. Two of the fields must begin with a bracketed [label] chosen from the taxonomy that follows the schema.

═══════════════════════════════════════
SCHEMA (one card per idea)
═══════════════════════════════════════

  problem_gap:  The gap the idea addresses, drawn from across the source
                documents -- an open question, tension, or unresolved difficulty
                that connects them and that none resolves on its own. State it as
                one clear gap, NOT as your specific solution to it. BEGIN this
                field with the single best-fitting OPPORTUNITY PATTERN label in
                square brackets (from the taxonomy below), then the prose gap
                (1-2 sentences). Pick the ONE dominant pattern even if several
                apply; if NONE genuinely fits, use "[Other]" rather than forcing a
                poor match. Example:
                "[Bridge Opportunity] Work on X and on Y each ... but neither
                connects ...".
  core_claim:   The couple of sentences the whole idea lives or dies on -- "we
                claim that X." It must be something that could turn out to be
                wrong: a definite, testable assertion / hypothesis / insight, not
                a topic or a direction. Do not just restate the gap, and do not
                describe what you would do about it. From these sentences alone a
                reader should know exactly what the idea is betting on
                (1-2 sentences).
  method:       At a high level, HOW the claim would be turned into research --
                the *kind* of move (combining ideas, extending to a new setting,
                making a fragile result robust, deriving formally, mapping a space
                empirically, building an artifact) and how it addresses the gap.
                It should be a PROPOSED approach, forward-looking: use present/infinitive framing ("to test this, X"; "the approach is to X"). Keep it at the level of an approach, NOT a step-by-step protocol. BEGIN this field with the
                single best-fitting METHOD PARADIGM label in square brackets (from
                the taxonomy below), then the prose approach (2-3 sentences). Pick
                the ONE dominant paradigm even if several apply; if NONE genuinely
                fits, use "[Other]". Example:
                "[Synthesis / Unification] To test this, the approach is to ...".
  inspiration:  Prose (three to five sentences) that ARGUES how the idea arises
                FROM the source documents -- the relationship BETWEEN them, not a
                per-document restatement. This field must be CRITICAL, not a
                credit ledger. Two things it must do:
                (1) NAME EACH CITED SOURCE'S LIMITATION -- for every Document you
                    lean on, say what it got wrong, missed, stopped short of, or
                    left unresolved, so the idea reads as a RESPONSE to a concrete
                    shortcoming. ("Document 3 measured X but only under assumption
                    A"; "Document 5 proposed Y yet never addressed Z".)
                (2) JUSTIFY THE NON-OBVIOUS LEAP -- argue WHY connecting these
                    documents is non-trivial: the tension, friction, or mismatch
                    that had to be resolved for the pieces to fit, so the idea does
                    not read as the obvious next step. If the connection IS
                    routine, say so plainly rather than inflating it.
                Refer to the numbered documents by their exact "Document <N>"
                labels wherever the idea genuinely builds on them. If you draw on
                any idea or analogy from OUTSIDE the provided documents, name that
                external source explicitly (it carries no Document <N>). Do not
                invent a document link.

Write `problem_gap`, `core_claim`, and `method` so they: state the gap, the claim/bet, and the approach, NOT the outcome. Keep `problem_gap` broad enough that the `core_claim` is doing real narrowing work.

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
and do not invent any label outside this list plus "[Other]". Use the bracketed
label ONLY at the very start of the field; do not repeat it in the prose or add
labels to `core_claim` or `inspiration`.

═══════════════════════════════════════
DIVERSITY BAR (across the {num_ideas} ideas)
═══════════════════════════════════════

The {num_ideas} ideas together should span:
• Multiple distinct problem_gaps (not variants of the same question).
• A mix of opportunity patterns and method paradigms -- do not produce ideas that
  all carry the same two [labels].
• Different core_claims -- even when two ideas share a gap, they must offer
  materially different bets on it.

Avoid: paraphrasing the source documents, restating consensus findings as
"claims," and cards whose core_claim collapses into the problem_gap.

═══════════════════════════════════════
ORDERING
═══════════════════════════════════════

Put your strongest single idea first -- most impactful, best-grounded, most
defensible. Each subsequent idea should be as distinct as possible from the ones
already listed, but equally impactful (if possible).

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════

Each idea object must have exactly this structure:

{
  "problem_gap": "[<one opportunity pattern>] ...",
  "core_claim": "...",
  "method": "[<one method paradigm>] ...",
  "inspiration": "... Document <N> did X but stopped short of ...; Document <M> offered the method yet ...; connecting them is non-obvious because ... (name any external source explicitly) ..."
}

═══════════════════════════════════════
SOURCE DOCUMENTS
═══════════════════════════════════════

{documents}

═══════════════════════════════════════
TASK
═══════════════════════════════════════

Generate {num_ideas} research ideas conforming to the schema, taxonomy,
diversity bar, and ordering rule above. Think before writing -- sketch the space
of possible gaps and claims internally, then reason through them, eliminating ideas that either have already been implemented, or do not make any logically sense. Then commit to {num_ideas} that together cover that space well.

Structure your response in two parts:

1. <think> ... </think>
   Use this block to sketch the space of possible gaps and claims before
   committing. Inside, work through:
   • What are the main threads or open questions running through the documents?
   • What candidate problem_gaps could be posed? Which are broad enough to admit
     multiple distinct core_claims?
   • For each promising gap, what core_claim -- drawn from the sources or a
     related domain -- gives it a distinctive angle, and what method paradigm
     would carry it? Note the two [labels].
   • Which {num_ideas} cards together cover the space well and hit the diversity
     bar?
   This block is for your reasoning only and will not be parsed as output.

2. <answer> ... </answer>
   Place the final JSON list of {num_ideas} idea objects inside this block,
   and nothing else. The content of <answer> must be a valid JSON list parseable
   on its own -- no prose, no markdown code fences, no commentary before or after
   the list, no enclosing object around the list.

Do not produce any text outside of these two blocks.
"""

# --------------------------------------------------------------------------- #
# K=1 (single-idea) variant of the SI prompt.
#
# The SI counterpart of IDEATION_V0_K1_TEMPLATE, and the prompt the SI DARLING
# baseline trains on. Same reason for existing: DARLING enforces diversity ACROSS
# rollouts via the reward's union-find partition, so a within-response diversity
# instruction is not just redundant, it is CONTRADICTORY -- rendering the K=3 SI
# template at num_ideas=1 produces "DIVERSITY BAR (across the 1 ideas)", "Multiple
# distinct problem_gaps", "even when two ideas share a gap", and "Which 1 cards
# together cover the space well". So this drops the DIVERSITY BAR and the ORDERING
# section (both multi-idea by construction) and switches every count to singular.
#
# Everything the reward touches is UNCHANGED from IDEATION_SI_TEMPLATE: the SCHEMA
# and its field definitions verbatim, the full TAXONOMY of 7+7 [labels], the
# <think>/<answer> two-block structure, and the JSON *list* output -- a ONE-element
# list, so parse_ideas_si, the format gate (len(ideas) == K with K=1), and the
# distance / impact / grounding judges all see exactly the shape they see at K=3.
# Answer format stays "si".
# NOTE: no {num_ideas} placeholder (single-idea by construction); {documents} is
# still substituted. Plain string + .replace (literal { } braces in the schema).
IDEATION_SI_K1_TEMPLATE = """You are a research ideation assistant. Given a set of source documents, your job is to generate a single research idea that could inspire a scientist working in or adjacent to the area covered by these documents.

The idea is a single card in the schema below. The four fields play DIFFERENT roles; a card whose fields collapse into restatements of each other is not useful. Two of the fields must begin with a bracketed [label] chosen from the taxonomy that follows the schema.

═══════════════════════════════════════
SCHEMA (one card)
═══════════════════════════════════════

  problem_gap:  The gap the idea addresses, drawn from across the source
                documents -- an open question, tension, or unresolved difficulty
                that connects them and that none resolves on its own. State it as
                one clear gap, NOT as your specific solution to it. BEGIN this
                field with the single best-fitting OPPORTUNITY PATTERN label in
                square brackets (from the taxonomy below), then the prose gap
                (1-2 sentences). Pick the ONE dominant pattern even if several
                apply; if NONE genuinely fits, use "[Other]" rather than forcing a
                poor match. Example:
                "[Bridge Opportunity] Work on X and on Y each ... but neither
                connects ...".
  core_claim:   The couple of sentences the whole idea lives or dies on -- "we
                claim that X." It must be something that could turn out to be
                wrong: a definite, testable assertion / hypothesis / insight, not
                a topic or a direction. Do not just restate the gap, and do not
                describe what you would do about it. From these sentences alone a
                reader should know exactly what the idea is betting on
                (1-2 sentences).
  method:       At a high level, HOW the claim would be turned into research --
                the *kind* of move (combining ideas, extending to a new setting,
                making a fragile result robust, deriving formally, mapping a space
                empirically, building an artifact) and how it addresses the gap.
                It should be a PROPOSED approach, forward-looking: use present/infinitive framing ("to test this, X"; "the approach is to X"). Keep it at the level of an approach, NOT a step-by-step protocol. BEGIN this field with the
                single best-fitting METHOD PARADIGM label in square brackets (from
                the taxonomy below), then the prose approach (2-3 sentences). Pick
                the ONE dominant paradigm even if several apply; if NONE genuinely
                fits, use "[Other]". Example:
                "[Synthesis / Unification] To test this, the approach is to ...".
  inspiration:  Prose (three to five sentences) that ARGUES how the idea arises
                FROM the source documents -- the relationship BETWEEN them, not a
                per-document restatement. This field must be CRITICAL, not a
                credit ledger. Two things it must do:
                (1) NAME EACH CITED SOURCE'S LIMITATION -- for every Document you
                    lean on, say what it got wrong, missed, stopped short of, or
                    left unresolved, so the idea reads as a RESPONSE to a concrete
                    shortcoming. ("Document 3 measured X but only under assumption
                    A"; "Document 5 proposed Y yet never addressed Z".)
                (2) JUSTIFY THE NON-OBVIOUS LEAP -- argue WHY connecting these
                    documents is non-trivial: the tension, friction, or mismatch
                    that had to be resolved for the pieces to fit, so the idea does
                    not read as the obvious next step. If the connection IS
                    routine, say so plainly rather than inflating it.
                Refer to the numbered documents by their exact "Document <N>"
                labels wherever the idea genuinely builds on them. If you draw on
                any idea or analogy from OUTSIDE the provided documents, name that
                external source explicitly (it carries no Document <N>). Do not
                invent a document link.

Write `problem_gap`, `core_claim`, and `method` so they: state the gap, the claim/bet, and the approach, NOT the outcome. Keep `problem_gap` broad enough that the `core_claim` is doing real narrowing work.

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
and do not invent any label outside this list plus "[Other]". Use the bracketed
label ONLY at the very start of the field; do not repeat it in the prose or add
labels to `core_claim` or `inspiration`.

═══════════════════════════════════════
QUALITY BAR
═══════════════════════════════════════

Avoid: paraphrasing the source documents, restating consensus findings as
"claims," and a card whose core_claim collapses into the problem_gap.

Commit to your strongest idea -- most impactful, best-grounded, most defensible.

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════

The idea object must have exactly this structure:

{
  "problem_gap": "[<one opportunity pattern>] ...",
  "core_claim": "...",
  "method": "[<one method paradigm>] ...",
  "inspiration": "... Document <N> did X but stopped short of ...; Document <M> offered the method yet ...; connecting them is non-obvious because ... (name any external source explicitly) ..."
}

═══════════════════════════════════════
SOURCE DOCUMENTS
═══════════════════════════════════════

{documents}

═══════════════════════════════════════
TASK
═══════════════════════════════════════

Generate ONE research idea conforming to the schema, taxonomy, and quality bar
above. Think before writing -- sketch the space of possible gaps and claims
internally, then reason through them, eliminating ideas that either have already
been implemented, or do not make any logical sense. Then commit to the single
strongest one.

Structure your response in two parts:

1. <think> ... </think>
   Use this block to sketch the space of possible gaps and claims before
   committing. Inside, work through:
   • What are the main threads or open questions running through the documents?
   • What candidate problem_gaps could be posed? Which are broad enough to admit
     a distinctive core_claim?
   • For each promising gap, what core_claim -- drawn from the sources or a
     related domain -- gives it a distinctive angle, and what method paradigm
     would carry it? Note the two [labels].
   • Which single card is the strongest, best-grounded, and most defensible?
   This block is for your reasoning only and will not be parsed as output.

2. <answer> ... </answer>
   Place the final JSON list containing exactly one idea object inside this
   block, and nothing else. The content of <answer> must be a valid JSON list
   parseable on its own -- no prose, no markdown code fences, no commentary
   before or after the list, no enclosing object around the list.

Do not produce any text outside of these two blocks.
"""

# Default number of ideas per generation (K in the reward doc).
DEFAULT_NUM_IDEAS = 8

# Top-level fields of each idea object the model emits. `insight` is a nested
# object; parsing.py additionally flattens it into insight_claim / grounding /
# grounding_type for display.
ANSWER_FIELDS = ("motivation", "problem", "insight")
# SI-format top-level fields (the flat 4-field SI card). Used by parsing.py to
# recognize + normalize SI-format answers.
SI_ANSWER_FIELDS = ("problem_gap", "core_claim", "method", "inspiration")

# Registry of prompt types -> template. Add new templates here and they become
# selectable via run_inference.py --prompt-type.
PROMPT_TEMPLATES = {
    "ideation_v0": IDEATION_V0_TEMPLATE,
    "ideation_v0_k1": IDEATION_V0_K1_TEMPLATE,
    "ideation_si": IDEATION_SI_TEMPLATE,
    "ideation_si_k1": IDEATION_SI_K1_TEMPLATE,
}
PROMPT_TYPES = tuple(PROMPT_TEMPLATES)
DEFAULT_PROMPT_TYPE = "ideation_v0"

# Per-prompt-type answer format -- tells the parser which schema to normalize to
# and the dataloader which source-summary field to feed as the Document N block.
# "v0" -> {motivation, problem, insight}, input_full_summaries_str.
# "si"  -> {problem_gap, core_claim, method, inspiration}, input_si_summaries_str.
PROMPT_ANSWER_FORMAT = {
    "ideation_v0": "v0",
    "ideation_v0_k1": "v0",
    "ideation_si": "si",
    "ideation_si_k1": "si",
}


def format_documents(input_fields: List[str]) -> str:
    """Join per-document strings into the numbered block the prompt expects.

    `input_fields` is a list of already-assembled "Title\n\nSummary" strings
    (see dataloader.build_input_fields)."""
    return "\n\n".join(
        f"Document {i + 1}:\n\n{doc}" for i, doc in enumerate(input_fields)
    )


def build_prompt(
    input_fields: List[str],
    prompt_type: str = DEFAULT_PROMPT_TYPE,
    num_ideas: int = DEFAULT_NUM_IDEAS,
) -> str:
    """Build the full user prompt from a record's input document strings.

    `num_ideas` is substituted into the template wherever the idea count appears
    (schema/diversity/task instructions)."""
    try:
        template = PROMPT_TEMPLATES[prompt_type]
    except KeyError:
        raise ValueError(
            f"Unknown prompt_type {prompt_type!r}; choices: {', '.join(PROMPT_TYPES)}"
        )
    documents = format_documents(input_fields)
    # Order matters: substitute {num_ideas} before {documents} so a document
    # body that happens to contain the literal "{num_ideas}" is left untouched.
    return (
        template.replace("{num_ideas}", str(num_ideas))
        .replace("{documents}", documents)
    )
