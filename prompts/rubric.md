# Evaluation Rubric for Chiron Prompt Variants

Use this rubric when judging eval run results (`evals/eval_*.json`).

## Per-Turn Scores

| Dimension | Scale | What to evaluate |
|-----------|-------|-----------------|
| MI Adherence | 1-5 | OARS technique use, stage-appropriate response, autonomy support |
| Response Specificity | 1-5 | References person's exact words/situation, avoids platitudes |
| Format Compliance | 1-3 | 2-3 sentences, under 50 words, no lists/jargon |
| Case Note Accuracy | 1-5 | Correct MI stage, What Changed reflects person (not coach), OARS terminology |

### MI Adherence Detail
- **5**: Clear OARS technique, stage-appropriate, supports autonomy
- **4**: Good technique use with minor issues
- **3**: Recognizable MI but technique choice could be better
- **2**: Generic counseling response, not MI-specific
- **1**: Advice-giving, confrontational, or dismissive

### Response Specificity Detail
- **5**: Directly quotes or paraphrases person's words, connects to their situation
- **4**: References situation but could be more specific
- **3**: Somewhat generic but relevant
- **2**: Could apply to almost anyone
- **1**: Completely generic or misses the content

### Format Compliance Detail
- **3**: 2-3 sentences, under 50 words, natural prose
- **2**: Minor violations (slightly over word count, or 4 sentences)
- **1**: Major violations (lists, multiple paragraphs, jargon)

### Case Note Accuracy Detail
- **5**: MI stage correct, What Changed accurately captures person's disclosure, OARS terminology used correctly
- **4**: Minor inaccuracy in one field
- **3**: One field clearly wrong but others correct
- **2**: Multiple fields inaccurate
- **1**: Notes are garbled, echoed exchange, or nonsensical

## Per-Combination Aggregate Scores

| Dimension | Scale | What to evaluate |
|-----------|-------|-----------------|
| Conversation Flow | 1-5 | Natural progression, appropriate questions, no abrupt shifts |
| MI Stage Progression | 1-5 | Sensible stage sequence across turns (e.g., engage→focus→evoke is good) |
| Theme Completeness | 1-3 | Monotonic accumulation, captures all relevant topics |
| Consistency | 1-5 | Same quality across all 5 turns (no degradation) |

### MI Stage Progression Detail
- **5**: Logical progression matching conversation content (e.g., E→E→F→E→V)
- **4**: Mostly logical with one questionable transition
- **3**: Some jumps but generally tracks the conversation
- **2**: Random-seeming stage assignments
- **1**: All same stage or completely backwards

### Theme Completeness Detail
- **3**: All disclosed topics captured, themes only grow
- **2**: Most topics captured, minor omissions
- **1**: Major topics missing or themes regress

## Scoring Template

For each combination, produce:

```
## [coach_variant] + [supervisor_variant]

### Turn-by-turn
| Turn | MI Adherence | Specificity | Format | Case Notes | Notes |
|------|-------------|-------------|--------|------------|-------|
| 1    |             |             |        |            |       |
| 2    |             |             |        |            |       |
| 3    |             |             |        |            |       |
| 4    |             |             |        |            |       |
| 5    |             |             |        |            |       |

### Aggregate
- Conversation Flow: X/5
- MI Stage Progression: X/5
- Theme Completeness: X/3
- Consistency: X/5

### Total: XX/100
(Sum: per-turn max = 5*18 = 90, aggregate max = 18, normalize to 100)
```
