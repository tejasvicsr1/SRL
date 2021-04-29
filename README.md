# Semantic Role Labelling For Hindi

## Semantic Role Labelling

Semantic Role Labelling, in natural language processing, is a process that assigns labels to different words in a sentence that indicate their **semantic role** in the sentence.
This helps in finding the meaning of the sentence, and more importantly, the role of a particlar word in creating that meaning of the sentence.
The task essentially boils down to identifying the various arguments associated with the _predicate_ or the main verb of the sentence and assigning them specific roles.

### Example
---

![SRL Example](https://paperswithcode.com/media/thumbnails/task/task-0000000396-b5ac8e48.jpg)

The above example has 3 distinct labels that can be seen - **Agent**, **Theme**, and the **Location**. It also has the predicate labelled. Using these labels we are then able to answer the question _"Who did what to whom where?"_

Some of the more common labels are -
- Agent
- Experiencer
- Theme
- Result
- Location

More labels (not exhaustive) can be found in these [slides](https://paperswithcode.com/media/thumbnails/task/task-0000000396-b5ac8e48.jpg)

### Problem Classification
---

The problem can the be further decomposed into the following -
> **Predicate Detection**: Findint the predicate in a given sentence.

> **Predicate Sense Disambiguation**: Disambiguating the sense of the predicate found.

> **Argument Identification**: Identifying the arguments for the given predicate for the given sense.
 
> **Argument Classification**: Assigning the labels to the arguments found.

### Hindi
---

For Semantic Role Labelling in Hindi, we will be labelling the words into the following roles:

| **Label**      | **Description**     |
| :---        |    :----  |
| ARG0      | Agent, Experiencer, or doer       |
| ARG1   | Patient or Theme        |
| ARG2      | Beneficiary       |
| ARG3   | Instrument        |
| ARG2-ATR      | Attribute or Quality       |
| ARG2-LOC   | Physical Location        |
| ARG2-GOL      | Goal       |
| ARG2-SOU   | Source        |
| ARGM-PRX      | Noun-Verb Construction        |
| ARGM-ADV   | Adverb        |
| ARGM-DIR      | Direction       |
| ARGM-EXT   | Extent or Comparision        |
| ARGM-MNR      | Manner       |
| ARGM-PRP   | Purpose        |
| ARGM-DIS      | Discourse       |
| ARGM-LOC   | Abstract Location        |
| ARGM-MNS      | Means       |
| ARGM-NEG   | Negation        |
| ARGM-TMP      | Time       |
| ARGM-CAU   | Cause or Reason        |

The following labels have been taken from this [paper](https://verbs.colorado.edu/hindiurdu/guidelines_docs/PBAnnotationGuidelines.pdf).
