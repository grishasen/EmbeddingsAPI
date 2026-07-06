# Rotating the Black Box: What PCA and ICA Actually Do to Embedding Predictors

*A follow-up to [Turning Marketing Copy into Model Predictors](https://medium.com/@gregorysen/turning-marketing-copy-into-model-predictors-a370f3092e4e).*

In the first post I walked through a production recipe for feeding offer text into a propensity model: **embed → reduce → predict → explain**. The *reduce* step got one section and a rule of thumb — use PCA for a compact baseline, use ICA when you want components a human can name — and then quietly did most of the work. A predictor called `lower_dim_embedding_9` ended up near the top of the SHAP ranking, we named it "the auto-loan theme," and moved on.

This post opens that step up. What does a dimensionality transform actually *do* to an embedding? Why does ICA produce nameable components when PCA and the raw dimensions don't? And is post-hoc transformation really better than building interpretable text features from scratch? These questions have concrete, demonstrable answers — no speculation required — and they're worth understanding before you bet a decisioning architecture on the recipe.

As before: all code is illustrative and deliberately small — it shows the principle and is not meant to be run as-is. It's distilled from experiments on a real corpus of ~15k multilingual offer texts (embedded once with a pinned model, `text-embedding-3-large`), with secrets and local paths removed. **All bank names, product names, and offer copy are fictional.**

## 1. The experimental setup: change the basis, nothing else

One design decision makes everything in this post easier to reason about: **hold the dimensionality fixed and change only the basis.**

We request a 64-dimensional embedding directly from the API (the `dimensions` parameter, Matryoshka-style truncation), then fit PCA and ICA with **64 components each**. Every transform below is a full-rank affine map from 64 numbers to 64 numbers:

```python
from sklearn.decomposition import PCA, FastICA

D = 64                                   # embedding dims requested from the API
pca = PCA(n_components=D).fit(matrix)    # matrix: (n_texts, 64) raw embeddings
ica = FastICA(n_components=D, whiten="unit-variance",
              random_state=42, max_iter=1000).fit(matrix)

pca_scores = pca.transform(matrix)       # same shape as matrix
ica_scores = ica.transform(matrix)       # same shape as matrix
```

No information is thrown away, no compression happens — so any difference between the raw dimensions, the PCA components, and the ICA components is purely about **how the same information is laid out across axes**. That separation matters: in the first post, "reduce" bundled two ideas (fewer numbers, and better-behaved numbers) into one step. Here we isolate the second idea. Transformation is not compression; you can rotate first and truncate later, and the two decisions deserve separate justification.

Serving stays exactly as before: a fitted PCA/ICA is an affine map `x → W·(x − mean)`, so it freezes into the `FrozenLinearReducer` from the first post and ships as a single linear layer. Everything in this article is deployable by construction.

## 2. What's inside a raw embedding

Start with what the 64 raw coordinates look like, because the case for transforming them at all rests on two observable defects.

**Defect 1: the dimensions are correlated.** Compute a plain Pearson correlation matrix across your corpus:

```python
import pandas as pd
import plotly.express as px

corr_raw = pd.DataFrame(matrix).corr()
px.imshow(corr_raw, title="Correlation between raw embedding dimensions").show()
```

*[FIGURE 1 — correlation heatmap of raw embedding dimensions: visible off-diagonal structure]*

The heatmap is not diagonal. Neighboring and distant dimensions carry overlapping information — which is exactly what strains Naive Bayes (the first post's §7) and, more subtly, splits credit between features in a gradient-boosted model: when two features encode the same signal, the trees split on either one arbitrarily, and SHAP importance smears across both. You can't fully trust a feature ranking over redundant features.

**Defect 2: the dimensions mean nothing individually.** Apply the "name it by its extreme examples" trick from the first post directly to a raw coordinate:

```python
out = actions.join(pd.DataFrame(matrix).add_prefix("raw_"))
out.sort_values("raw_4").tail(10).TextData
```

Typical result — a mixed bag:

```
=== Raw dimension 4, top examples ===
  EverSpaar Termijndeposito — 3,5 % rente, 12 maanden vast
  Carte AuroraCard Platinum — 1 % de cashback sur tout
  Prêt Auto Lumina — financez votre voiture à 2,9 % TAEG
  Dépôt à terme EverSpaar — 3,5 % d'intérêt, bloqué 12 mois
  ...
```

No single theme. (Amusingly, in our data this particular dimension *loosely* tracked language — the tail was dominated by French copy. That's luck, not design: occasionally a raw axis happens to align with a dominant factor of variation. Most don't.)

Neither defect is a training failure. The embedding model is optimized so that **directions and distances** carry meaning — cosine similarity between whole vectors. Rotate the entire space by any orthogonal matrix and every cosine similarity is unchanged; the training objective can't tell the difference. The individual coordinate axes are an accident of initialization and optimization. So:

> ***The basis of an embedding space is arbitrary for geometry, but decisive for anything that reads coordinates one at a time — axis-aligned tree splits, per-feature SHAP values, per-predictor monitoring, and human interpretation. That's the entire motivation for choosing the basis deliberately.***

## 3. PCA: the rotation that tidies up

PCA re-centers the cloud of vectors and rotates it so the axes are uncorrelated and ordered by variance. Three properties are worth demonstrating on your own data, because together they make PCA the safe deployment default.

**Property 1: it decorrelates — globally, not approximately.**

```python
corr_pca = pd.DataFrame(pca_scores).corr()
px.imshow(corr_pca, title="Correlation between PCA components").show()
```

*[FIGURE 2 — correlation heatmap of PCA components: clean diagonal]*

The heatmap is the identity matrix, to numerical precision. This is the "near-diagonal sanity check" mentioned in the first post's aside, now with the mechanism visible: decorrelation is what PCA *is*, not a side effect. Redundancy across predictors — Defect 1 — is gone, and with it the worst of the SHAP credit-splitting.

**Property 2: it preserves the geometry the embedding was trained for.** An orthogonal rotation leaves inner products and Euclidean distances untouched (centering shifts cosine values slightly). The practical check: compute the top cosine-similarity pairs before and after, and compare the *ranking*:

```python
from sentence_transformers import util

def top_pairs(vectors, texts, k=5):
    scores = util.cos_sim(vectors, vectors)
    pairs = [(float(scores[i][j]), texts[i], texts[j])
             for i in range(len(texts)) for j in range(i + 1, len(texts))]
    return sorted(pairs, reverse=True)[:k]

top_pairs(matrix, texts)       # raw space
top_pairs(pca_scores, texts)   # PCA space
top_pairs(ica_scores, texts)   # ICA space (whitened: scores shift, see below)
```

For PCA the result is close to guaranteed — rotation preserves inner products exactly, and centering perturbs cosines only slightly. The more interesting empirical check is ICA, whose whitening rescales axes and *does* change the scores. In our runs the top of the list still consists of the same pairs in both transformed spaces — including the pairs that make this whole approach valuable, the cross-language ones (stylized):

```
0.94   EverSpaar Termijndeposito — 3,5 % rente, 12 maanden vast
       Dépôt à terme EverSpaar — 3,5 % d'intérêt, bloqué 12 mois

0.81   NordKredit Autokredit — Ihr neues Auto ab 2,9 % eff. Jahreszins
       Prêt Auto Lumina — financez votre voiture à 2,9 % TAEG
```

The same campaign in two languages, and two different brands' car loans, stay neighbors after the transform. The semantic neighborhood structure that powers cold-start inheritance survives intact. K-means clusters computed before and after the transform tell the same story visually:

*[FIGURE 3 — t-SNE of k-means clusters, raw embeddings vs PCA scores: same cluster structure]*

**Property 3 (negative, and underappreciated): the variance spectrum won't tell you how many components to keep.**

```python
exp_var = PCA(n_components=None).fit(matrix).explained_variance_ratio_
# plot exp_var and its cumsum: no elbow
```

*[FIGURE 4 — explained variance per component: a long, flat tail; no obvious elbow]*

On embedding data the curve typically has no clean elbow — variance is spread across many components, and (worse) variance is simply not the same thing as usefulness for *your* prediction task. A low-variance component can carry the promo-mechanic signal your propensity model needs. This is the empirical backing for the first post's checklist item: **choose the component count by downstream lift (sweep 8/16/32/64 against held-out AUC), never by explained variance.**

What PCA does *not* deliver is nameability. Look at the extreme examples of a typical PCA component and you get another mixed bag — cleaner than a raw dimension, but still a blend of themes. The histogram of a PCA component's scores across the corpus hints at why: it tends to look roughly Gaussian — most texts spread smoothly around the middle, no distinguished "extreme club" to read a theme from. Variance-maximizing directions are democratic; they collect a little of everything.

## 4. ICA: the rotation that makes components nameable

FastICA also whitens and rotates — same family of transform, deployable the same way — but it optimizes for a different property: **statistical independence**, which in practice means it seeks directions whose score distributions are as *non-Gaussian* as possible.

That objective sounds abstract until you plot what it does. Compare the distribution of one PCA component with one ICA component across the corpus:

```python
px.histogram(x=pca_scores[:, 9], title="A PCA component").show()
px.histogram(x=ica_scores[:, 9], title="An ICA component").show()
```

*[FIGURE 5 — per-component histograms: PCA roughly bell-shaped; ICA sharply peaked at zero with heavy tails]*

The ICA histogram is the signature: **a spike at zero and heavy tails.** Most offers score near zero on the component — it's simply "not about them" — while a small set scores extremely. A component that is *silent for most texts and loud for a few* is precisely a component you can name: the loud ones share the theme.

> ***This is the mechanism behind the naming trick from the first post. It isn't that ICA "finds meaning" — it's that ICA finds sparse, heavy-tailed directions, and heavy tails are what make the extreme-examples heuristic informative. PCA components can't play this game because their extremes are just the smooth ends of a bell curve.***

The naming utility, upgraded slightly from the first post — ICA components are *signed*, so check both tails; the two ends of one component can carry different (sometimes opposite) themes:

```python
import numpy as np

def get_extreme_items(texts, scores, component_idx, top_k=10, direction="positive"):
    vals = scores[:, component_idx]
    idxs = np.argsort(vals)
    idxs = idxs[-top_k:][::-1] if direction == "positive" else idxs[:top_k]
    return [(texts[i], float(vals[i])) for i in idxs]
```

Typical output on our corpus (fictional, as always) — the auto-loan component from the first post, and a purer surprise, a *language* component:

```
=== ICA component 9 (positive tail) ===
  +4.1   Prêt Auto Lumina — financez votre voiture à 2,9 % TAEG
  +3.9   NordKredit Autokredit — Ihr neues Auto ab 2,9 % eff. Jahreszins
  +3.6   Aurora Auto Loan — drive away today at 3.4 % APR
  # -> "auto-loan / car financing", cross-language

=== ICA component 17 (positive tail) ===
  +3.8   Prêt Auto Lumina — financez votre voiture à 2,9 % TAEG
  +3.7   Dépôt à terme EverSpaar — 3,5 % d'intérêt, bloqué 12 mois
  +3.5   Carte AuroraCard Platinum — 1 % de cashback sur tout
  # -> every top item is French, regardless of product: a language component
```

The language component is a nice confirmation that ICA recovers *real factors of variation* in the corpus, because language genuinely is an independent factor — any product can appear in any language. And it's not a party trick — it's the cleanest way to *see* everything this post has argued so far. Train the same language probe three times, on the raw dimensions, the PCA scores, and the ICA scores, and compare the SHAP summary plots:



*[FIGURE 6 — SHAP importance for the language probe, three panels: raw / PCA / ICA]*

The progression in our run:

- **Raw:** importance smeared across dozens of correlated dimensions — the top feature carries only a small share of the total. This is §2's credit-splitting, photographed.
- **PCA:** importance collapses onto the *first* component, with a visible second-place remainder. No surprise once you know what PCA optimizes: language is the dominant factor of variation in a bilingual corpus, so the top variance axis catches it. PCA does make the *biggest* theme nameable — it just blends everything below it.
- **ICA:** a single component carries essentially all of it. One concept, one component, one predictor — the property Naive Bayes always wanted and embeddings never gave it.

Same information in every panel (full-rank transforms, remember), radically different arrangement. That figure is the article's thesis in one image.

**Closing the loop with the propensity model.** As in the first post, the payoff comes from chaining: train the model on ICA components, rank components by importance, then name the top ones:

```python
importance = clf.get_feature_importance(data=pool, prettified=True)
for dim in importance["Feature Id"].head(5):
    for text, v in get_extreme_items(texts, ica_scores, int(dim)):
        print(f"  {v:+.2f}  {text}")
```

And a small trick that lands surprisingly well with stakeholders: **plot an actual tree.**

```python
clf.plot_tree(tree_idx=0, pool=pool)
```

*[FIGURE 7 — one CatBoost tree over ICA features: first split on the "auto-loan" component]*

When the features are named ICA components, the tree's top splits read as business rules — "if auto-loan-ness is high and the customer's vehicle-finance activity is recent, propensity goes up." The same tree over raw embedding dimensions reads as noise. Nothing about the model changed; only the basis did.

The caveat from the first post carries over unchanged and matters more here: **ICA component order and sign are not deterministic across runs.** Bootstrap-refit, match components across runs by correlation, align signs, and only publish names for components that reproduce. In our experiments the strong, nameable components (language, major product themes) are exactly the ones that do reproduce — but verify on your data before quoting component numbers in a governance document.

## 5. KernelPCA: the detour that rarely pays

For completeness we also ran KernelPCA (RBF kernel) over the same matrix. Two things to know before you're tempted:

- **It breaks the deployment story.** KernelPCA is not an affine map, so there is no frozen-linear-layer trick: scoring a new offer requires kernel evaluations against (a subset of) the training texts, shipped and versioned alongside the model. That's a real serving dependency, not a detail.
- **It has to earn that cost with visibly better structure, and in our exploration it showed no sign of doing so.** Its component spectrum and projections revealed nothing that plain PCA was missing on this corpus — consistent with embeddings already being a heavily nonlinear representation. The raw text→vector map already spent its nonlinearity budget; adding more on top mostly reshuffles.

Keep it in the toolbox for corpora where linear methods demonstrably fail; don't reach for it by default.

## 6. The road not taken: building interpretable embeddings from scratch

The post-hoc rotation story invites an obvious challenge: *if you want interpretable components, why not build interpretable text features directly?* Classical count-based embeddings do exactly that, and taking the challenge seriously — actually running it — is the best way to appreciate what the LLM embedding is contributing.

The recipe is PPMI + SVD, the pre-neural workhorse: build a word co-occurrence matrix over your corpus, reweight by Positive Pointwise Mutual Information, factorize, and average word vectors into sentence vectors:

```python
from sklearn.decomposition import TruncatedSVD

counts = build_cooccurrence(tokenized_corpus, window=6)   # (vocab, vocab)
ppmi   = positive_pmi(counts)                              # max(log p(w,c)/p(w)p(c), 0)
svd    = TruncatedSVD(n_components=64).fit(ppmi)
word_vectors = svd.transform(ppmi)
# sentence vector = mean of its word vectors
```

The interpretability really is free — each component is a weighted bundle of actual vocabulary words, no naming trick required:

```python
for k, comp in enumerate(svd.components_):
    top = vocab[np.argsort(comp)[::-1][:8]]
    print(f"Component {k}: {', '.join(top)}")
```

But look at what the components are actually made of on a real marketing corpus:

```
Component 1: de, la, votre, le, et, sur, avec, pour
Component 2: %, p.a., rente, zins, taux, cashback, bonus, taeg
Component 3: gratis, gratuit, kostenlos, bij, offre, aankoop, ...
```

Mostly function words and promotional boilerplate — the highest-frequency co-occurrence structure in any campaign corpus — with a couple of usable themes (rate language, free-with-purchase mechanics) mixed in. Interpretable, yes; *discriminative*, much less so. In our runs, a ~30-class product-category probe scored roughly seventeen accuracy points lower with PPMI features than with the LLM-embedding features on the same texts (both were training-fit diagnostics, not held-out estimates — see §7 — but the gap is large and consistent). Only trivially separable attributes like language stayed easy.

The deeper limitations are structural, and worth spelling out because they map exactly to the cold-start use case:

- **No transfer.** PPMI knows only your corpus. The LLM embedding arrives knowing that "Autokredit," "car loan," and "prêt auto" are the same concept — knowledge your few thousand offers could never teach.
- **Out-of-vocabulary cold start.** A brand-new brand name is an unseen word: PPMI has literally no vector for it. The LLM embedding places it sensibly from subword structure and context. The exact scenario that motivated this whole project — a new offer inheriting propensity from its semantic neighbors — is the scenario where count-based features have nothing to offer.

> ***So the comparison lands in an instructive place: interpretability-by-construction is real but buys you components made of stopwords and a weaker representation. The LLM-embedding-plus-ICA route gets quality from the large model and interpretability from the basis change — you're not choosing between them.***

## 7. Where the trick stops working (and other honesty)

**Naming needs a domain-coherent corpus.** We also ran the full pipeline — same embedding model, same ICA, same naming trick — on ~176k generic English dictionary words instead of offer texts. The result is a useful negative: component extremes come out as obscure, loosely related word clusters, nothing a human would confidently name; and the nearest-neighbor structure at word level is dominated by morphology (inflections and variant spellings of the same lemma) as much as by meaning. ICA can only align axes with the factors of variation *present in the data*. Offer copy has a few strong, recurring factors (language, product theme, promo mechanics), so components snap to them. A general vocabulary has thousands of weak factors, so components snap to nothing. Expect the naming trick to work in proportion to how thematically concentrated your corpus is.

**The probe numbers in this post are wiring checks, not results.** The probe accuracies mentioned here were computed on training data — fine for "the pipeline works and the signal exists," useless as generalization estimates. If a number is going into a results table, hold out data; if the claim is about cold-start, hold out *entire actions* (first post, §6). Same discipline, still non-negotiable.

**Nothing here demonstrates lift.** Every experiment in this post is about the *representation*: correlation structure, geometry preservation, component distributions, nameability. None of it shows that ICA features beat PCA features (or raw ones) at predicting clicks — full-rank rotations carry identical information, so any downstream difference would come from interaction with the tree learner's axis-aligned inductive bias and from truncation choices, and that's an empirical question for the cold-start ablation, not something to assert from theory. If you need lift claims, run the first post's §10 checklist. This post's claims stop at: *the transform changes how information is arranged, and that arrangement is what makes explanation possible.*

**And a repeat offender:** ICA stability. Yes, again. It's the difference between "component 9 is the auto-loan theme" being a fact about your pipeline or an anecdote about one random seed.

## 8. Takeaways

Three things to remember:

1. **An embedding's basis is arbitrary — exploit that.** Training constrains directions and distances, not axes, so you are free to choose the basis that suits your consumers of coordinates: decorrelated axes for models and monitoring (PCA), sparse heavy-tailed axes for human explanation (ICA). Full-rank rotation loses nothing; rotation and truncation are separate decisions with separate justifications.
2. **Nameability has a mechanism, not a mystique.** ICA components are readable because their score distributions are heavy-tailed: silent for most texts, loud for a themed few. Check both tails, verify stability across resamples, and the "component 9 = auto loans" explanation becomes defensible.
3. **Post-hoc beats by-construction.** Hand-built interpretable features (PPMI+SVD) give you components made of stopwords, no transfer, and no answer for out-of-vocabulary cold start. Rotating an LLM embedding keeps the large model's knowledge *and* recovers interpretability.

And the discipline point, one more time: a change of basis changes what components *mean*, never what the model can *know*. Demonstrations of structure (this post) and demonstrations of lift (the first post's ablation) are different experiments. Keep them in their own lanes and both survive scrutiny.

## Appendix: minimal skeleton

```python
# 0. EMBED (pinned model, fixed dims) -----------------------------------------
matrix = np.vstack(actions["TextData"].map(embed))          # (n_texts, 64)

# 1. FIT BOTH TRANSFORMS, SAME RANK -------------------------------------------
pca = PCA(n_components=64).fit(matrix)
ica = FastICA(n_components=64, whiten="unit-variance",
              random_state=42, max_iter=1000).fit(matrix)
pca_scores, ica_scores = pca.transform(matrix), ica.transform(matrix)

# 2. VERIFY THE STRUCTURAL CLAIMS ON YOUR DATA --------------------------------
#    - corr(matrix) vs corr(pca_scores): off-diagonal -> diagonal        (§2, §3)
#    - top_pairs(matrix) vs top_pairs(pca_scores): same neighbors        (§3)
#    - histogram per component: Gaussian-ish vs heavy-tailed             (§4)

# 3. NAME WHAT MATTERS ---------------------------------------------------------
#    - train model on ica_scores, rank components by importance          (§4)
#    - get_extreme_items(texts, ica_scores, top_component, direction=..) (§4)
#    - bootstrap-refit ICA, keep only names that reproduce               (§7)

# 4. FREEZE AND SHIP -----------------------------------------------------------
reducer = FrozenLinearReducer(ica.components_, ica.mean_)   # first post, §4
```

*Replace `embed`, the corpus, and the model with your own; the checks are the point.*
