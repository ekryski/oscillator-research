# Features, superposition and the transformer block: a primer

Working notes, cleaned from the first thirteen turns of a conversation with Google AI Mode on 2026-10-17 (the provenance copy is `source-transcript-2026-10-17.md`, lines 9 to 1046). Each section opens with the question as it was asked, then gives the answer rewritten in prose, then a caveat wherever the search engine's reply oversimplified or got something wrong. The references at the end were checked against their publisher or arXiv records on 2026-10-17. The last section says where this thread handed over to the oscillator work in `a-unified-theory-of-intelligence.md`.

## 1. What is a feature?

Asked: "What is a feature in machine learning?"

The word has two meanings that sit at opposite ends of a model, and the conversation moved from the first to the second without saying so. In classical machine learning a feature is an input variable: a house's floor area, a patient's age, the mel-frequency cepstral coefficients of an audio clip, the count of a word in an email. Features in this sense are chosen, cleaned and transformed before training (feature engineering), pruned to the informative subset (feature selection) or computed from raw signals by a fixed procedure (feature extraction). The label is the quantity the model is asked to predict from them.

In interpretability research a feature is something the model has learned rather than something it was given: a direction in the activation space of a layer that corresponds to an interpretable property of the input, such as "this token is inside a quoted string" or "the text is about legal procedure". The two meanings share the idea of a measurable property with predictive value, but the second kind lives inside the network and has to be discovered. Every later section of this primer uses the second meaning.

Caveat: the reply described only the first meaning. The second is a working hypothesis rather than a settled definition; the claim that features are linear directions (the "linear representation hypothesis") has strong evidence in language models but is not a theorem, and some properties appear to be represented nonlinearly or in circuits rather than in single directions.

## 2. ReLU

Asked: "What is ReLU?"

The rectified linear unit is the function ReLU(x) = max(0, x): a negative input becomes zero, a positive input passes unchanged. Despite looking like two straight lines, it is nonlinear, and composing linear maps with it lets a network represent functions that no linear model can. Its practical advantages over the sigmoid and tanh units that preceded it are that it does not saturate for positive inputs, so its gradient there is exactly one and does not shrink as depth grows; that it costs a comparison rather than an exponential; and that it zeroes a fraction of units on every input, which gives sparse activations. Its known failure is the "dying ReLU": a unit whose pre-activation becomes negative for every training input has zero gradient and never recovers. Leaky ReLU gives the negative side a small slope, PReLU learns that slope, and ELU replaces the negative side with a smooth exponential that pushes mean activations toward zero ([Nair & Hinton 2010](https://icml.cc/Conferences/2010/papers/432.pdf); [Glorot, Bordes & Bengio 2011](https://proceedings.mlr.press/v15/glorot11a.html)).

Caveat: "ReLU solves the vanishing gradient problem" is too strong. It removes one cause (saturation), and deep networks still needed careful initialization, normalization layers and residual connections before they trained reliably.

## 3. Superposition, polysemanticity and interpretability

Asked: "When people talk about interpretability and superposition, what is meant by that in relation to features and neurons?"

The hope that started mechanistic interpretability was that each neuron would be monosemantic: one unit, one concept, so a model could be read by listing what each unit responds to. What was found instead, repeatedly, is polysemanticity: a single unit that fires for academic citations, for images of car wheels and for a particular punctuation pattern, with no common theme. Superposition is the proposed explanation ([Elhage et al. 2022](https://transformer-circuits.pub/2022/toy_model/index.html)). A layer with n units has an n-dimensional activation space, but the world has far more than n properties worth representing. If the properties are sparse, in the sense that few of them are present in any one input, the network can assign each property its own direction, pack many more than n such directions into the space by letting them be nearly rather than exactly orthogonal, and rely on the nonlinearity to suppress the small interference that the overlap produces. In the toy models of Elhage et al. the network does exactly this once features are sparse enough, arranging feature directions into regular polytopes, and the degree of superposition rises with sparsity.

Polysemantic neurons are the visible symptom: a unit's axis is not aligned with any one feature direction, so it participates in several. The practical response has been dictionary learning. A sparse autoencoder (SAE) is trained to reconstruct a layer's activations from a much larger set of latent units under a sparsity penalty, so that each latent captures one direction that is active rarely. On a one-layer transformer this recovers thousands of latents that are far more interpretable than the neurons ([Bricken et al. 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html)); the same method applied to larger language models gives similar results ([Cunningham et al. 2023](https://arxiv.org/abs/2309.08600)), and it has been scaled to a production model with millions of latents ([Templeton et al. 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)).

Caveat: superposition is a hypothesis with strong toy-model and dictionary-learning evidence, not an established mechanism for every polysemantic unit. SAE latents depend on the dictionary size and the sparsity penalty (features split or merge as the dictionary grows), the reconstruction leaves an unexplained residual, and whether SAE latents are the model's own units of computation or a convenient basis is an open question. The reply's "SAEs successfully disentangle the superposed directions into pure, monosemantic features" overstates what the papers claim.

## 4. Activation space and feature space

Asked: "What is meant by activation space in this context? What is the other space?"

Activation space is the vector space of a layer's outputs: with n units, a point in R^n whose coordinates are the units' activations on one input. Feature space is the set of interpretable directions the network uses, of which there may be many more than n. Superposition is the map from the second into the first. The hotel analogy from the reply is worth keeping: a hotel with 100 rooms (units) and 500 guests (features) can accommodate everyone only by giving several guests the same room on different schedules, and sparsity is what makes the schedules rarely collide.

| | Activation space | Feature space |
|---|---|---|
| What it is | The n coordinates of a layer's units | The m directions the network uses to represent properties, m > n |
| Natural basis | The units themselves | Interpretable properties ("legal syntax", "the colour magenta") |
| Reading it directly | Dense and polysemantic | Sparse; one direction, one property |

Caveat: the table's "basis vectors" language is loose on the feature side. Feature directions are not a basis of anything (there are more of them than dimensions, so they are linearly dependent). The geometric fact being used is that in n dimensions one can place exponentially many unit vectors with every pairwise cosine below a small ε, which is the same high-dimensional phenomenon the Johnson–Lindenstrauss lemma rests on ([Johnson & Lindenstrauss 1984](https://doi.org/10.1090/conm/026/737400)).

## 5. Hidden layers, hidden space, and why "hidden"

Asked, in two turns: "When we refer to hidden layers, is that the same as hidden space?" and "Why are they called hidden layers when they are physically there?"

A hidden layer is a piece of architecture: a row of units between the input and the output, with its weights in memory and its activations computable on demand. Hidden space, more often called latent space, is the geometry of what those activations represent: which inputs land near which others, and along which directions properties vary. The layer is the floor of the hotel; the space is how the guests are arranged on it. The name "hidden" is old and only means that the layer is not part of the interface: a user feeds the input layer and reads the output layer, and the middle layers are hidden from that transaction, as a kitchen is hidden from the dining room. Nothing about them is inaccessible to anyone with the weights.

Eric's reframing in the conversation is the better description of the real problem: what is hidden is the feature space, because many properties are compressed into each unit. That is why the field talks about latent space and activation space rather than about hidden layers.

Caveat: in the transformer literature the space that carries information between blocks is called the residual stream, and it is the clearer term for what the conversation later means by "the 768 numbers" ([Elhage et al. 2021](https://transformer-circuits.pub/2021/framework/index.html)).

## 6. The transformer block: attention, the MLP, and the four-fold expansion

Asked: "Isn't there a hidden layer projection from linear attention layers? What is a linear attention layer? I remember learning that the hidden layers are larger than attention layers. Is that true?"

A transformer block has two sublayers, each wrapped in a residual connection ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)). The attention sublayer lets each position read from the others: it projects every position's vector into queries, keys and values, scores each query against every key, and sums the values with softmax weights. The projections are d_model by d_model matrices split across heads, so for GPT-2 small (d_model = 768, 12 heads) each head works in 64 dimensions. The MLP sublayer, applied to each position on its own, widens the vector from d_model to 4·d_model (3,072 for GPT-2 small), applies a nonlinearity, and narrows it back. With the usual ratios the MLPs hold about two thirds of a block's parameters, which is the sense in which the "hidden layers are larger than the attention layers".

Linear attention is a different thing from the MLP's hidden layer, and the reply ran the two together. Standard attention costs time and memory that grow with the square of the sequence length, because every query is scored against every key. Linear attention replaces the softmax with a kernel feature map φ so that the scores factorize, and the sum over past positions can be kept as a running matrix S_t = Σ φ(k_i)ᵀ v_i of fixed size. Each new position updates S_t and reads from it with its query, which makes the layer a recurrent network with a matrix-valued state ([Katharopoulos et al. 2020](https://arxiv.org/abs/2006.16236)). That state is the "hidden projection" the question was reaching for. Its descendants include RetNet, gated linear attention and the state-space models (Mamba) that keep a constant-size state by construction.

Caveat: the reply said the cost of standard attention "explodes exponentially"; it grows quadratically with sequence length. And the 4× ratio is a convention from the original transformer, not a law: models with SwiGLU use 8/3 × d_model (Section 11), and some recent models use other ratios.

## 7. What the MLP does: up-projection, gate, down-projection

Asked: "What does it mean that data moves out of attention and into the MLP hidden layer? What is happening with the memory and the matrix math? Why does this need to happen?"

Take one position's vector x in R^768 after attention has added context to it. The MLP computes h = σ(W_up x + b_up), a vector of 3,072 numbers, and then y = W_down h + b_down, back in 768 dimensions, which is added to the residual stream. One useful way to read the two matrices is as a key–value memory ([Geva et al. 2021](https://arxiv.org/abs/2012.14913)): each of the 3,072 columns of W_up is a key, the dot product with x measures how well the current state matches that key, the nonlinearity σ keeps only strong matches, and each matching unit adds its corresponding row of W_down, the value, into the output. The widening gives the layer room for 3,072 such lookups rather than 768, and the classical argument for why lifting to a higher dimension helps is Cover's: a set of points that is not linearly separable in a low dimension is likely to become separable when mapped nonlinearly into a higher one ([Cover 1965](https://doi.org/10.1109/PGEC.1965.264137)).

On memory: the weights are never copied. W_up and W_down sit in accelerator memory once, and activations stream through them in a matrix multiplication. The hardware cost during inference is dominated by reading those resident weights, not by moving them.

Caveat: the reply's story that the up-projection "untangles the superposition" into clean, non-overlapping concept slots is a simplification. The 3,072 hidden units are themselves polysemantic, and the hidden layer is also in superposition, as sparse probing across layers shows ([Gurnee et al. 2023](https://arxiv.org/abs/2305.01610)). The key–value reading is an interpretation that fits many units well, not a mechanism the architecture enforces.

## 8. Is the nonlinearity choosing tokens?

Asked: "Does it copy the weights to multiple locations? What is in W_up? What is in the one-dimensional data vector? Do ReLU, GELU or SwiGLU determine the most likely tokens, and are the top k passed to the down-projection?"

There are no tokens inside the block. The vector is one position's state in the residual stream, a mixture of features accumulated by the embedding, by every earlier block and by this block's attention. W_up is the key bank described above. The nonlinearity is a gate, not a selection: every unit whose match exceeds its threshold passes, in proportion to how far it exceeds it, whether that is two units or five hundred. Sparsity is a consequence, not a rule. Top-k selection does exist in transformers, but elsewhere: mixture-of-experts routing picks the k best experts for each token, and sparse attention variants pick the k most relevant keys.

Caveat: GELU and SwiGLU are not hard gates. They are smooth, pass small negative values, and the "mostly zeros" picture is only approximately true for them.

## 9. Every position, every layer, in parallel

Asked: "So the 1D data vector is one token? In a forward pass we need to do the up/down projection and activation for every token across every hidden layer?"

Yes. The MLP is applied to every position in every block, so a prompt of 100 positions through 32 blocks is 3,200 MLP applications. The reason this is fast is that the MLP treats positions independently, so the 100 vectors stack into a 100 × 768 matrix and one matrix multiplication handles them all; the same holds for every layer's projections. During generation, the KV cache stores past positions' keys and values so attention does not recompute them, but each new position still runs through every block's MLP.

## 10. Do the 768 numbers correspond to 768 features?

Asked: "So does that mean each of the 768 numbers correlates to a feature?"

No. A feature is a direction, a particular combination of all 768 coordinates, and there are many more directions in use than coordinates. The piano analogy from the reply holds: no single key means "sadness", a chord does. There is a further structural point. In the residual stream no coordinate is special, because nothing acts on it coordinate by coordinate, so the "768 numbers" have no privileged basis at all. The MLP hidden layer is different: its nonlinearity acts on each of the 3,072 units separately, which makes the unit basis privileged there and is why single hidden units are more interpretable than single residual-stream coordinates ([Elhage et al. 2021](https://transformer-circuits.pub/2021/framework/index.html)).

Caveat: "more interpretable" is not "monosemantic"; see the caveat to Section 7.

## 11. Activation functions: ReLU, GELU, SwiGLU

Asked: "Are there downsides to ReLU, GELU, SwiGLU? What is the most commonly used activation function now?"

ReLU's downsides are the dying-unit problem and the sharp corner at zero. GELU, x·Φ(x) with Φ the Gaussian cumulative distribution, is smooth and slightly negative for small negative inputs; it costs an error function or its tanh approximation and was adopted by BERT and the GPT-2 and GPT-3 families ([Hendrycks & Gimpel 2016](https://arxiv.org/abs/1606.08415)). SwiGLU is not a pointwise function but a gated layer: h = Swish(x W) ⊙ (x V), a Swish-activated branch multiplying a linear branch, followed by the down-projection ([Shazeer 2020](https://arxiv.org/abs/2002.05202)). It has three matrices where the plain MLP has two, so to keep the parameter count equal the hidden width is reduced to 8/3 × d_model, which is where that number in the reply comes from. PaLM, LLaMA and their descendants use it ([Chowdhery et al. 2022](https://arxiv.org/abs/2204.02311); [Touvron et al. 2023](https://arxiv.org/abs/2302.13971)). The gating is what makes it different in kind: the nonlinearity becomes multiplicative and input-dependent, so one branch can decide how much of the other passes.

Caveat: "the undisputed gold standard" is the reply's phrasing, not the literature's. The measured gains over GELU are modest, Shazeer offered no explanation for them, and GELU remains in wide use.

## 12. Would a Lagrangian be better?

Asked: "Would it be better to use a Lagrangian?"

A Lagrangian is not a forward-pass operation and cannot replace an activation function; the question is where the conversation turned toward energy and dynamics. Lagrangians appear in machine learning in four places. Constrained optimization: a penalty term such as an L1 or L2 regularizer is the Lagrangian form of a constraint on the weights, and the multiplier is the penalty strength. Physics-informed networks: a differential-equation residual is added to the loss so that the fitted function respects the physics ([Raissi, Perdikaris & Karniadakis 2019](https://doi.org/10.1016/j.jcp.2018.10.045)). Lagrangian neural networks: the network learns a Lagrangian L(q, q̇) and the dynamics follow from the Euler–Lagrange equations, which builds conservation laws into the model ([Cranmer et al. 2020](https://arxiv.org/abs/2003.04630)). Energy-based learning: in equilibrium propagation the network's own dynamics minimize an energy, inference is the relaxation to that minimum, and the weight update is a local contrast between a free equilibrium and one nudged toward the target ([Scellier & Bengio 2017](https://doi.org/10.3389/fncom.2017.00024)). The last of these is the one that matters for oscillator networks, and it is an energy function rather than a Lagrangian: the networks are driven and dissipative, so there is no action to make stationary, but there can be a function that decreases along every trajectory.

Caveat: the reply's framing "Lagrangian multipliers create L1 regularization" is acceptable shorthand. The later uses of "Lagrangian" in the oscillator turns (thresholds, feature sinks, localized Lagrangians) all refer to energy functions and attractors, and `a-unified-theory-of-intelligence.md` renames them.

## 13. Where this handed over

The thread of the thirteen turns is this: a feature is a direction; a layer packs more directions than it has dimensions; the nonlinearity suppresses the interference that packing causes; and the MLP widens to separate what the residual stream had to compress. Eric's question in the oscillator turns was whether a unit with more internal structure than a scalar, a phase, an amplitude and possibly further state dimensions, could carry features in its own coordinates so that reading the unit's state reads the feature, and whether the gate could be a bifurcation the unit passes through rather than a fixed function applied to it. Those are the first two claims of the theory notes, and they get a fair-comparison rule there (matched state dimension, never matched unit count) and an experiment each.

## References

Checked against publisher or arXiv records on 2026-10-17. Status: peer-reviewed unless marked.

- [Bricken et al. 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html) Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread. Not peer-reviewed (research post).
- [Chowdhery et al. 2022](https://arxiv.org/abs/2204.02311) PaLM: Scaling Language Modeling with Pathways. arXiv 2204.02311 (2022); Journal of Machine Learning Research 24(240):1–113 (2023).
- [Cover 1965](https://doi.org/10.1109/PGEC.1965.264137) Geometrical and Statistical Properties of Systems of Linear Inequalities with Applications in Pattern Recognition. IEEE Transactions on Electronic Computers EC-14(3):326–334.
- [Cranmer et al. 2020](https://arxiv.org/abs/2003.04630) Lagrangian Neural Networks. arXiv 2003.04630; ICLR 2020 Deep Differential Equations workshop. Workshop paper, not a main-track publication.
- [Cunningham et al. 2023](https://arxiv.org/abs/2309.08600) Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv 2309.08600; published at ICLR 2024 with the author order Huben, Cunningham, Smith, Ewart & Sharkey.
- [Elhage et al. 2021](https://transformer-circuits.pub/2021/framework/index.html) A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread. Not peer-reviewed (research post).
- [Elhage et al. 2022](https://transformer-circuits.pub/2022/toy_model/index.html) Toy Models of Superposition. Transformer Circuits Thread; also arXiv 2209.10652. Not peer-reviewed (research post).
- [Geva et al. 2021](https://doi.org/10.18653/v1/2021.emnlp-main.446) Transformer Feed-Forward Layers Are Key-Value Memories. Proceedings of EMNLP 2021, 5484–5495.
- [Glorot, Bordes & Bengio 2011](https://proceedings.mlr.press/v15/glorot11a.html) Deep Sparse Rectifier Neural Networks. AISTATS 2011.
- [Gurnee et al. 2023](https://arxiv.org/abs/2305.01610) Finding Neurons in a Haystack: Case Studies with Sparse Probing. Transactions on Machine Learning Research, 2023; arXiv 2305.01610.
- [Hendrycks & Gimpel 2016](https://arxiv.org/abs/1606.08415) Gaussian Error Linear Units (GELUs). arXiv 1606.08415. Preprint.
- [Johnson & Lindenstrauss 1984](https://doi.org/10.1090/conm/026/737400) Extensions of Lipschitz mappings into a Hilbert space. Contemporary Mathematics 26:189–206.
- [Katharopoulos et al. 2020](https://arxiv.org/abs/2006.16236) Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020, PMLR 119:5156–5165.
- [Nair & Hinton 2010](https://icml.cc/Conferences/2010/papers/432.pdf) Rectified Linear Units Improve Restricted Boltzmann Machines. ICML 2010.
- [Raissi, Perdikaris & Karniadakis 2019](https://doi.org/10.1016/j.jcp.2018.10.045) Physics-informed neural networks. Journal of Computational Physics 378:686–707.
- [Scellier & Bengio 2017](https://doi.org/10.3389/fncom.2017.00024) Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation. Frontiers in Computational Neuroscience 11:24.
- [Shazeer 2020](https://arxiv.org/abs/2002.05202) GLU Variants Improve Transformer. arXiv 2002.05202. Preprint.
- [Templeton et al. 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Transformer Circuits Thread; also arXiv 2605.29358 (2026). Not peer-reviewed (research post).
- [Touvron et al. 2023](https://arxiv.org/abs/2302.13971) LLaMA: Open and Efficient Foundation Language Models. arXiv 2302.13971. Preprint.
- [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) Attention Is All You Need. Advances in Neural Information Processing Systems 30 (NeurIPS 2017).
