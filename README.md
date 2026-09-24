# oscillator-research

![Sound enters a lattice of coupled oscillators; the field's collective state is read out as a digit.](resources/field.png)

> Scientific research into coupled oscillators as a computational substrate for speech recognition and generation.

## What this is about

A coupled-oscillator network is a population of rhythmic units that pull each other toward phase agreement. Wire them onto a geometric structure with a coupling function and drive them with sound, and the field's collective state becomes a representation of that sound. The physics does the transducing, and what you read out is the network's own response to the input.

The idea is quite old. It runs from Huygens noticing in 1665 that two pendulum clocks on one beam fall into step, through the Hopf model of the cochlea, reservoir computing, and Mead's case for letting physics do the computing directly. What is new is that three fields have arrived at the same dynamics from different directions. Physics now describes synchronization regimes that exhibit computational properties rather than mere order. Neuroscience has identified the neuron as a complex chemical and physical system with functional oscillatory properties. Machine learning has recently produced oscillator architectures that are effective at classification, image generation and path navigation, with evidence suggesting that oscillatory neural networks can be trained, learn, remember and reason.

That convergence is the motivation here. The working hypothesis is that a substrate whose native operations are resonance and entrainment more closely resembles the biological neurons that evolution refined to sense and learn from signals in the physical world, and that a trainable model built from those dynamics would make questions about learning, forgetting and rhythm disruption addressable in simulation.

Speech is the natural place to explore this idea, because speech is oscillation at every scale: prosody near 1 Hz, syllable rhythm at 4-8 Hz, phone transitions at 10-40 Hz, pitch and formants from 100 Hz to several kHz. An oscillator field is a frequency-selective medium with intrinsic timescales, locking behaviour and spatial wave modes, which is the representational vocabulary that structure would seem to want.

Whether any of that survives contact with controlled measurement is the open question, and it is the question this research addresses. The work is deliberately small-scale and control-heavy: single-variable comparisons, a control that reads the input alone under every accuracy, parameter-matched conventional baselines, and comparisons paired over matched configurations, test clips and seeds. Negative results are reported with the same weight as positive ones.

Every experimental paper ships with the code and the raw per-run data that produced its numbers, so any claim here can be peer reviewed and validated or refuted.

## Papers

| # | Paper | What it does |
|---|---|---|
| 01 | [From Synchronization Physics to Trained Dynamics: A Survey of Oscillator Networks in Machine Learning](papers/01-evidence-audit/from-synchronization-physics-to-trained-dynamics.md) | A critical survey of oscillator networks in machine learning, tracing the idea from Huygens in 1665 to the current revival and drawing on 130 sources across physics, mathematics, neuroscience and neuromorphic computing. It sorts eighteen published systems by what is actually learned, and finds that of the sixty-five control comparisons that would isolate the physics, sixteen have been run, ten of them by three systems. |
| 02 | [Spoken-Digit Recognition Without Training: Geometry, Coupling, and Drive Effects in Oscillator Networks](papers/02-untrained-reservoirs/) | Reads an untrained network of 1,024 coupled oscillators with a linear readout on spoken digits from held-out speakers in noise (AudioMNIST), against the readout on its input alone, the same network uncoupled, non-oscillating leaky-integrator banks and five trained networks of the same ~2k parameters, over 6,275 runs that vary the coupling function, lattice geometry (a cochlea among them), natural frequencies, input pathway, input gain, readout width and training data. Most of the accuracy comes from the input and the readout: the Kuramoto network reads within a point of its own input in noise, and carries the order of events by integrating its input, as a leaky-integrator bank does. A free oscillator amplitude (Stuart–Landau) is the one design choice that lifts it above its input, into the trained networks' range; geometry moves it by less than a point, and the trained networks still lead. |

## Layout

```
papers/
├── 01-evidence-audit/
│   ├── README.md
│   ├── from-synchronization-physics-to-trained-dynamics.md   the manuscript
│   ├── ….html .epub                                          built beside it
│   ├── ….docx  …-title-page.docx  …-highlights.docx         the Word submission files
│   ├── …-tmlr.pdf  …-neunet.pdf  …-arxiv.tar.gz              submission builds, one per venue
│   ├── …-preprint.pdf  …-preprint.tex                       the same paper with the author named
│   ├── metadata/    front matter, and how to cite this paper
│   └── references/  the works it cites
└── 02-untrained-reservoirs/
    ├── … the same, plus
    ├── src/         harness, experiment driver, tests
    ├── results/     the record: 6,275 runs, one file per experiment, and what each holds
    └── resources/   figures and audio

publishing/          the scripts that turn a manuscript into those formats
```

Each paper is self-contained: the manuscript, the formats built from it, the bibliography it cites, and its own citation metadata all live in one folder.

## Publishing

The manuscripts are plain Markdown. `publishing/` turns them into every other format without touching the source, writing each one beside the Markdown it came from. Ultimately producing a styled HTML page, EPUB, DOCX, a TMLR submission build against the journal's own style file, and a ready-to-upload arXiv bundle:

```bash
bash publishing/publish.sh
```

Citations are resolved against a real bibliography whose author lists, titles, venues and years come from the DOI registries and DBLP rather than from prose. Every build runs a set of checks over the sources and fails loudly rather than shipping something wrong: whether any entry is too incomplete to publish, whether a citation label is claimed by two different works, whether every system and
author is cited where it is first named, whether every "Section 2.3.3" and "Appendix D" pointer resolves, whether the section numbers in the headings are still current, and whether any invisible character has found its way into the source. Section numbers are written into the Markdown itself, so a cross-reference read on GitHub points at a heading you can see.

To cite one of these papers, see the citation section in its README ([Paper 01](papers/01-evidence-audit/README.md#citing-this-paper), [Paper 02](papers/02-untrained-reservoirs/README.md#citing-this-paper)) — or [CITATION.cff](CITATION.cff) for the repository as a whole. More details in [publishing/README.md](publishing/README.md).

## Data

Speech corpora are not committed. Paper 02 uses [AudioMNIST](https://github.com/soerenab/AudioMNIST); download it and build the
local bank as described in that paper's README.

## License

[Apache 2.0](LICENSE).
