# Title: The Signal Processing Frontier: Beyond Standard Algorithms

## Keywords

signal processing, algorithm design, gpu primitives, novel transformations, mathematical derivation

## TL;DR

Standard algorithms (FFA, FFT) were designed for serial CPUs. We aim to invent fundamentally NEW mathematical operators that are native to massive parallelism.

## Abstract

The history of signal processing is dominated by two giants: the Fast Fourier Transform (FFT) and the Fast Folding Algorithm (FFA). Both were invented in an era of serial computation, optimizing for minimal operation counts ($O(N \log N)$ or $O(N \log \log N)$). However, modern computational substrates (GPUs, TPUs) favor massive parallelism and memory bandwidth over operation count. This creates a disconnect: we are running "serial-native" algorithms on "parallel-native" hardware.

This workshop challenges researchers to **invent** new mathematical transformations for periodic signal detection that are axiomatically designed for massive parallelism. We do not want optimizations of existing algorithms (e.g., "faster GPU FFT"). We want **novel operators** that:

1.  Leverage tensor contractions, atomic scatter/gather, and massive concurrency as first-class mathematical primitives.
2.  Capture properties that standard transforms miss (e.g., simultaneous phase-frequency evolution, non-sinusoidal periodicity, or cyclostationary features).
3.  Are not constrained by the $O(N)$ efficiency mantras of the 1960s if they unlock superior sensitivity or feature extraction capabilities on modern hardware.

Proposals must focus on mathematical novelty and algorithmic derivation. It is acceptable to propose "computationally expensive" methods if they provide a theoretical sensitivity gain, as hardware scales faster than algorithmic innovation.
