# Title: The Signal Processing Frontier: Beyond Standard Algorithms

## Keywords

signal processing, algorithm design, gpu primitives, novel transformations, mathematical derivation

## TL;DR

Standard algorithms (FFA, FFT) were designed for serial CPUs. We aim to invent fundamentally NEW mathematical operators that are native to massive parallelism.

## Abstract

The history of signal processing is dominated by two giants: the Fast Fourier Transform (FFT) and the Fast Folding Algorithm (FFA). Both were invented in an era of serial computation, optimizing for minimal operation counts ($O(N \log N)$ or $O(N \log \log N)$). However, modern computational substrates (GPUs, TPUs) favor massive parallelism and memory bandwidth over operation count. This creates a disconnect: we are running "serial-native" algorithms on "parallel-native" hardware.

This workshop challenges researchers to **invent** new mathematical transformations for periodic signal detection that are axiomatically designed for massive parallelism. We do not want optimizations of existing algorithms (e.g., "faster GPU FFT"). We want **novel operators** that:

1.  **Inherently Coherent**: The method must preserve phase information throughout the transformation to maximize sensitivity, prioritizing coherent summation (e.g., FFA-like) over incoherent detection (e.g., power-summing).
2.  **Parallel-Native**: The formulation should naturally map to massive concurrency and high-bandwidth memory architectures, without being tied to specific implementation primitives (like scatter/gather).
3.  **Unconstrained**: We encourage exploration of the vast theoretical space of parallel operators that utilize the full capability of modern hardware, even if they differ radically from standard serial algorithms or seem computationally expensive.

Proposals must focus on mathematical novelty and algorithmic derivation. We seek a fundamental rethink of how we search for periodicity when "compute is free" but "bandwidth is king."
