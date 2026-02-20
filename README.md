# music-source-separation-literature-review
A structured literature review of Music Source Separation research, analyzing 70+ papers across architectures, datasets, and evaluation metrics.

## Overview
This repository contains an ongoing literature mapping study on Music Source Separation (MSS) research. The project aims to build a structured understanding of deep learning approaches for separating musical sources from mixture signals.

So far, more than 70 research papers have been reviewed at the abstract and methodological overview level. A structured comparative framework has been designed to enable deeper cross-paper analysis, with detailed entries currently being expanded.

## Objectives:
- Map the evolution of deep learning architectures in MSS
- Compare models across datasets and evaluation metrics
- Identify performance trends and methodological shifts
- Highlight open research challenges and gaps
- Build a structured reference framework for future research

## Comparison Framework:
Each paper is analyzed using a structured comparison table including:
- Model Architecture (CNN, U-Net, Transformer, Hybrid, etc.)
- Input Representation (Time-domain, Spectrogram, Hybrid)
- Dataset (MUSDB18, WHAM!, custom datasets, etc.)
- Evaluation Metrics (SDR, SI-SDR, PESQ, etc.)
- Supervision Type
- Real-time Capability
- Computational Considerations
- Strengths and Limitations

## Current Progress:
- 70+ papers identified and summarized
- Abstract-level analysis completed
- Comparative framework designed
- Detailed structured comparisons currently in progress for selected key papers

## Purpose:
This project serves both as:
- A structured academic review resource
- A foundation for future research contributions in Music Source Separation

## Data Table Preview

| paper_id | Reference | Title | Year | paper_type | main_contribution | problem | separation_task | channel_configuration | training_dataset | test_dataset | real_or_synthetic_mix | data_augmentation | preprocessing | visualization_type | learning_type | architecture_family | specific_models | loss_function | evaluation_metrics | separation_performance |
|---:|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Araki, S., Ito, N., Haeb-Umbach, R., Wichern, G., Wang, Z. Q., & Mitsufuji, Y. (2025, April). 30+ years of source separation research: Achievements and future challenges. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE. | 30+ Years of Source Separation Research: Achievements and Future Challenges | 2025 | Review | A comprehensive survey summarizing more than 30 years of research in speech, audio, and music source separation, covering model-based, deep learning-based, and hybrid approaches, as well as benchmarks, datasets, evaluation metrics, and future research directions | Separating individual acoustic source signals from one or more mixtures under blind or semi-blind settings | Speech separation / audio source separation / music source separation | Single-channel and Multi-channel | - | - | - | - | STFT-based time-frequency representation (general problem formulation) | Time-Frequency domain | Supervised + Unsupervised (surveyed) | Model-based + Deep Learning + Hybrid | ICA / IVA / ILRMA / NMF / TF masking / Beamforming / Conv-TasNet / DPRNN / Transformer-based models | - | SDR / SI-SDR / SIR / SAR / STOI / PESQ / WER / MOS | - |
| 2 | Lu, W. T., Wang, J. C., Kong, Q., & Hung, Y. N. (2024, April). Music source separation with band-split rope transformer. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 481-485). IEEE. | MUSIC SOURCE SEPARATION WITH BAND-SPLIT ROPE TRANSFORMER | 2024 | Research (ICASSP paper) | Proposes BS-RoFormer, a frequency-domain music source separation model using a band-split front-end and hierarchical interleaved Transformers with Rotary Position Embedding (RoPE) for multi-band complex mask estimation; achieves state-of-the-art SDR and ranked 1st in SDX’23 music separation track | Music source separation (4-stem) from stereo recordings is challenging due to complex signals and multiple sources; improve frequency-domain modeling efficiency and separation quality by leveraging band-wise priors and Transformer sequence modeling with stable positional encoding | 4-stem MSS (vocals, bass, drums, other) via complex ideal ratio mask (cIRM) estimation in STFT domain | Stereo (2-channel) | MUSDB18HQ + In-House (500 songs; 450 train, 50 val) for SDX’23 system; ablations use MUSDB18HQ only | MUSDB18HQ test/eval set (50 songs); SDX’23 organizer private test set for leaderboard C | Real (recorded multitrack stems) + synthetic mixtures via random mixing of stems (dynamic pool) | Random gain ±3 dB per stem; 10% chance a stem replaced with silence; random mixing of stems possibly from different songs; random 8-second cropping with loudness > -50 dB; dynamic segment pool | STFT/iSTFT; complex spectrogram input; band-split into 62 non-overlapping subbands; segment enframe/deframe (TC or OA); RMSNorm; mixed precision (STFT/iSTFT FP32, others FP16); EMA; flash attention; checkpointing | Time-Frequency (complex spectrogram) | Supervised | Transformer (hierarchical/interleaved) | BS-RoFormer (Band-Split RoPE Transformer); baseline comparisons include BSRNN, HTDemucs, etc. | Time-domain MAE + multi-resolution complex spectrogram MAE (S=5 multi-resolution STFTs) | SDR (museval); median SDR reporting (per 1-second chunks) | 9.80 dB average SDR on MUSDB18HQ without extra data (BS-RoFormer L=6, OA); 9.97 dB mean global SDR on SDX’23 leaderboard C final (private set); 11.99 dB avg SDR for SDX’23 submission model with extra data (L=12, OA) |
| 3 | Mariani, G., Tallini, I., Postolache, E., Mancusi, M., Cosmo, L., & Rodolà, E. (2023). Multi-source diffusion models for simultaneous music generation and separation. arXiv preprint arXiv:2302.02257. | MULTI-SOURCE DIFFUSION MODELS FOR SIMULTANEOUS MUSIC GENERATION AND SEPARATION | 2024 | Conference (ICLR) | Introduces a Multi-Source Diffusion Model that learns the joint distribution of contextual musical stems, enabling total generation, partial generation (source imputation), and source separation within a single unified model, and proposes a novel Dirac-based likelihood for posterior score estimation in separation | Joint modeling of multiple dependent musical sources to enable both generation and separation without conditioning architecture on mixtures | Music source separation; music mixture generation; partial generation (source imputation) | Multi-source (4 stems); mono waveform per stem (stacked channels) | Slakh2100 (145h; 4 most abundant classes: Bass, Drums, Guitar, Piano) | Slakh2100 test set | Real (rendered multi-track stems) | NG | Waveform domain; Gaussian forward diffusion process; 22kHz resampling; chunking (~12s); noise schedule with σ(t); ODE-based sampling | Waveform (time-domain) | Supervised (score-matching diffusion); also weakly-supervised variant with independent priors | Diffusion model (Score-based; U-Net backbone) | MSDM (Multi-Source Diffusion Model); ISDM; MSDM Dirac; MSDM Gaussian | Denoising score-matching loss (MSE between predicted and true denoised signal) | SI-SDRI; FAD; sub-FAD; subjective listening scores (quality; coherence; density) | Up to 17.27 dB SI-SDRI (ISDM Dirac with correction); MSDM Dirac comparable to Demucs baseline; competitive with state-of-the-art regressor models on Slakh2100 |
| 4 | Fabbro, G., Uhlich, S., Lai, C. H., Choi, W., Martínez-Ramírez, M., Liao, W., ... & Mitsufuji, Y. (2023). The Sound Demixing Challenge 2023– Music Demixing Track. arXiv preprint arXiv:2308.06979. | D3Net: Densely Connected Multi-Dilated DenseNet for Music Source Separation | 2021 | Conference (ICASSP) | Proposes D3Net, a densely connected multi-dilated DenseNet architecture that enlarges the receptive field while maintaining parameter efficiency for music source separation, achieving improved SDR compared to prior convolutional models | Improving music source separation performance by better capturing multi-scale temporal and spectral context with efficient convolutional modeling | Music source separation (4 stems: vocals, drums, bass, other) | Stereo (2-channel) | MUSDB18 | MUSDB18 test set | Real (professionally recorded multitrack stems) | Data augmentation via random mixing and segment cropping (NG exact details) | STFT-based magnitude spectrogram input; U-Net style encoder-decoder with dense and dilated convolutions | Time-Frequency (magnitude spectrogram) | Supervised | Convolutional Neural Network (DenseNet-based U-Net) | D3Net | L1 loss on magnitude spectrogram (NG if exact formulation unspecified) | SDR (BSS Eval / museval) | Improved average SDR over prior CNN-based baselines on MUSDB18 (exact value NG) |
| 5 | Sawata, R., Takahashi, N., Uhlich, S., Takahashi, S., & Mitsufuji, Y. (2024). The whole is greater than the sum of its parts: improving music source separation by bridging networks. EURASIP Journal on Audio, Speech, and Music Processing, 2024(1), 39. | Hybrid Demucs: A Hybrid Spectrogram and Waveform Source Separation Model | 2023 | Conference (ICASSP) | Proposes Hybrid Demucs, a model that combines waveform-domain and spectrogram-domain U-Net architectures to leverage both time-domain and frequency-domain representations for improved music source separation | Improving music source separation by jointly modeling waveform and spectrogram features to overcome limitations of purely time-domain or frequency-domain approaches | Music source separation (4 stems: vocals, drums, bass, other) | Stereo (2-channel) | MUSDB18-HQ + additional proprietary and public datasets (as part of training strategy) | MUSDB18-HQ test set | Real (professionally recorded multitrack stems) + synthetic mixtures during training | Data augmentation including random remixing of stems, pitch shifting, time stretching, and random segment cropping | STFT for spectrogram branch; raw waveform input for time-domain branch; hybrid encoder-decoder with cross-domain fusion | Waveform + Time-Frequency (Hybrid) | Supervised | Hybrid CNN-based U-Net (waveform + spectrogram) | Hybrid Demucs | L1 loss on waveform + multi-resolution STFT loss | SDR (museval) | State-of-the-art SDR on MUSDB18-HQ at time of publication (exact value NG) |

## Data Cloumns Descriptions:
- paper_id: Unique identifier assigned to each paper in the table; ensures consistent indexing, referencing, and cross-linking across analyses.
- Reference: Full bibliographic citation of the paper; provides traceability, proper attribution, and direct lookup in academic databases.
- abstract_summary_tr: Concise Turkish summary of the paper’s abstract; enables quick understanding for Turkish readers and supports bilingual review documentation.
- Title: The official paper title; helps quickly identify and search the work.
- Year: Publication year; useful for tracking trends and comparing methods over time.
- paper_type: Paper category (e.g., research, survey, dataset/tool); clarifies the paper’s intent and how to compare it.
- main_contribution: The key novelty or value of the paper; provides a quick reason why the work matters.
- problem: The specific challenge the paper addresses; helps align papers tackling similar limitations.
- separation_task: The separation setting/task (e.g., 4-stem MSS, speech separation); enables task-level grouping and fair comparison.
- channel_configuration: Input channel setup (mono/stereo/multi-channel); important for comparing assumptions and model applicability.
- training_dataset: Datasets used for training; indicates data scale/domain and reproducibility.
- test_dataset: Datasets used for evaluation; ensures results are comparable across papers.
- real_or_synthetic_mix: Whether mixtures are real recordings, synthetic mixes, or both; impacts realism and generalization.
- data_augmentation: Augmentation methods used during training; helps interpret gains and replication details.
- preprocessing: Signal processing steps (e.g., STFT settings, normalization); affects model input and fairness in comparisons.
- visualization_type: Representation domain (waveform, spectrogram, complex TF, hybrid); summarizes what the model “sees.”
- learning_type: Training paradigm (supervised/unsupervised/semi-supervised); key for comparing label requirements.
- architecture_family: High-level model family (CNN/RNN/Transformer/Diffusion/Hybrid); supports taxonomy-level analysis.
- specific_models: Concrete model names/variants; useful for linking to code, checkpoints, and known baselines.
- loss_function: Optimization objective(s); critical for understanding training behavior and performance drivers.
- evaluation_metrics: Metrics used (SDR, SI-SDRi, etc.); enables consistent reporting and metric-aware comparison.
- separation_performance: Reported results (scores/claims); the main quantitative outcome for benchmarking.
