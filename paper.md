# SET-BASED TRANSFORMER FOR ATMOSPHERIC COMPENSATION IN STANDOFF LWIR HYPERSPECTRAL IMAGING

**Fabian Perez, Nicolas Quintero, Jeferson Acevedo, Hoover Rueda-Chacón**  
Department of Computer Science, Universidad Industrial de Santander  
Bucaramanga, 680002, Colombia  
`{perez2258059,nicolas2220090,jeferson2221790}@correo.uis.edu.co, hfarueda@uis.edu.co`

## Abstract

Passive long-wave infrared (LWIR) hyperspectral imaging under a standoff geometry depends on atmospheric absorption and emission, as well as reflected radiance, thus making atmospheric compensation essential to get knowledge of a target of interest. Despite its importance, this compensation has been largely overlooked due to its practical and modeling difficulty. In this paper, we present a lightweight set-based deep learning framework that takes multiple radiance measurements, collected at different standoff ranges, as input and jointly estimates transmittance, atmospheric path radiance, and a shared downwelling spectrum. We analyze the learned representation with a sparse autoencoder and observe that several latent features do activate on geographically coherent subsets of the test data despite the absence of location supervision. Experiments on a MODTRAN generated standoff LWIR dataset demonstrate low spectral distortion across all estimated products. The dataset and code will be made publicly available.

**Index Terms—** LWIR, atmospheric compensation, hyperspectral imaging, sparse autoencoder.

## I. INTRODUCTION

Unlike the visible to shortwave infrared (VIS-SWIR) range, which is dominated by reflected solar radiation (0.4-2.5 µm) [1], the long-wave infrared (LWIR) senses thermally emitted radiation within the 8-14 µm atmospheric window [2]. This enables sensing capabilities independent of solar illumination, thus making LWIR suitable for day-night operation. As a result, LWIR has been widely used in security and surveillance [3, 4], autonomous driving and infrastructure monitoring [5, 6], wildfire detection [7], environmental remote sensing [8], and hyperspectral target detection [9].

LWIR measurements acquired under a standoff sensing configuration are governed by the radiative transfer equation (RTE) and are affected by atmospheric attenuation through the path transmittance $\tau$, atmospheric absorption and self-emission $L_a$ (upwelling), as well as by surface reflection $L_{\mathrm{ref}}$ driven by downwelling $L_d$. The forward standoff model, following the RTE in [2, 10], is given by

$$
L(\lambda; r_n; T) = \tau(\lambda; r_n)
\left[
\underbrace{\epsilon\, B(\lambda; T_0)}_{L_{\mathrm{obj}}}
+
\underbrace{\rho\, L_d(\lambda)}_{L_{\mathrm{ref}}}
\right]
+ L_a(\lambda; r_n).
\tag{1}
$$

where $L(\lambda; r_n; T)$ denotes the at-sensor radiance at wavelength $\lambda$ for the $n$-th standoff range (distance) $r_n$ and target temperature $T$, $\epsilon$ is the object emmisivity and $\rho$ is the surface albedo. Unlike airborne and satellite remote sensing, standoff acquisition operates at shorter ranges where the same scene may be observed across multiple standoff distances, making atmospheric effects range-dependent as depicted in Fig. 1. Hence, accurate downstream tasks on the target require atmospheric compensation (AC), which demands the estimation of transmittance, upwelling, and downwelling (TUD).

*Fig. 1. Standoff LWIR imaging configuration. The atmosphere is discretized into 126 layers. Downwelling irradiance from the sky $L_d$ illuminates the target. The at-sensor radiance at the hyperspectral LWIR camera is shown as the sum of (i) target thermal emission $L_{\mathrm{obj}}$, (ii) reflected downwelling irradiance $L_{\mathrm{ref}}$, and (iii) atmospheric path emission along the line of sight $L_a$.*

Building on this, the AC literature has largely focused on airbone and satellite rather than standoff sensing. In near-nadir viewing geometry, the upwelling radiance and transmittance can be treated as approximately the same for all pixels [11], which simplifies estimation. State-of-the-art works like [12] proposed a neural network to estimate atmospheric components jointly with land-surface temperature, but the method is tailored to satellite nadir observations. In [13], a variational autoencoder (VAE) learning framework is proposed, again restricted to the nadir regime. Oblique remote configurations, where upwelling and transmittance vary across the field of view have been explored in [14], but assume known range or rely on MODTRAN [15] (MODerate resolution atmospheric TRANsmission) look-up tables for AC estimation. Similarly, [16] proposes a convolutional neural network (CNN) to estimate atmospheric products (TUD estimation) but evaluates only remote configurations.

In standoff LWIR, near-horizontal and range-diverse paths make the radiative terms strongly geometry-dependent, and the problem remains comparatively underexplored. In [17], a diffusion-based framework models long-range absorption for VIS/NIR/SWIR but does not address LWIR. In [18], a CNN-based approach is proposed but neglects the atmospheric path-radiance contribution. In [19], absorption is leveraged to estimate depth, while omitting the reflected downwelling term in the RTE. Collectively, these limitations highlight the need for AC methods that explicitly handle LWIR standoff imaging.

In this work, we propose a lightweight set-based deep learning framework for atmospheric compensation in passive standoff LWIR imaging. Here, set-based refers to jointly processing multiple radiance measurements acquired at different standoff ranges, without assuming any ordering [20]. By jointly exploiting multiple radiance measurements acquired at different standoff ranges, the method estimates the following AC products: atmospheric transmittance ($\tau$), upwelling radiance ($L_a$), and a shared downwelling spectrum ($L_d$). The main contributions are listed as follows:

- A set-based framework for joint TUD estimation in passive standoff LWIR imaging, coupled with a sparse autoencoder to enable latent-space interpretability.
- The first publicly available MODTRAN-generated dataset for atmospheric compensation in standoff LWIR imaging.

## II. PROPOSED METHOD

We propose a set-based framework to estimate AC products from standoff hyperspectral LWIR observations as illustrated in Fig. 2. Each sample input consist of $N$ measurements acquired at different standoff ranges, each represented as a spectrum over $B$ spectral bands. We first describe the neural network that predicts the AC products, and then introduce a sparse autoencoder that exposes structure in the learned latent representations.

*Fig. 2. Proposed Framework. The input consists of $N$ radiance measurements selected from the globally generated dataset by sampling a single location (pink dots on the globe) and extracting its observations at $N$ different standoff ranges; each sample has $B = 256$ spectral bands beetween $8\,\mu\mathrm{m}$ and $13\,\mu\mathrm{m}$, stacked as $X \in \mathbb{R}^{N \times B}$. A Set-Transformer encoder with two ISAB modules encode the set of measurements into a latent representation. The latent features are then routed to three task-specific decoders: an FFN with a sigmoid activation predicts atmospheric transmittance $\hat{Y}_\tau \in \mathbb{R}^{N \times B}$; an FFN predicts atmospheric path radiance $\hat{Y}_a \in \mathbb{R}^{N \times B}$; a PMA module followed by an MLP aggregates across the set to estimate a shared downwelling spectrum $\hat{y}_d \in \mathbb{R}^{B}$.*

### A. Set-based Neural Network

Let the $n$-th standoff measurement, acquired at range $r_n$, be a radiance spectrum $x_n \in \mathbb{R}^{B}$, and let the input set be written as $X \in \mathbb{R}^{N \times B}$ whose rows are the elements $\{x_n\}_{n=1}^{N}$ collected at distinct standoff ranges $\mathcal{R} = \{r_n\}_{n=1}^{N}$. Note that multiple ranges provide additional constraints that mitigate the ill-posed problem of estimating the (AC) products, since $\tau$ and $L_a$ vary with $r_n$. The network predicts three outputs: a range-wise transmittance $\hat{Y}_\tau \in \mathbb{R}^{N \times B}$, a range-wise upwelling $\hat{Y}_a \in \mathbb{R}^{N \times B}$, and a shared downwelling spectrum $\hat{y}_d \in \mathbb{R}^{B}$. We denote the three task-specific branches as

$$
\hat{Y}_\tau = G_T(X), \qquad
\hat{Y}_a = G_U(X), \qquad
\hat{y}_d = F_d(X).
\tag{2}
$$

**Symmetry requirements.** Because the $N$ measurements form a set, their ordering is arbitrary. Let $P \in \{0, 1\}^{N \times N}$ be a permutation matrix and denote the row-permuted input by $P \cdot X$. Accordingly, $\hat{Y}_\tau$ and $\hat{Y}_a$ are predicted per standoff range, so each output must stay aligned with the same input; thus, each branch must be permutation equivariant, that is,

$$
G_T(P \cdot X) = P \cdot G_T(X), \qquad
G_U(P \cdot X) = P \cdot G_U(X).
\tag{3}
$$

In contrast, downwelling is shared across the $N$ standoff ranges, for a fixed atmosphere, and therefore must be permutation invariant, which can be denoted as

$$
F_d(P \cdot X) = F_d(X).
\tag{4}
$$

**Set encoder.** To satisfy the requirements described above, we adopt a Set-Transformer encoder [21]. Specifically, we use ISAB (Induced Set Attention Block) which is built from a Multihead Attention Block (MAB) [22]. Using $m$ inducing points $I \in \mathbb{R}^{m \times d}$, ISAB introduces a compact set of learnable latent anchors that attend to the input set to form a global summary $S$, and then lets each element in $X$ attend to this summary [21]. ISAB can be expressed as:

$$
\mathrm{ISAB}_m(X) = \mathrm{MAB}(X, S) \in \mathbb{R}^{N \times d},
\tag{5}
$$

where

$$
S = \mathrm{MAB}(I, X) \in \mathbb{R}^{m \times d}.
\tag{6}
$$

We use two stacked ISAB layers to increase the depth of the set encoder so as to capture higher-order interactions

$$
Z_2 = \mathrm{ISAB}_2\!\left(\mathrm{ISAB}_1(X)\right).
\tag{7}
$$

ISAB is permutation equivariant by construction, so permuting the input permutes the latent tokens in the same way [21].

**Equivariant transmittance and upwelling decoders.** The transmittance and upwelling branches apply the same feed-forward network (FFN) independently to each latent token. Concisely, we use two token-wise FFNs, $\mathrm{FFN}_\phi$ and $\mathrm{FFN}_\psi$, each mapping $\mathbb{R}^{d} \to \mathbb{R}^{B}$ and applied to the encoder output $Z_2$:

$$
\hat{Y}_\tau = \sigma\!\left(\mathrm{FFN}_\phi(Z_2)\right), \qquad
\hat{Y}_a = \mathrm{FFN}_\psi(Z_2),
\tag{8}
$$

where $\sigma(\cdot)$ is an element-wise sigmoid that enforces $\hat{Y}_\tau \in (0,1)$, and $\mathrm{FFN}_\phi$, $\mathrm{FFN}_\psi$ share the same structure but do not share weights across tasks. For the $n$-th token $z_n$, we define the token-wise FFN output by

$$
\big[\mathrm{FFN}(Z_2)\big]_n := \mathrm{FFN}(z_n), \qquad
z_n = [Z_2]_n.
\tag{9}
$$

The FFN computes

$$
h_n^{(1)} = \mathrm{LN}_1\!\left(\mathrm{GELU}(W_1 z_n + b_1)\right),
$$

$$
h_n^{(2)} = \mathrm{LN}_2\!\left(\mathrm{GELU}(W_2 h_n^{(1)} + b_2)\right),
\tag{10}
$$

where LN denotes layer normalization [23] and GELU refer to the activation function [24], and outputs

$$
o_n = W_3 h_n^{(2)} + b_3, \qquad
o_n \in \mathbb{R}^{B}.
\tag{11}
$$

Stacking $\{o_n\}_{n=1}^{N}$ yields $\mathrm{FFN}(Z_2) \in \mathbb{R}^{N \times B}$. Applying the FFN to every token makes the output permutation-equivariant.

**Invariant downwelling decoder.** We assume downwelling is common to all the measurements in a set, since they are acquired under the same atmosphere; therefore, we aggregate $Z_2$ into a single vector using Pooling by Multihead Attention (PMA) [21]. PMA introduces $k$ learned seed vectors $S \in \mathbb{R}^{k \times d}$ and pools the set through attention as implemented by

$$
v = \mathrm{PMA}(Z_2) := \mathrm{MAB}(S, Z_2) \in \mathbb{R}^{k \times d}.
\tag{12}
$$

We use $k = 1$ so that the pooled embedding is $v \in \mathbb{R}^{d}$. We then post-process this pooled representation with a linear projection to obtain the $B$-band downwelling estimate:

$$
\hat{y}_d = \mathrm{MLP}(v), \qquad
\hat{d} \in \mathbb{R}^{B},
\tag{13}
$$

where

$$
\mathrm{MLP}(v) := W_d\, v + b_d.
\tag{14}
$$

**Training objective.** We train the network by minimizing the mean squared error (MSE) between the predicted atmospheric products $(\hat{Y}_\tau, \hat{Y}_a, \hat{y}_d)$ and their corresponding ground-truth targets $(Y_\tau, Y_a, y_d)$. We define the per-sample losses as

$$
L_\tau = \left\lVert \hat{Y}_\tau - Y_\tau \right\rVert_2^2, \qquad
L_a = \left\lVert \hat{Y}_a - Y_a \right\rVert_2^2, \qquad
L_d = \left\lVert \hat{y}_d - y_d \right\rVert_2^2,
\tag{15}
$$

and optimize the total objective $L$,

$$
L = L_\tau + L_a + L_d.
\tag{16}
$$

### B. Sparse Autoencoder (SAE)

To better understand what the trained model encodes and whether its internal features align with physically meaningful factors, we analyze the frozen trained set encoder using a sparse autoencoder (SAE) [25]. Concretely, given an input set $X$, the encoder outputs $Z_2 \in \mathbb{R}^{N \times d}$, and we treat each token $z_n = [Z_2]_n \in \mathbb{R}^{d}$ as an activation vector to be reconstructed with a sparse, overcomplete representation as depicted in Fig. 3.

Let $x \in \mathbb{R}^{d}$ denote a token activation, i.e., $x = z_n$. The SAE encoder produces sparse features $a \in \mathbb{R}^{M}$ using TopK gating [25]:

$$
a := \mathrm{TopK}(W_{\mathrm{enc}} x + b_{\mathrm{enc}}), \qquad
W_{\mathrm{enc}} \in \mathbb{R}^{M \times d},
\tag{17}
$$

where $\mathrm{TopK}(\cdot)$ retains only the $K$ largest-magnitude entries and sets the remaining activations to zero. The decoder then reconstructs the token activation with a linear dictionary:

$$
\hat{x} = a^\top W_{\mathrm{dec}} + b_{\mathrm{dec}}, \qquad
W_{\mathrm{dec}} \in \mathbb{R}^{M \times d}, \; b_{\mathrm{dec}} \in \mathbb{R}^{d}.
\tag{18}
$$

We train the SAE using the per-token $\ell_2$ reconstruction loss:

$$
L_{\mathrm{SAE}}(x) = \left\lVert x - \hat{x} \right\rVert_2^2.
\tag{19}
$$

*Fig. 3. Sparse Autoencoder pipeline. Encoder tokens $Z_2$ are the input to the SAE that reconstructs token activations by minimizing $L_{\mathrm{SAE}}$.*

## III. DATASET GENERATION

To the best of our knowledge, there is currently no publicly available dataset tailored to AC in standoff LWIR imaging. To address this gap, we generate a large-scale simulated dataset using the clear-sky atmospheric profile database (CSP) [26]. This database is derived from the European Centre for Medium-Range Weather Forecast Reanalysis V5 (ERA5) and contains 82,828 profiles with key atmospheric state variables, including latitude, longitude, pressure, temperature, specific humidity and ozone concentration at 136 layers $l_n$. We further filter these profiles by discarding cases with cloud coverage exceeding 10% or relative humidity above 90%, and by removing a subset of ocean-surface profiles, thus yielding 36,547 clear-sky profiles. Radiance at-sensor was simulated using seven target temperatures $T$ ranging from 280K to 310K inclusive, assuming a fixed gray-body like emissivity $\epsilon$ value of 0.95.

MODTRAN5 [15], accessed through an internally developed Python tool, was used to calculate atmospheric products and the at-sensor radiance following Eq. 1. Specifically, it computes band-integrated at-sensor radiance convolving with the instrumental spectral response function (ISRF), and we report radiance in microflicks ($\mu\mathrm{W} \cdot \mathrm{sr}^{-1} \cdot \mathrm{cm}^{-2} \cdot \mu\mathrm{m}^{-1}$). Gas concentrations were specified independently for each of the first 126 layers $\ell_{126}$ (maximum number in MODTRAN5) using the filtered database. Downwelling radiance $L_d$ was calculated at a fixed viewing angle of $45^\circ$ from the target as an approximated representation of the hemispherical sky radiance incident on the target. We define $N = 7$ and $\mathcal{R} = \{30, 90, 150, 210, 270, 330, 390\}\,\mathrm{m}$. The atmospheric path radiance $L_a$ (upwelling) and transmittance $\tau$ were computed along the line of sight for each $r_n$, as illustrated in Fig. 1.

The resulting standoff MODTRAN-generated dataset has 36,547 profiles, 7 ranges, and 7 temperatures, yielding a total of 255,829 samples. The dataset was randomly split into 70% for training, 10% for validation, and 20% for testing.

## IV. RESULTS AND ANALYSIS

We evaluate on the test split of the generated standoff LWIR dataset. We use $N = 1$ with $r_n = 270\,\mathrm{m}$ and $N = 7$ standoff measurements per sample, and an encoder embedding size of $d = 256$. The spectral dimension is set to $B = 256$, consistent with the bands of a real standoff sensor, the DARPA LWIR hyperspectral sensor [19, 27] over the 8-13 µm window; we model each channel spectral response as a Gaussian with FWHM = 40 nm. Training is performed for 510k iterations with batch size 512 and learning rate $1 \times 10^{-3}$, using AdamW with weight decay 0.01. All experiments are run on a single NVIDIA RTX PRO 6000 Blackwell GPU; training requires only $\sim 10\%$ of the available VRAM, indicating that the proposed architecture is lightweight and scalable. Tab. I reports test-set performance for the three estimated AC products. We report Spectral Angle Mapper (SAM) and normalized RMSE (NRMSE). As expected, performance improves substantially from $N = 1$ to $N = 7$, since single-measurement estimation is inherently ill-posed while range-diverse measurements better constrain the solution. Overall, the model achieves low spectral distortion across all AC products, with particularly strong agreement for transmittance. Fig. 4 shows qualitative predictions for one randomly selected test sample, including the estimated $\hat{Y}_\tau$, $\hat{Y}_a$, and $\hat{y}_d$ compared against their corresponding ground-truths. The plotted spectra highlight that the network preserves fine-grained spectral structure.

### TABLE I  
**TEST-SET PERFORMANCE FOR DIFFERENT SET SIZES $N$.**

| N | Target | SAM ↓ | NRMSE ↓ |
|---|---|---:|---:|
| 1 | Transmittance | 0.0057 | 0.0554 |
| 1 | Upwelling | 0.1244 | 0.0564 |
| 1 | Downwelling | 0.1937 | 0.1740 |
| 7 | Transmittance | 0.0025 | 0.0093 |
| 7 | Upwelling | 0.0330 | 0.0093 |
| 7 | Downwelling | 0.0409 | 0.0193 |

*Fig. 4. Qualitative results on one test sample predicted (dashed) and ground-truth (solid) are shown for transmittance $Y_\tau$, atmospheric path radiance $Y_a$, and downwelling radiance $y_d$ across the LWIR window. The left column shows the full spectral range, while the right column provides a zoomed view.*

*Fig. 5. Top-activating locations for two SAE features. Each point denotes a test sample among the highest activations.*

**SAE analysis.** To better understand the latent space learned by the encoder, we train a SAE on the frozen activations $Z_2 \in \mathbb{R}^{N \times d}$. We use a sparse feature dimension of $M = 3072$ (i.e., $12 \times d$) to decouple latent factors, and we optimize the SAE for 20k iterations using AdamW. The TopK gating parameter is set to $k = 16$, selected via cross-validation. After training, we perform a top-activating analysis [28] to identify the samples that most strongly activate each sparse feature. Interestingly, for multiple features, the highest-activation examples consistently correspond to radiance sets originating from highly similar geographic regions, despite the fact that the network is never provided with explicit location. This emergent behavior suggests that the model organizes the features by clustering atmospheres with similar physical conditions, Fig. 5 plots the top-activating examples for two sparse features; note how the activations form clear location-related clusters.

## V. CONCLUSIONS

We presented a set-based deep learning framework for standoff LWIR atmospheric compensation that estimates transmittance, atmospheric path radiance, and a shared downwelling radiance from range-diverse measurements. We showed that SAE provides a powerful interpretability tool for probing physically grounded representations. Future work will model angle-dependent downwelling, incorporate spectrally varying emissivity, and study sensitivity to the number of measurements and the selected standoff ranges, while exploring alternative SAE variants and model capacities.

## VI. ACKNOWLEDGMENTS

This work was supported by the Air Force Office of Scientific Research (AFOSR) through the Southern Office of Aerospace Research and Development (SOARD) under grant number FA8655-25-1-8010

## REFERENCES

[1] B.-C. Gao, M. Montes, C. Davis, and A. Goetz, “Atmospheric correction algorithms for hyperspectral remote sensing data of land and ocean,” *Remote Sensing of Environment*, vol. 113, 09 2009.

[2] D. Manolakis, M. Pieper, E. Truslow, R. Lockwood, A. Weisner, J. Jacobson, and T. Cooley, “Longwave infrared hyperspectral imaging: Principles, progress, and challenges,” *IEEE Geoscience and Remote Sensing Magazine*, vol. 7, no. 2, pp. 72-100, 2019.

[3] C. Rodríguez-Pulecio, H. Benítez-Restrepo, and A. Bovik, “Making long-wave infrared face recognition robust against image quality degradations,” *Quantitative InfraRed Thermography Journal*, vol. 16, no. 3-4, pp. 218-242, Oct. 2019.

[4] T. Müller and D. Manger, “Person detection in LWIR imagery using image retrieval,” in *Automatic Target Recognition XXIII*, F. A. Sadjadi and A. Mahalanobis, Eds., vol. 8744, International Society for Optics and Photonics. SPIE, 2013, p. 87440E.

[5] N. Li, Y. Zhao, Q. Pan, S. G. Kong, and J. C.-W. Chan, “Illumination-invariant road detection and tracking using lwir polarization characteristics,” *ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 180, pp. 357-369, 2021.

[6] K. M. Judd, M. P. Thornton, and A. A. Richards, “Automotive sensing: assessing the impact of fog on lwir, mwir, swir, visible, and lidar performance,” in *Infrared Technology and Applications XLV*, B. F. Andresen, G. F. Fulop, and C. M. Hanson, Eds., vol. 11002. SPIE: International Society for Optics and Photonics, 2019, p. 110021F.

[7] J. M. Johnston, M. J. Wooster, and T. J. Lynham, “Experimental confirmation of the mwir and lwir grey body assumption for vegetation fire flame emissivity,” *International Journal of Wildland Fire*, vol. 23, no. 4, pp. 463-479, 05 2014.

[8] J. Feng, D. Rogge, and B. Rivard, “Comparison of lithological mapping results from airborne hyperspectral vnir-swir, lwir and combined data,” *International Journal of Applied Earth Observation and Geoinformation*, vol. 64, pp. 340-353, 2018.

[9] J. A. Hackwell, D. W. Warren, R. P. Bongiovi, S. J. Hansel, T. L. Hayhurst, D. J. Mabry, M. G. Sivjee, and J. W. Skinner, “LWIR/MWIR imaging hyperspectral sensor for airborne and ground-based remote sensing,” in *Imaging Spectrometry II*, M. R. Descour and J. M. Mooney, Eds., vol. 2819, International Society for Optics and Photonics. SPIE, 1996, pp. 102-107.

[10] K. Thome, F. Palluconi, T. Takashima, and K. Masuda, “Atmospheric correction of aster,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 36, no. 4, pp. 1199-1211, 1998.

[11] M. Pieper, D. Manolakis, E. Truslow, T. Cooley, M. Brueggeman, A. Weisner, and J. Jacobson, “In-scene lwir downwelling radiance estimation,” in *Imaging Spectrometry XXI*, vol. 9976. SPIE, 2016, pp. 74-92.

[12] M. Chen, L. Ni, X. Jiang, Z. Li, and H. Wu, “Retrieval of atmospheric and land surface parameters from satellite-based thermal infrared hyperspectral data using an artificial neural network technique,” in *IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium*. IEEE, 2018, pp. 2745-2748.

[13] N. Westing, K. C. Gross, B. J. Borghetti, C. M. S. Kabban, J. Martin, and J. Meola, “Multimodal representation learning and set attention for lwir in-scene atmospheric compensation,” *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 14, pp. 127-140, 2020.

[14] D. S. O’Keefe, S. N. Nauyoks, M. R. Hawks, J. Meola, and K. C. Gross, “Oblique in-scene atmospheric compensation,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 60, pp. 1-15, 2022.

[15] A. Berk, G. P. Anderson, P. K. Acharya, L. S. Bernstein, L. Muratov, J. Lee, M. J. Fox, S. M. Adler-Golden, J. H. Chetwynd Jr, M. L. Hoke et al., “Modtran5: A reformulated atmospheric band model with auxiliary species and practical multiple scattering options,” in *Algorithms and Technologies for Multispectral, Hyperspectral, and Ultraspectral Imagery X*, vol. 5425. SPIE, 2004, pp. 341-347.

[16] F. Xu, G. Cervone, G. Franch, and M. Salvador, “Multiple geometry atmospheric correction for image spectroscopy using deep learning,” *Journal of Applied Remote Sensing*, vol. 14, no. 2, pp. 024 518-024 518, 2020.

[17] D. Stelter, E. Brewer, and R. Sundberg, “Atmospheric correction using diffusion models and modtran for constrained training,” in *Algorithms, Technologies, and Applications for Multispectral and Hyperspectral Imaging XXX*, vol. 13031. SPIE, 2024, pp. 93-102.

[18] F. Bao, X. Wang, S. H. Sureshbabu, G. Sreekumar, L. Yang, V. Aggarwal, V. N. Boddeti, and Z. Jacob, “Heat-assisted detection and ranging,” *Nature*, vol. 619, no. 7971, pp. 743-748, 2023.

[19] U. Dorken Gallastegi, H. Rueda-Chacon, M. J. Stevens, and V. K. Goyal, “Absorption-based, passive range imaging from hyperspectral thermal measurements,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 47, no. 5, pp. 4044-4060, 2025.

[20] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. R. Salakhutdinov, and A. J. Smola, “Deep sets,” in *Advances in Neural Information Processing Systems*, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017.

[21] J. Lee, Y. Lee, J. Kim, A. Kosiorek, S. Choi, and Y. W. Teh, “Set transformer: A framework for attention-based permutation-invariant neural networks,” in *International Conference on Machine Learning*. ICML, 2019, pp. 3744-3753.

[22] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[23] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” *arXiv preprint arXiv:1607.06450*, 2016.

[24] D. Hendrycks, “Gaussian error linear units (gelus),” *arXiv preprint arXiv:1606.08415*, 2016.

[25] A. Makhzani and B. Frey, “K-sparse autoencoders,” *arXiv preprint arXiv:1312.5663*, 2013.

[26] S. L. Ermida and I. F. Trigo, “A comprehensive clear-sky database for the development of land surface temperature algorithms,” *Remote Sensing*, vol. 14, no. 10, 2022.

[27] F. Yellin, S. McCloskey, C. Hill, E. Smith, and B. Clipp, “Concurrent band selection and traversability estimation from long-wave hyperspectral imagery in off-road settings,” in *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 2024, pp. 7483-7492.

[28] L. Gao, T. D. la Tour, H. Tillman, G. Goh, R. Troll, A. Radford, I. Sutskever, J. Leike, and J. Wu, “Scaling and evaluating sparse autoencoders,” in *The Thirteenth International Conference on Learning Representations*, 2025.
