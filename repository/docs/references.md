# References

This file collects the citations that ground the design decisions in
`docs/scientific_upgrades.md` and across the pipeline.

## Bundle adjustment, SfM, MVS

- Schönberger, J. L., & Frahm, J.-M. (2016). *Structure-from-Motion Revisited*. **CVPR 2016**. (COLMAP.)
- Schönberger, J. L., Zheng, E., Frahm, J.-M., & Pollefeys, M. (2016). *Pixelwise View Selection for Unstructured Multi-View Stereo*. **ECCV 2016**. (COLMAP MVS.)
- Lindenberger, P., Sarlin, P.-E., Larsson, V., & Pollefeys, M. (2021). *Pixel-Perfect Structure-from-Motion with Featuremetric Refinement*. **ICCV 2021**. (PixSfM.)

## Robust ICP and registration

- Besl, P. J., & McKay, N. D. (1992). *A Method for Registration of 3-D Shapes*. **IEEE TPAMI**, 14(2), 239–256.
- Chen, Y., & Medioni, G. (1991). *Object Modelling by Registration of Multiple Range Images*. **Image and Vision Computing**. (Point-to-plane ICP.)
- Rusinkiewicz, S., & Levoy, M. (2001). *Efficient Variants of the ICP Algorithm*. **3DIM 2001**.
- Zhang, J., Yao, Y., & Deng, B. (2021). *Fast and Robust Iterative Closest Point*. **IEEE TPAMI** / arXiv 2007.07627.
- Tuley, J., & Singh, S. (2024). *A Field Analysis on Degeneracy-aware Point Cloud Registration in the Wild*. arXiv 2408.11809.
- Rusu, R. B., Blodow, N., & Beetz, M. (2009). *Fast Point Feature Histograms (FPFH) for 3D Registration*. **ICRA 2009**.

## Robust statistics (used by ICP kernels and bootstrap CIs)

- Huber, P. J. (1964). *Robust Estimation of a Location Parameter*. **Annals of Mathematical Statistics**, 35(1), 73–101. (Huber loss.)
- Beaton, A. E., & Tukey, J. W. (1974). *The Fitting of Power Series, Meaning Polynomials, Illustrated on Band-Spectroscopic Data*. **Technometrics**, 16(2), 147–185. (Tukey biweight.)
- Wilson, E. B. (1927). *Probable Inference, the Law of Succession, and Statistical Inference*. **JASA**, 22(158), 209–212. (Wilson score interval.)
- Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife*. **Annals of Statistics**, 7(1), 1–26.

## Scan-to-BIM and progress monitoring

- Bosché, F. (2010). *Automated Recognition of 3D CAD Model Objects in Laser Scans and Calculation of As-Built Dimensions for Dimensional Compliance Control in Construction*. **Advanced Engineering Informatics**, 24(1).
- Tuttas, S., Braun, A., Borrmann, A., & Stilla, U. (2017). *Acquisition and Consecutive Registration of Photogrammetric Point Clouds for Construction Progress Monitoring*. **PFG**.
- Kavaliauskas, P., Fernandez-Cabal, B. J., & Migilinskas, D. (2022). *Automation of Construction Progress Monitoring by Integrating 3D Point Cloud Data with an IFC-Based BIM Model*. **Buildings**, 12(10), 1754.
- Knapitsch, A., Park, J., Zhou, Q.-Y., & Koltun, V. (2017). *Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction*. **ACM TOG (SIGGRAPH 2017)**, 36(4). (Accuracy/completeness/F-score @ τ.)
- *Enhanced Objective Function for Point Cloud Completion* (2024). arXiv 2505.14218. (Forward/backward Chamfer decomposition.)

## Monocular depth (Stage 6 reference)

- Yang, L., Kang, B., Huang, Z., et al. (2024). *Depth Anything V2*. **NeurIPS 2024** / arXiv 2406.09414.
- Bhat, S. F., Birkl, R., Wofk, D., et al. (2023). *ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth*. arXiv 2302.12288.

## Gaussian splatting (Stage 4.5 / 7.7 stance)

- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. **ACM TOG (SIGGRAPH 2023)**.
- Huang, B., Yu, Z., Chen, A., Geiger, A., & Gao, S. (2024). *2D Gaussian Splatting for Geometrically Accurate Radiance Fields*. **SIGGRAPH 2024** / arXiv 2403.17888. (Why naive 3DGS surfaces are not metric.)
- Wolf, Y., Bracha, A., & Kimmel, R. (2024). *Surface Reconstruction from Gaussian Splatting via Novel Stereo Views* (GS2Mesh). https://gs2mesh.github.io/

## Software components

- Zhou, Q.-Y., Park, J., & Koltun, V. (2018). *Open3D: A Modern Library for 3D Data Processing*. arXiv 1801.09847.
- Agarwal, S., Mierle, K., et al. *Ceres Solver*. http://ceres-solver.org (used by COLMAP for BA.)
- IfcOpenShell. https://ifcopenshell.org
