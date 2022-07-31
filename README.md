# PyFOCUSR

Python implementation of the FOCUSR <br>
FOCUSR = Feature Oriented Correspondence using Spectral Regularization and is described in [1]. 

# Introduction / Background

This package will find correspondence between points on two surfaces using spectral coordinate information to regularize the surface matching. Non-rigid registration between surface points is conducted using Coherent Point Drift (CPD) as described in [2] and impelmented in CyCPD [3]. More recent versions/updated versions of this (spectral alignment/registration) algorithm were developed by the original authors [4], [5] but are not covered here. 

[1] Lombaert H, Grady L, Polimeni JR, Cheriet F. FOCUSR: Feature Oriented Correspondence Using Spectral Regularization--A Method for Precise Surface Matching. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2013;35(9):2143-2160. doi:10.1109/TPAMI.2012.276<br>

[2] Myronenko A, Xubo Song. Point Set Registration: Coherent Point Drift. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2010;32(12):2262-2275. doi:10.1109/TPAMI.2010.46<br>
    An open-source version of the manuscript can be found here: https://tinyurl.com/tph4u7e<br>

[3] https://github.com/gattia/cycpd<br>

[4] Lombaert H, Sporring J, Siddiqi K. Diffeomorphic Spectral Matching of Cortical Surfaces. In: Gee JC, Joshi S, Pohl KM, Wells WM, Zöllei L, eds. Information Processing in Medical Imaging. Vol 7917. Lecture Notes in Computer Science. Berlin, Heidelberg: Springer Berlin Heidelberg; 2013:376-389. doi:10.1007/978-3-642-38868-2_32 <br>

[5] Lombaert H, Arcaro M, Ayache N. Brain Transfer: Spectral Analysis of Cortical Surfaces and Functional Maps. In: Ourselin S, Alexander DC, Westin C-F, Cardoso MJ, eds. Information Processing in Medical Imaging. Vol 9123. Lecture Notes in Computer Science. Cham: Springer International Publishing; 2015:474-487. doi:10.1007/978-3-319-19992-4_37


# Installation
It is recommended that installtion is performed in a new environment. 

```bash
conda create --name focusr python=3.7
git clone https://github.com/gattia/pyfocusr
cd pyfocusr

# install dependencies
make requirements

# install pyfocusr
make install
```

# Examples

Jupyter notebook *Example_registering_two_bone_meshes*  in /examples shows extended example with visualizations along the way. Some example steps include:

### Spectral coordinates
Normalized spectral coordinates (eigenvectors) are calculated for each mesh. Below shows examples for the first 3 eigenvectors. 

| *Eigen Vector 1 - Fiedler vector*    | *Eigen Vector 2*           | *Eigen Vector 3*           |
| :---:                               | :---:                      | :---:                      |
|![](/images/eig_vec_1_fiedler.png)   | ![](/images/eig_vec_2.png) | ![](/images/eig_vec_3.png) |


Next, the spectral coordinates (eigenvectors) for each node of the mesh (shown above) are used as xyz positions and are aligned. 
#### 
![](/images/ezgif.com-gif-maker.gif)


The following includes the meshes at various steps of the registration process as well as one mesh calculated as the average of the source & target. 

|*Source Mesh* | *Target Mesh* |
|:---:       |:---:        |
|![](/images/source.png)   | ![](/images/target.png) |
|*Source Transformed to Target*               | *Average Mesh*                |
| ![](/images/mesh_transformed_to_target.png) | ![](/images/average_mesh.png) |


MIT License.
