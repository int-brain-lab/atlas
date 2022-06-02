# Brain flatmaps generation

This repository describes the methods and provides the Python code for the reproduction of an existing method for generating brain flatmaps based on the Allen Mouse Brain atlas.

![](screenshot.png)

> **Note:** this is a work in progress. Only part of the method (streamlines) has been implemented so far.

> **Note:** the code has only tested on the isocortex region so far. We plan to run it on other regions.
>

## How to run it

A powerful computer is required to handle the 10um Allen atlas volume.

### Hardware requirements

- At least 64GB of RAM are recommended.
- At least 250GB of free space on an SSD.
- A NVIDIA graphics processing unit (GPU) with at least 8GB of video memory is required.

### Software requirements

- Python 3
- NumPy
- SciPy
- Numba
- Cupy
- nrrd
- h5py
- tqdm

### Running the code

The code requires data files that we do not provide directly. If needed, ask [Cyrille Rossant](https://cyrille.rossant.net/) for more guidance.

The input files are to put in `input/`, the code will generate output files in `regions/isocortex/`.

1. Put the following input files in the `input/` subfolder:

   - `isocortex_boundary_10.nrrd`
   - `isocortex_mask_10.nrrd`

2. Run `python surface.py`. This script should create:

    - `regions/isocortex/mask.npy` (a volume with labels indicating the surfaces and the regions between them)
    - `regions/isocortex/normal.npy` (a 3D vector field with the surface normal vectors)

3. Run `python laplacian.py`. This script should create:

    - `regions/isocortex/laplacian.npy` (a 3D scalar field within the region volume)

4. Run `python gradient.py`. This script should create:

    - `regions/isocortex/gradient.npy` (a 3D vector field with the laplacian's gradient)

5. Run `python streamlines.py`. This script should create:

    - `regions/isocortex/streamlines.npy` (a 3D array with N 3D paths of size 100 in the 3D coordinate space of the 10um volume)

6. The next steps for generating the flatmaps using the streamlines are not yet available.

#### Constants

Some useful constants are defined in `common.py`, including:

```python
# Volume shape.
N, M, P = 1320, 800, 1140

# Values used in the mask file
V_OUTSIDE = 0   # voxels outside of the surfaces and brain region
V_ST = 1        # top (outer) surface
V_VOLUME = 2    # volume between the two surfaces
V_SB = 3        # bottom (inter) surface
V_SE = 4        # intermediate surfaces
```


## How it works

This section describes the method for generating the streamlines and flatmaps.

The general methods consists of computing streamlines between a bottom and top surface around a brain region by solving Laplace's partial differential equation $\Delta u = 0$ inside the brain region, with a combination of Dirichlet and Neumann boundary conditions on the surfaces, and integrating the gradient field $\nabla u$ to link every voxel of the bottom surface to the top surface of the volume.

The streamlines allow one to generate flatmaps by mapping every pixel of the flattened surface to an average value along the streamline that starts from that voxel.

### Notations

We start by defining some notations.

#### General notations

- The 1-norm of a vector $\mathbf p=(x,y,z)$ is $\lVert\mathbf p\rVert_1 = |x|+|y|+|z|$.
- The Euclidean norm of a vector $\mathbf p=(x,y,z)$ is $\lVert\mathbf p\rVert_2 = \sqrt{x^2+y^2+z^2}$.
- The gradient of a scalar field $u$ is $\displaystyle\nabla u =\left(\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y}, \frac{\partial u}{\partial z}\right)$.
- The Laplacian of a scalar field $u$ is $\displaystyle\Delta u =\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}$.

#### Surfaces

- $\Omega = \left[0, N\right] \times \left[0, M\right] \times \left[0, P\right]$ is the 3D volume containing the brain atlas.
- $\mathcal V \subset \Omega$ is the **brain region** to flatten
- $\mathcal S = \partial\mathcal V \subset \Omega$ is the boundary surface of the volume
- $\mathcal S_T \subset \mathcal S$ is the top (outer) surface of the brain region $\mathcal V$
- $\mathcal S_B \subset \mathcal S$ is the bottom (inner) surface of the brain region $\mathcal V$
- $\mathcal S_E \subset \mathcal S$ is the edge surface of the brain region $\mathcal V$

The topological boundary of the volume is the union of these three non-intersecting surfaces:

$$\mathcal S = \partial\mathcal V = \mathcal S_T \sqcup \mathcal S_B \sqcup \mathcal S_E$$

#### Coordinate system

We use the Allen CCF coordinate system:

[![](http://help.brain-map.org/download/attachments/5308472/3DOrientation.png)](http://help.brain-map.org/display/mousebrain/API)

#### Voxels

- $p = (i, j, k) \in \Omega$ is a voxel in the volume
- $p_x^- = (i-1, j, k) \in \Omega$ is the neighbor voxel in front of $p$
- $p_x^+ = (i+1, j, k) \in \Omega$ is the neighbor voxel behind $p$
- $p_y^- = (i, j-1, k) \in \Omega$ is the neighbor voxel on top of $p$
- $p_y^+ = (i, j+1, k) \in \Omega$ is the neighbor voxel below $p$
- $p_z^- = (i, j, k-1) \in \Omega$ is the neighbor voxel to the left of $p$
- $p_z^+ = (i, j, k+1) \in \Omega$ is the neighbor voxel to the right of $p$

For each subset $\mathcal S \subset \Omega$, we define its indicator function $\chi_{\mathcal A}$ as:

$$
\forall p \in \Omega, \quad \chi_{\mathcal A}(p) = \begin{cases}
1 & \textrm{if} \quad p \in \mathcal A \\
0 & \textrm{otherwise}
\end{cases}
$$


#### Mask

The mask $\mu$ is defined as the function $\Omega \longrightarrow \\{ 0,1,2,3,4 \\}$ that maps every voxel of the volume $\Omega$ to:

$$
\forall p \in \Omega, \quad
\mu(p) = \begin{cases}
0 & \textrm{if} \quad p \not\in \mathcal V \cup \mathcal S \\
v_t = 1 & \textrm{if} \quad p \in \mathcal S_T \\
v = 2 & \textrm{if} \quad p \in \mathcal V \\
v_b = 3 & \textrm{if} \quad p \in \mathcal S_B \\
v_e = 4 & \textrm{if} \quad p \in \mathcal S_E \\
\end{cases}
$$

> **Implementation notes:** The mask $\mu$ is stored in `mask.npy` that is computed in the first step below, from the input nrrd files. This file is a 3D array with shape `(N, M, P)` and a data type `uint8`.


### Step 1. Surface normal

The first step is to estimate the normal to the surface for every voxel of each of the three surfaces. The normals will be used as boundary conditions when simulating the partial differential equation in Step 2.

#### Crude local estimatation

With the $\chi_{\mathcal V}$ indicator function of the brain region, and the neighbor voxels, we can make a first estimation of the surface normal at each voxel belonging to one of the surfaces:

$$
\forall p \in \mathcal S, \quad
\nu^0(p) =
\begin{pmatrix}
\chi_{\mathcal V}(p_x^+) - \chi_{\mathcal V}(p_x^-) \\
\chi_{\mathcal V}(p_y^+) - \chi_{\mathcal V}(p_y^-) \\
\chi_{\mathcal V}(p_z^+) - \chi_{\mathcal V}(p_z^-) \\
\end{pmatrix}
\in \\{ -1, 0, +1 \\}^3
$$

On each axis, the component of the vector $\nu^0(p)$ is +1 if the positive neighbor voxel on that axis belongs to the brain region $\mathcal V$ and the negative neighbor does not, or -1 if that's the reverse, or 0 if neither or both of these neighbors belong to the brain region.

TODO: screenshot of surface normal

#### Gaussian smoothing

Once this crude local estimate of the normal vectors to the surfaces is obtained, we can smooth it and normalize it to improve the accuracy of the boundary conditions in Step 2.

We define a Gaussian kernel as follows:

$$\forall q \in \mathbb R^3, \quad g_\sigma(q) = \lambda \exp \left(- \frac{\lVert q\rVert_2^2}{\sigma^2}\right) \quad \textrm{where $\lambda$ is defined such as} \quad \int g(q) dq=1.$$

We smooth the normal with a partial Gaussian convolution on the surface:

$$
\forall p \in \mathcal S, \quad
\widetilde \nu(p) = \frac{\displaystyle\int_{\mathcal S} \nu^0(q) g(p-q) dq}{\displaystyle\int_{\mathcal S} g(p-q)dq}
$$

Finally, we normalize the normal vectors:

$$
\forall p \in \mathcal S, \quad
\nu(p) = \begin{cases}
\displaystyle \frac{\widetilde \nu(p)}{\lVert \widetilde \nu(p) \rVert_2} & \textrm{if} \quad {\lVert \widetilde \nu(p) \rVert_2} > 0\\
0 & \textrm{otherwise}
\end{cases}
$$

> **Implementation notes:** this convolution is implemented with nested `for` loops in Python accelerated with JIT compilation using Numba.

TODO: screenshot of surface normal

### Step 2. Laplacian

Step 2 is the most complex and computationally intensive step of the process. It requires a GPU to be tractable on the 10 $\mu$m atlas.

Mathematically, the goal is to solve the following partial differential equation (PDE) with a mixture of Dirichlet and Neumann boundary conditions:

$$
\begin{align*}
\Delta u &= 0 & \textrm{on} \quad \mathcal V\\
u &= 0 & \textrm{on} \quad \mathcal S_T\\
\nabla u \cdot \nu &= 1 & \textrm{on} \quad \mathcal S_B\\
\nabla u \cdot \nu &= 0 & \textrm{on} \quad \mathcal S_E\\
\end{align*}
$$

#### Numerical scheme

An approximate solution of this equation can be obtained with an iterative numerical scheme.

We start from $u_0(p) = \chi_{\mathcal S_B}(p)$, equal to 1 on the bottom surface $\mathcal S_B$, and 0 elsewhere. Then, for $n \geq 0$, we iteratively apply the following scheme to estimate the Laplacian:

$$\forall p \in \mathcal V, \quad u^{n+1}(p) = \frac{u^n(p_x^+) + u^n(p_x^-) + u^n(p_y^+) + u^n(p_y^-) + u^n(p_z^+) + u^n(p_z^-)}{6}$$

We need to use another numerical scheme on the surface boundaries to take into account the boundary conditions, which allow us to compute $u^{n+1}(p)$ on the surfaces.

On $\mathcal S_T$, we just use $u^{n+1}(p) = 0$ for the Dirichlet boundary condition.

On $\mathcal S_B$ and $\mathcal S_E$, we need to implement the Neuman boundary conditions. We use central, forward, or backward finite difference schemes for $\nabla u(p)$ depending on the value of each $x$, $y$, $z$ component of the crude normal vector $\nu^0(p)$.

We note $k=1$ for $\mathcal S_B$, and $k=0$ for $\mathcal S_E$. We also define:

$$
\forall p \in \mathcal S, \quad
u_x^{n+1}(p)=
\begin{cases}
u^{n+1}(p_x^+) & \textrm{if} \quad \nu^0_x(p)=+1\\
u^{n+1}(p_x^-) & \textrm{if} \quad \nu^0_x(p)=-1\\
0 & \textrm{if} \quad \nu^0_x(p)=0\\
\end{cases}
$$

and similarly for the other components, $u_y^{n+1}$ and $u_z^{n+1}$.

Then, we find the following scheme for the Neumann boundary condition:

$$
\forall p \in \mathcal S_B \cup \mathcal S_E, \quad
u^{n+1}(p) =
\begin{cases}
\displaystyle\frac{u_x^{n+1}(p) \\, |\nu_x(p)| + u_y^{n+1}(p)  \\, |\nu_y(p)| + u_z^{n+1}(p)  \\, |\nu_z(p)| + k}{|\nu_x(p)| + |\nu_y(p)| + |\nu_z(p)| + k} & \textrm{if} \quad \lVert\nu^0(p)\rVert_1 \geq 1\\
0 & \textrm{otherwise}
\end{cases}
$$

#### GPU implementation

We wrote a GPU implementation with the Cupy Python package leveraging the NVIDIA CUDA API. There are a few tricks:

- We use two CUDA kernels: one for the numerical scheme in the brain region $\mathcal V$, another for the one on the surfaces $\mathcal S_B$ and $\mathcal S_E$ (Neumann conditions). Every iteration involves a call to both kernels.

- We use two 3D arrays for the Laplacian, `U_1` and `U_2`. The CUDA kernels use one array for reading the old values ($u^n$), another one for writing the new values ($u^{n+1}$). At the next iteration, we swap `U_1` and `U_2`.

- To avoid using too much GPU memory (there are wide empty spaces around a given brain region $\mathcal V$), we compute the axis boundaries of the mask array and we pad each side with a few voxels.

- To ensure all arrays fit in memory, we cut the brain in half (two hemispheres), which is possible as long as the streamlines are not expected to cross the sagittal midline within the brain region.

- We achieve about 1000 iterations per minute on an NVIDIA Geforce RTX 2070 SUPER.

- Empirically, a total of 10,000 iterations seems to be necessary for proper convergence of the algorithm.

TODO: screenshot of laplacian


### Step 3. Gradient

Once the solution of Laplace's equation has been obtained, we can estimate its gradient that will be used to integrate the streamlines in Step 4.

We use central, forward, or backward differences for the numerical scheme of the derivative of $u$ depending on whether the voxel is inside the volume or on the surface, and depending on the relative position of the voxel compared to the volume (which is encoded in $\nu^0(p)$).

We get:

$$
\forall p \in \mathcal V \cup \mathcal S, \quad
\widetilde{\nabla u}_x(p) =
\begin{cases}
\displaystyle
\frac{u_x(p_x^+) + u_x(p_x^-)}{2} & \textrm{if} \quad p \in \mathcal V\\
u_x(p_x^+) - u_x(p) & \textrm{if} \quad p \in \mathcal S, \\, \nu^0(p)=+1\\
u_x(p) - u_x(p_x^-) & \textrm{if} \quad p \in \mathcal S, \\, \nu^0(p)=-1\\
0 & \textrm{if} \quad p \in \mathcal S, \\, \nu^0(p)=0\\
\end{cases}
$$

and similarly for $\widetilde{\nabla u}_y(p)$ and $\widetilde{\nabla u}_z(p)$.

Finally, we normalize the gradient:

$$
\forall p \in \mathcal V \cup \mathcal S, \quad
\nabla u(p) = \begin{cases}
\displaystyle \frac{\widetilde{\nabla u}(p)}{\lVert \widetilde{\nabla u}(p) \rVert_2} & \textrm{if} \quad {\lVert \widetilde{\nabla u}(p) \rVert_2} > 0\\
0 & \textrm{otherwise}
\end{cases}
$$

### Step 4. Streamlines

To compute streamlines, we start from voxels in the bottom surface and we integrate the Laplace's equation's solution's gradient.

More precisely, we solve the following ordinary differential equation (ODE):

$$
\forall p \in \mathcal S_B, \quad \phi_p : \mathbb R_+ \longrightarrow \Omega
$$

satisfies:

$$
\forall t \geq 0, \forall p \in \mathcal S_B, \quad
\phi'_p(t) = \nabla u \left( \phi_p(t) \right)
$$

with initial conditions:

$$
\forall p \in \mathcal S_B, \quad
\begin{cases}
\phi_p(0) &= p\\
\phi'_p(0) &= \nabla u(p)\\
\end{cases}
$$

#### Numerical integration

We use the forward Euler method to integrate this ODE numerically.

At every time step, we use a linear interpolation to estimate the gradient at a position between voxels.

We also stop the integration for streamlines that go beyond the volume $\mathcal V$.

Finally, once obtained, we resample the streamlines to reparametrize them in 100 steps.

TODO: screenshot of 2D and 3D streamlines

### Step 5. Flatmaps

TO DO.


## References

Some references:

- Jones, S. E., Buchbinder, B. R., & Aharon, I. (2000). Three‐dimensional mapping of cortical thickness using Laplace's equation. Human brain mapping, 11(1), 12-32.
- Lerch, J. P., Carroll, J. B., Dorr, A., Spring, S., Evans, A. C., Hayden, M. R., ... & Henkelman, R. M. (2008). Cortical thickness measured from MRI in the YAC128 mouse model of Huntington's disease. Neuroimage, 41(2), 243-251.

Other implementations:

- https://github.com/KimLabResearch/CorticalFlatMap
- https://github.com/AllenInstitute/cortical_coordinates
