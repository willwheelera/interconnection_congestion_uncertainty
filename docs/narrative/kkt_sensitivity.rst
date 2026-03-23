KKT-Based LMP Sensitivity Analysis
====================================

This page gives a self-contained mathematical derivation of the method
implemented in :mod:`sensitivity_analysis`, explains why the same approach
returns zero for linear-cost (LP) problems, and shows how the code maps onto
each step of the derivation.

.. contents:: Contents
   :local:
   :depth: 2

----

DC Optimal Power Flow
---------------------

Problem Formulation
~~~~~~~~~~~~~~~~~~~

The DC optimal power flow (DCOPF) solved by EGRET takes the form

.. math::

   \min_{\theta, p_g, f^{\text{DC}}} \quad
       & \sum_{g \in \mathcal{G}} c_g(p_g)
   \\
   \text{s.t.} \quad
       & \sum_{g: \text{bus}(g)=k} p_g
         + \sum_{l: \text{to}(l)=k} f^{\text{DC}}_l
         - \sum_{l: \text{from}(l)=k} f^{\text{DC}}_l
         - \sum_{l \in \mathcal{L}_k} B_l (\theta_k - \theta_{\text{other}(l,k)})
         = d_k
         \quad \forall k \in \mathcal{N}
         \tag{PB}
   \\
       & \theta_{\text{ref}} = 0
         \tag{REF}
   \\
       & -F_l \le B_l (\theta_{\text{from}(l)} - \theta_{\text{to}(l)}) \le F_l
         \quad \forall l \in \mathcal{L}^{\text{AC}}
         \tag{TH}
   \\
       & p_g^{\min} \le p_g \le p_g^{\max}
         \quad \forall g \in \mathcal{G}
         \tag{CAP}
   \\
       & -F^{\text{DC}}_l \le f^{\text{DC}}_l \le F^{\text{DC}}_l
         \quad \forall l \in \mathcal{L}^{\text{DC}}
         \tag{DCL}

Here:

- :math:`\theta_k` — voltage angle at bus :math:`k` (radians)
- :math:`p_g` — real power output of generator :math:`g` (MW)
- :math:`f^{\text{DC}}_l` — real power flow on DC link :math:`l` (MW)
- :math:`B_l = -S_{\text{base}} / x_l` — line susceptance (MW/rad), where
  :math:`x_l` is the per-unit reactance and :math:`S_{\text{base}}` is the
  system MVA base
- :math:`d_k` — real power demand at bus :math:`k` (MW)
- :math:`F_l, F^{\text{DC}}_l` — thermal / DC rating (MW)

For a **quadratic-cost** (QP) problem the generator cost function is

.. math::

   c_g(p_g) = a_{g,2}\, p_g^2 + a_{g,1}\, p_g + a_{g,0}

For a **linear-cost** (LP) problem :math:`a_{g,2} = 0` for all :math:`g`.

Locational Marginal Prices
~~~~~~~~~~~~~~~~~~~~~~~~~~

The **LMP** at bus :math:`k` is the dual variable (shadow price, Lagrange
multiplier) :math:`\lambda_k` associated with the power-balance constraint
**(PB)** at that bus.  It represents the marginal cost, in $/MWh, of supplying
one additional MW of load at bus :math:`k`.

After solving the DCOPF, EGRET stores LMPs in the ``ModelData`` object:

.. code-block:: python

   lmp_k = md_sol.elements("bus")[k]["lmp"]   # $/MWh

----

KKT Conditions and the Bordered-Hessian System
-----------------------------------------------

Collecting the Primal Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stack all primal decision variables into a single vector :math:`z`:

.. math::

   z = \bigl(\underbrace{\theta_1,\dots,\theta_n}_{\text{angles}},\;
              \underbrace{p_{g_1},\dots,p_{g_m}}_{\text{generators}},\;
              \underbrace{f^{\text{DC}}_1,\dots,f^{\text{DC}}_d}_{\text{DC links}}\bigr)
   \in \mathbb{R}^{n_{\text{var}}}

with :math:`n_{\text{var}} = n + m + d`.  The code uses the ordering

.. code-block:: python

   # indices into z
   bus_idx  = {b: i for i, b in enumerate(bus_names)}   # 0 … n-1
   gen_idx  = {g: i for i, g in enumerate(gen_names)}   # n … n+m-1
   dc_idx   = {d: i for i, d in enumerate(dc_names)}    # n+m … n+m+d-1

Objective Hessian **H**
~~~~~~~~~~~~~~~~~~~~~~~

Only generators with quadratic costs contribute to the Hessian.  Because
:math:`c_g(p_g) = a_{g,2} p_g^2 + \cdots`, the second derivative with respect
to :math:`p_g` is :math:`2 a_{g,2}`.  Voltage angles and DC flows do not appear
in the objective, so their second derivatives are zero.

.. math::

   H = \operatorname{diag}(H_1, \dots, H_{n_{\text{var}}}), \qquad
   H_{n + \text{idx}(g)} = 2\, a_{g,2}, \quad
   \text{all other entries} = 0.

In code:

.. code-block:: python

   H_diag = np.zeros(n_vars)
   for g_name, g_data in gens_orig.items():
       cost = g_data.get("p_cost", {})
       if cost.get("cost_curve_type") == "polynomial":
           a2 = cost.get("values", {}).get("2", 0.0) or 0.0
           if a2 != 0.0:
               H_diag[n_bus + gen_idx[g_name]] = 2.0 * a2
               is_qp = True
   H = np.diag(H_diag)

Active-Constraint Jacobian **J**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**J** is the stacked Jacobian of all *active* (equality + binding inequality)
constraints evaluated at the optimal solution.  It has dimensions
:math:`n_{\text{active}} \times n_{\text{var}}`.

1. **Power-balance block** :math:`J_{\text{PB}} \in \mathbb{R}^{n \times n_{\text{var}}}` —
   always active (equality constraint).

   The power balance at bus :math:`k` is

   .. math::

      \sum_{g: \text{bus}(g)=k} p_g
      + \sum_{l: \text{to}(l)=k} f^{\text{DC}}_l
      - \sum_{l: \text{from}(l)=k} f^{\text{DC}}_l
      + \sum_{l \in \mathcal{L}_k} B_l (\theta_k - \theta_{\text{other}(l,k)})
      - d_k = 0

   Differentiating with respect to :math:`z`:

   .. math::

      \frac{\partial (\text{PB}_k)}{\partial \theta_j}
          &= \begin{cases}
               -\sum_{l \ni k} B_l & j = k \\
               +B_l & l \text{ connects } k \text{ and } j \\
               0 & \text{otherwise}
             \end{cases}
      \\
      \frac{\partial (\text{PB}_k)}{\partial p_g}
          &= \mathbf{1}[\text{bus}(g) = k]
      \\
      \frac{\partial (\text{PB}_k)}{\partial f^{\text{DC}}_l}
          &= \mathbf{1}[\text{to}(l) = k] - \mathbf{1}[\text{from}(l) = k]

   The :math:`\theta`-block of :math:`J_{\text{PB}}` is exactly the *DC bus
   susceptance matrix* :math:`B_{\text{bus}}`.

2. **Reference-angle block** :math:`J_{\text{ref}} \in \mathbb{R}^{1 \times n_{\text{var}}}` —
   single row with a 1 at the reference bus column.

3. **Binding thermal limits** :math:`J_{\text{TH}}` — one row per binding AC
   branch limit.  For a branch :math:`l` at its upper limit
   :math:`pf_l = +F_l`:

   .. math::

      \frac{\partial (pf_l)}{\partial \theta_k} =
          B_l \cdot (\mathbf{1}[k = \text{from}(l)] - \mathbf{1}[k = \text{to}(l)])

4. **Binding generator capacity constraints** :math:`J_{\text{cap}}` — one row per
   binding bound (``p_g = p_g^{\max}`` or ``p_g = p_g^{\min}``).  Only the
   column for that generator is nonzero (value +1).

5. **Binding DC-link limits** :math:`J_{\text{DC}}` — one row per binding DC
   rating, with ±1 in the DC-flow column.

Binding constraints are detected by **primal activity**, not by the dual value:

.. code-block:: python

   # Binding thermal limit (upper)
   if abs(pf_mw - rating) < tol * max(abs(rating), 1.0):
       row = np.zeros(n_vars)
       row[f_i], row[t_i] = B_l, -B_l
       ineq_rows.append(row)

Using the dual value would be unreliable because EGRET's LP solutions are often
degenerate (a constraint can be active with a zero dual variable).

The full Jacobian is assembled as

.. math::

   J = \begin{pmatrix} J_{\text{PB}} \\ J_{\text{ref}} \\ J_{\text{TH}} \\ J_{\text{cap}} \\ J_{\text{DC}} \end{pmatrix}
   \in \mathbb{R}^{n_{\text{active}} \times n_{\text{var}}}

The Bordered-Hessian (KKT Matrix) **K**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At a regular optimal point the **bordered Hessian** (or *KKT matrix*) is

.. math::

   K =
   \begin{pmatrix}
     H   & J^\top \\
     J   & 0
   \end{pmatrix}
   \in \mathbb{R}^{(n_{\text{var}} + n_{\text{active}}) \times (n_{\text{var}} + n_{\text{active}})}

For a strictly convex QP this matrix is nonsingular.

----

Deriving the Sensitivity Formula
----------------------------------

Perturbing a Power Injection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Imagine increasing the net power injection at bus :math:`j` by a small amount
:math:`\delta P_j`.  This modifies the right-hand side of the power-balance
equation at bus :math:`j`:

.. math::

   (\text{PB}_j): \quad \cdots = d_j - \delta P_j

Treating :math:`P_j` as a parameter and differentiating the KKT optimality
conditions with respect to :math:`P_j`, the primal and dual variables respond
as :math:`(z, \nu) \to (z + dz, \nu + d\nu)` where the perturbation satisfies

.. math::

   \begin{pmatrix} H & J^\top \\ J & 0 \end{pmatrix}
   \begin{pmatrix} dz/dP_j \\ d\nu/dP_j \end{pmatrix}
   =
   \begin{pmatrix} 0 \\ e_j \end{pmatrix}

Here :math:`e_j` is the standard basis vector that selects the :math:`j`-th
power-balance constraint (the row in :math:`J` corresponding to bus :math:`j`).

.. note::

   The right-hand side has zeros in the first :math:`n_{\text{var}}` positions
   because the objective is not perturbed — only the constraint RHS changes.

The **LMP sensitivity** matrix is read from the dual-variable block of the
solution:

.. math::

   \frac{\partial \lambda_i}{\partial P_j}
   = \bigl[d\nu / dP_j\bigr]_i, \qquad i = 1, \dots, n

where the dual vector :math:`\nu` begins with the :math:`n` power-balance
multipliers :math:`\lambda_1, \dots, \lambda_n`.

Solving All Columns Efficiently
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full :math:`n \times n` sensitivity matrix requires solving the KKT system
once per injection bus :math:`j`.  Because the coefficient matrix :math:`K` is
the same for all right-hand sides, a single LU factorisation suffices:

.. code-block:: python

   K_lu = scipy.linalg.lu_factor(K)          # O(N^3) once

   for j in range(n_bus):
       rhs = np.zeros(n_vars + n_active)
       rhs[n_vars + j] = 1.0                 # select bus j power-balance row
       d = scipy.linalg.lu_solve(K_lu, rhs)  # O(N^2) per bus
       sensitivity[:, j] = d[n_vars : n_vars + n_bus]

The column ``sensitivity[:, j]`` gives :math:`\partial \lambda_i / \partial P_j`
for all observation buses :math:`i`.

Units
~~~~~

:math:`\lambda_i` is in $/MWh, :math:`P_j` is in MW, so the sensitivity has
units **$/MWh per MW**.  The code never converts to per-unit internally — line
susceptances :math:`B_l` are already expressed in MW/rad (using
:math:`B_l = -S_{\text{base}} / x_{\text{pu}}`), so the power-balance
equations are natively in MW.

----

Why LP Sensitivity is Identically Zero
---------------------------------------

For a **linear-cost** problem, :math:`a_{g,2} = 0` for all generators, so
:math:`H = 0`.  The bordered Hessian becomes

.. math::

   K = \begin{pmatrix} 0 & J^\top \\ J & 0 \end{pmatrix}

This matrix is **singular**: it has a null space of dimension
:math:`n_{\text{var}} - n_{\text{active}}`.  Intuitively, any combination of
primal directions that lies in the null space of :math:`J` (feasible primal
perturbations) can be added freely without changing the objective or constraint
residuals, because there is no curvature to resist them.

As a consequence, within the current LP basis the dual variables (LMPs) do
**not** change when a parameter changes, provided the basis remains the same.
The piecewise-constant nature of LP duals is a well-known phenomenon: LMPs jump
only when a basis change occurs — i.e. when a generator enters or leaves its
capacity limit or a line becomes binding or non-binding.

The implementation detects this case, returns an all-zero sensitivity matrix,
and prints an explanatory message:

.. code-block:: python

   if not is_qp:
       return pd.DataFrame(np.zeros((n_bus, n_bus)), ...), False

To observe cross-basis sensitivity for LP problems, use the **finite-difference
method** (:mod:`finite_difference_sensitivity`) with a perturbation large
enough to trigger a basis change, or the barrier method described below.

----

Barrier-Regularised Sensitivity (LP and QP)
--------------------------------------------

Motivation
~~~~~~~~~~

The singularity of the LP bordered Hessian is caused by :math:`H = 0` leaving
the primal block with no curvature.  The **barrier approach** (inspired by
interior-point theory, see the IPOPT paper) adds a synthetic curvature term
:math:`Y` that encodes how "close" every inequality constraint is to its bound.
The resulting regularised system is nonsingular for both LP and QP and
converges to the hard-constraint bordered Hessian as the barrier parameter
:math:`\varepsilon \to 0`.

The Barrier Matrix Y
~~~~~~~~~~~~~~~~~~~~~

For each inequality constraint :math:`k` with slack :math:`s_k` and dual
:math:`z_k`, define a per-constraint stiffness weight

.. math::

   y_k = \frac{z_k + \varepsilon}{s_k + \varepsilon}

When a constraint is **active** (:math:`s_k \approx 0`, :math:`z_k > 0`),
:math:`y_k \approx z_k / \varepsilon` is large — the constraint contributes
strong curvature.  When a constraint is **inactive** (:math:`z_k \approx 0`,
:math:`s_k \gg \varepsilon`), :math:`y_k \approx \varepsilon / s_k` is small.

The contribution of constraint :math:`k` to the barrier matrix :math:`Y` is

.. math::

   Y \mathrel{+}= y_k \left(\frac{\partial s_k}{\partial x}\right)^\top
                        \left(\frac{\partial s_k}{\partial x}\right)

Because slacks are linear in the primal variables :math:`x = (\theta, p_g,
f^{\text{DC}})`, each contribution is a rank-1 (or diagonal) block:

* **Generator capacity** — :math:`s_k = p_g^{\max} - p_g` or :math:`p_g - p_g^{\min}`,
  so :math:`\partial s_k / \partial p_g = \pm 1` and :math:`Y` acquires a diagonal
  entry at position :math:`(p_g, p_g)`:

  .. math::

     Y_{p_g, p_g} \mathrel{+}= \frac{z_{\max} + \varepsilon}{s_{\max} + \varepsilon}
                              + \frac{z_{\min} + \varepsilon}{s_{\min} + \varepsilon}

* **Branch thermal** — :math:`s_k = F_l \mp B_l(\theta_{\text{from}} - \theta_{\text{to}})`,
  so :math:`\partial s_k / \partial \theta = \mp B_l (e_{\text{from}} - e_{\text{to}})`.
  The outer product gives a :math:`2 \times 2` block:

  .. math::

     Y_{\theta,\theta} \mathrel{+}=
       (y_{\text{upper}} + y_{\text{lower}})\, B_l^2
       \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}_{\text{from, to}}

* **DC branch bounds** — diagonal, analogous to generator capacity.

Dual values :math:`z_k` are recovered from KKT stationarity at the optimal
solution.

**Generator capacity** — from stationarity of the objective with respect to
:math:`p_g`:

.. math::

   z_{\max,g} &= \max\!\bigl(0,\; \lambda_{\text{bus}(g)} - c_g'(p_g)\bigr) \\
   z_{\min,g} &= \max\!\bigl(0,\; c_g'(p_g) - \lambda_{\text{bus}(g)}\bigr)

**Branch thermal** — EGRET's B-theta DCOPF introduces an explicit power-flow
variable :math:`f_l` with a flow-equality constraint

.. math::

   f_l = B_l (\theta_{\text{from}(l)} - \theta_{\text{to}(l)})
   \quad\text{(FEQ}_l\text{)}

with thermal bounds :math:`-F_l \le f_l \le F_l` enforced as variable bounds
on :math:`f_l`.  Let :math:`\nu_l` denote the dual of (FEQ\ :math:`_l`).
KKT stationarity for :math:`f_l` gives the *reduced cost*

.. math::

   \mathrm{rc}_l
   \;=\; (\lambda_{\text{from}(l)} - \lambda_{\text{to}(l)}) - \nu_l

The reduced cost is non-negative when the lower bound is active and
non-positive when the upper bound is active, so

.. math::

   z_{\text{lower},l} &= \max(0,\;\mathrm{rc}_l) \\
   z_{\text{upper},l} &= \max(0,\;-\mathrm{rc}_l)

:math:`\nu_l` is extracted directly from the solved Pyomo model:

.. code-block:: python

   nu_l = m.dual[m.eq_pf_branch[l]]   # flow-equality dual ($/MWh)

and passed into :func:`_pf_ylim` and :func:`_compute_mu_base` via
``nu_dict``.

Why ν_l Cannot Be Omitted
~~~~~~~~~~~~~~~~~~~~~~~~~~

A naive approximation sets :math:`\nu_l = 0`, giving

.. math::

   z_{\text{upper},l}^{\text{naive}} &= \max(0,\; \lambda_{\text{from}(l)} - \lambda_{\text{to}(l)}) \\
   z_{\text{lower},l}^{\text{naive}} &= \max(0,\; \lambda_{\text{to}(l)} - \lambda_{\text{from}(l)})

This approximation has two systematic errors in a meshed network:

1. **Wrong direction.** When branch :math:`l` is at its lower bound
   (reverse flow at its limit), the congestion is attributed to the upper
   bound if :math:`\lambda_{\text{from}} > \lambda_{\text{to}}`.  LMP
   differences reflect contributions from *all* congested branches via the
   PTDF matrix and do not uniquely identify which bound is active on a
   given branch.

2. **Wrong magnitude.** The true reduced cost is
   :math:`\mathrm{rc}_l = (\lambda_{\text{from}} - \lambda_{\text{to}}) - \nu_l`;
   omitting :math:`\nu_l` under-estimates (or over-estimates) :math:`z_k`
   by :math:`|\nu_l|`.

3. **False positives.** In a meshed network an uncongested branch can carry
   a non-zero LMP difference because congestion elsewhere propagates price
   signals through the network.  The naive formula assigns a non-zero
   :math:`z_k` to such branches, inflating their barrier weights.

Additionally, LP solvers (including Gurobi's simplex) can return a
*dual-degenerate* solution where the constraint dual is zero even when the
constraint is active, making it impossible to infer :math:`z_k` from the
raw solver dual of the named inequality constraint.

The Regularised Bordered System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With :math:`Y` replacing explicit active-constraint rows, only the equality
constraints (power balance + reference angle) appear in :math:`J`.  Small
diagonal regularisations :math:`D_w = d_w I` and :math:`D_c = d_c I` are
added for numerical stability:

.. math::

   K' =
   \begin{pmatrix}
     H + Y + D_w  & J^\top \\
     J            & -D_c
   \end{pmatrix}

The sensitivity is obtained by solving the same multi-RHS system as before:

.. math::

   K' \begin{pmatrix} \mathrm{d}x/\mathrm{d}P_j \\ \mathrm{d}\nu/\mathrm{d}P_j \end{pmatrix}
   = \begin{pmatrix} 0 \\ e_j \end{pmatrix}

with :math:`\mathrm{d}\lambda_i/\mathrm{d}P_j` extracted from the first
:math:`n` entries of :math:`\mathrm{d}\nu/\mathrm{d}P_j`.

Relation to the README derivation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The README (following the IPOPT paper) writes the sensitivity system as

.. math::

   M \, \mathrm{d}V/\mathrm{d}p = -\tfrac{\partial M}{\partial p} V

where :math:`M` is the full KKT matrix and :math:`V = (x, \lambda, \mu)`.
For our DCOPF the injection parameter :math:`p` enters only through the
right-hand side of the equality constraints (it does not appear inside
:math:`M`), so :math:`\partial M / \partial p = 0` and the system reduces to

.. math::

   M \begin{pmatrix} \mathrm{d}x \\ \mathrm{d}\lambda \\ \mathrm{d}\mu \end{pmatrix}
   = \begin{pmatrix} 0 \\ e_j \\ 0 \end{pmatrix}

Block-elimination of the inequality dual block :math:`\mathrm{d}\mu` via the
perturbed complementarity condition :math:`X Z = \varepsilon` yields the
2-block system above.

The README also proposes a formula for :math:`\mathrm{d}\mu_k/\mathrm{d}p`.
The correct result from the differentiated complementarity condition is

.. math::

   \frac{\mathrm{d}z_k}{\mathrm{d}p} = -y_k \frac{\partial s_k}{\partial x} \frac{\mathrm{d}x}{\mathrm{d}p}

which can be evaluated once :math:`\mathrm{d}x/\mathrm{d}p` is known.  The
formula quoted in the README differs from this and appears to contain an error;
since the primary quantity of interest is :math:`\mathrm{d}\lambda/\mathrm{d}p`
(LMP sensitivity), we use only the 2-block reduced system.

----

Summary of Variable Mapping
----------------------------

The table below maps each mathematical object to its code counterpart.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Symbol
     - Code variable
     - Description
   * - :math:`n`
     - ``n_bus``
     - Number of buses
   * - :math:`m`
     - ``n_gen``
     - Number of generators
   * - :math:`d`
     - ``n_dc``
     - Number of DC links
   * - :math:`n_{\text{var}}`
     - ``n_vars``
     - Total primal variables (:math:`n + m + d`)
   * - :math:`n_{\text{active}}`
     - ``n_active``
     - Number of active constraints (rows of :math:`J`)
   * - :math:`H`
     - ``H``
     - Objective Hessian (diagonal, :math:`n_{\text{var}} \times n_{\text{var}}`)
   * - :math:`J_{\text{PB}}`
     - ``J_pb``
     - Power-balance Jacobian (:math:`n \times n_{\text{var}}`)
   * - :math:`J_{\text{ref}}`
     - ``J_ref``
     - Reference-angle constraint (:math:`1 \times n_{\text{var}}`)
   * - :math:`B_l`
     - ``B_l``
     - Line susceptance, ``-base_mva / (tau * x_l)`` (MW/rad)
   * - :math:`K`
     - ``K``
     - Bordered Hessian (:math:`(n_{\text{var}}+n_{\text{active}}) \times \cdot`)
   * - :math:`e_j`
     - ``rhs[n_vars + j] = 1.0``
     - Unit vector selecting bus :math:`j`'s power-balance row
   * - :math:`\partial \lambda / \partial P`
     - ``sensitivity``
     - :math:`n \times n` result matrix ($/MWh per MW)

----

Numerical Verification
-----------------------

LMP sensitivity (dλ/dP)
~~~~~~~~~~~~~~~~~~~~~~~~~

For QP problems the KKT and finite-difference results agree to machine
precision when the perturbation :math:`\delta P` is small (finite-difference
error is :math:`O(\delta P)`).  The finite-difference script uses the default
perturbation

.. math::

   \delta P = \max\!\bigl(10\;\text{MW},\; 0.15 \times P_{\text{load,total}}\bigr)

which is intentionally large for LP problems (to cross basis boundaries) but
introduces :math:`O(1\%)` error for QP problems.  Use a smaller ``--delta``
when running a QP cross-check.

See :mod:`finite_difference_sensitivity` for details on the finite-difference
implementation, and :mod:`plot_sensitivity` for side-by-side comparison plots.

Barrier weight accuracy (z_k vs. solver reduced costs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script ``code/test_mu_consistency.py`` checks that the dual estimates
:math:`z_k` used as barrier weights match the true thermal-limit shadow
prices returned by Gurobi.

For each AC branch it computes:

* **Analytical** — :math:`z_{\text{upper},l}` and :math:`z_{\text{lower},l}`
  from :func:`_compute_mu_base` using the flow-equality duals ``nu_dict``.
* **Solver RC** — the reduced cost of the :math:`f_l` variable extracted via
  a Pyomo ``rc`` suffix.  Gurobi is run with ``Method=2, Crossover=0``
  (barrier without crossover) to avoid LP dual degeneracy.

The two should agree to floating-point precision because both use the same
solved DCOPF.  To run the test::

   uv run python code/test_mu_consistency.py
   uv run python code/test_mu_consistency.py --hour 18

Expected output (the only branch with a non-zero shadow price is shown):

.. code-block:: text

   Branch                 Ana upper    RC upper     Δ upper   Ana lower    RC lower     Δ lower   pf/rating
   A27                       0.0000      0.0000     +0.0000      0.6991      0.6991     -0.0000      -0.717
   Max |Δ|: 0.0000 $/MWh
   PASS

The test also demonstrates three key properties:

* Only branches at their thermal limit (``|pf/rating| ≈ 1``) have non-zero
  reduced costs.  Uncongested branches have zero RC regardless of their LMP
  difference — the naive formula without :math:`\nu_l` incorrectly assigns
  non-zero weights to them.
* The correct bound is identified: branch A27 is at its *lower* limit
  (``pf/rating ≈ −1``) and :math:`z_{\text{lower}}` is non-zero.
* The magnitude includes the :math:`\nu_l` correction: the shadow price
  (0.6991 $/MWh) differs from the bare LMP difference (0.497 $/MWh) by
  exactly :math:`|\nu_{A27}| = 0.202` $/MWh.
