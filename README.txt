File names: 

    "Section_Subsection_Subsubsection.py"

    For example, G_1_1.py corresponds to experiments in Appendix G.1.1

    Note that the theoretical coverage probabilities obtained in Section 8.1 can also be used in Section 8.2 by proper scaling. Therefore, there is no separate code for Section 8.2.

Input variables:

    dim: dimension of each sample
    function target(x): the function \psi(P) when P is given by the empirical distribution of x
    u,v,w: 1st, 2nd, and 3rd order gradients (see input of Algorithm 1 in the paper)
    Sigma, gamma, kappa: 2nd, 3rd, 4th cumulants (see input of Algorithm 1 in the paper)



Output variables:

    S_acturals: the empirical coverages of sectioning
    S_theos: the coverage of sectioning as implied by the first order expansion

    Similarly when "S" above is replaced by B, SJ, or SB which corresponds to batching, sectioned jackknife, and sectioning-batching respectively.