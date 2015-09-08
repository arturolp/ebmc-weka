# EBMC package for WEKA
A java WEKA extension of the Efficient Bayesian Multivariate Classifier (EBMC) algorithm

## Description
EBMC builds a tree-augmented naïve Bayes model (TAN). EBMC greedily searches over the subspace of Bayesian networks that best predict the target node. To make the search efficient, it initially starts with an empty model and it identifies the set of nodes that are independent parents of the target and predicts it well. Then, EBMC transforms the temporary network into its statistically equivalent network where the parent nodes become children of the target with arcs between them. It then iterates the search for a new set of parents given the current structure. Finally, it greedily eliminates arcs between the children nodes.

## Citation
For citation and more information refer to:

>G. F. Cooper, P. Hennings-Yeomans, S. Visweswaran, & M. Barmada, (2010). An efficient Bayesian method for predicting clinical outcomes from genome-wide data. AMIA Annual Symposium Proceedings. 127-131. http://www.ncbi.nlm.nih.gov/pubmed/21346954

For pseudocode of the algorithm, refer to the supplementary material from:

>Jiang, X., Cai, B., Xue, D., Lu, X., Cooper, G. F., & Neapolitan, R. E. (2014). A comparative analysis of methods for predicting clinical outcomes using high-dimensional genomic datasets. Journal of the American Medical Informatics Association, 21(e2), e312–e319. http://doi.org/10.1136/amiajnl-2013-002358

## Current development status
This package is an un-official weka package that can be installed using Weka's Package Manager
