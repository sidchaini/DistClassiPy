"""
A module providing a variety of distance metrics to calculate the distance between two points.

This module includes implementations of various distance metrics, including both common and less
common measures. It allows for the calculation of distances between data points in a vectorized
manner using numpy arrays.
A part of this code is based on the work of Andrzej Zielezinski, originally retrieved on 20 November 2022 from
https://github.com/aziele/statistical-distances/blob/04412b3155c59fc7238b3d8ecf6f3723ac5befff/distance.py, which was released via the GNU General Public License v3.0.

It was originally modified by Siddharth Chaini on 27 November 2022.

Notes
-----

    Modifications by Siddharth Chaini include the addition of the following distance measures:
        1. Meehl distance
        2. Sorensen distance
        3. Ruzicka distance
        4. Inner product distance
        5. Harmonic mean distance
        6. Fidelity
        7. Minimimum Symmetric Chi Squared
        8. Probabilistic Symmetric Chi Squared

    In addition, the following code was added to all functions for array conversion:
        u,v = np.asarray(u), np.asarray(v)
"""

import numpy as np


class Distance:

    def __init__(self, epsilon=None):
        """
        Initialize the Distance class with an optional epsilon value.

        Parameters
        ----------
        - epsilon: A small value to avoid division by zero errors.
        """
        self.epsilon = np.finfo(float).eps if not epsilon else epsilon

    def acc(self, u, v):
        """
        Calculate the average of Cityblock/Manhattan and Chebyshev distances.
        This function computes the ACC distance, also known as the Average distance, between two
        vectors u and v. It is the average of the Cityblock (or Manhattan) and Chebyshev distances.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The ACC distance between the two vectors.

        References
        ----------
            1. Krause EF (2012) Taxicab Geometry An Adventure in Non-Euclidean Geometry. Dover Publications.
            2. Sung-Hyuk C (2007) Comprehensive Survey on Distance/Similarity Measures between Probability
               Density Functions. International Journal of Mathematical Models and Methods in Applied Sciences.
               vol. 1(4), pp. 300-307.
        """
        return (self.cityblock(u, v) + self.chebyshev(u, v)) / 2

    def add_chisq(self, u, v):
        """
        Compute the Additive Symmetric Chi-square distance between two vectors.
        The Additive Symmetric Chi-square distance is a measure that can be used to compare two vectors.
        This function calculates it based on the input vectors u and v.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Additive Symmetric Chi-square distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C (2007) Comprehensive Survey on Distance/Similarity Measures between Probability
               Density Functions. International Journal of Mathematical Models and Methods in Applied Sciences.
               vol. 1(4), pp. 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        uvmult = u * v
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sum(np.where(uvmult != 0, ((u - v) ** 2 * (u + v)) / uvmult, 0))

    def bhattacharyya(self, u, v):
        """
        Calculate the Bhattacharyya distance between two vectors.

        Returns a distance value between 0 and 1.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Bhattacharyya distance between the two vectors.

        References
        ----------
            1. Bhattacharyya A (1947) On a measure of divergence between two
               statistical populations defined by probability distributions,
               Bull. Calcutta Math. Soc., 35, 99–109.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
            3. https://en.wikipedia.org/wiki/Bhattacharyya_distance
        """
        u, v = np.asarray(u), np.asarray(v)
        return -np.log(np.sum(np.sqrt(u * v)))

    def braycurtis(self, u, v):
        """
        Calculate the Bray-Curtis distance between two vectors.

        The Bray-Curtis distance is a measure of dissimilarity between two non-negative vectors,
        often used in ecology to measure the compositional dissimilarity between two sites based on counts
        of species at both sites. It is closely related to the Sørensen distance and is also known as
        Bray-Curtis dissimilarity.

        Notes
        -----
            When used for comparing two probability density functions (pdfs),
            the Bray-Curtis distance equals the Cityblock distance divided by 2.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Bray-Curtis distance between the two vectors.

        References
        ----------
            1. Bray JR, Curtis JT (1957) An ordination of the upland forest of
               southern Wisconsin. Ecological Monographs, 27, 325-349.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
            3. https://en.wikipedia.org/wiki/Bray–Curtis_dissimilarity
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.sum(np.abs(u - v)) / np.sum(np.abs(u + v))

    def canberra(self, u, v):
        """
        Calculate the Canberra distance between two vectors.

        The Canberra distance is a weighted version of the Manhattan distance, used in numerical analysis.

        Notes
        -----
            When `u[i]` and `v[i]` are 0 for given i, then the fraction 0/0 = 0
            is used in the calculation.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Canberra distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        with np.errstate(invalid="ignore"):
            return np.nansum(np.abs(u - v) / (np.abs(u) + np.abs(v)))

    def chebyshev(self, u, v):
        """
        Calculate the Chebyshev distance between two vectors.

        The Chebyshev distance is a metric defined on a vector space where the distance between two vectors
        is the greatest of their differences along any coordinate dimension.

        Synonyms:
            Chessboard distance
            King-move metric
            Maximum value distance
            Minimax approximation

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Chebyshev distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.amax(np.abs(u - v))

    def chebyshev_min(self, u, v):
        """
        Calculate the minimum value distance between two vectors.

        This measure represents a custom approach by Zielezinski to distance measurement, focusing on the minimum absolute difference.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The minimum value distance between the two vectors.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.amin(np.abs(u - v))

    def clark(self, u, v):
        """
        Calculate the Clark distance between two vectors.

        The Clark distance equals the square root of half of the divergence.

        Notes
        -----
            When `u[i]` and `v[i]` are 0 for given i, then the fraction 0/0 = 0
            is used in the calculation.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Clark distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(np.nansum(np.power(np.abs(u - v) / (u + v), 2)))

    def cosine(self, u, v):
        """
        Calculate the cosine distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The cosine distance between the two vectors.

        References
        ----------
            1. SciPy.
        """
        u, v = np.asarray(u), np.asarray(v)
        return 1 - np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))

    def correlation_pearson(self, u, v):
        """
        Calculate the Pearson correlation distance between two vectors.

        Returns a distance value between 0 and 2.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Pearson correlation distance between the two vectors.
        """

        u, v = np.asarray(u), np.asarray(v)
        r = np.ma.corrcoef(u, v)[0, 1]
        return 1.0 - r

    def czekanowski(self, u, v):
        """
        Calculate the Czekanowski distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Czekanowski distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.sum(np.abs(u - v)) / np.sum(u + v)

    def dice(self, u, v):
        """
        Calculate the Dice dissimilarity between two vectors.

        Synonyms:
            Sorensen distance

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Dice dissimilarity between the two vectors.

        References
        ----------
            1. Dice LR (1945) Measures of the amount of ecologic association
               between species. Ecology. 26, 297-302.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        u_v = u - v
        return np.dot(u_v, u_v) / (np.dot(u, u) + np.dot(v, v))

    def divergence(self, u, v):
        """
        Calculate the divergence between two vectors.

        Divergence equals squared Clark distance multiplied by 2.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The divergence between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        with np.errstate(invalid="ignore"):
            return 2 * np.nansum(np.power(u - v, 2) / np.power(u + v, 2))

    def euclidean(self, u, v):
        """
        Calculate the Euclidean distance between two vectors.

        The Euclidean distance is the "ordinary" straight-line distance between two points in Euclidean space.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Euclidean distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.linalg.norm(u - v)

    def fidelity(self, u, v):
        """
        Calculate the fidelity distance between two vectors.

        The fidelity distance measures the similarity between two probability distributions.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The fidelity distance between the two vectors.

        Notes
        -----
            Added by SC.
        """
        u, v = np.asarray(u), np.asarray(v)
        return 1 - (np.sum(np.sqrt(u * v)))

    def google(self, u, v):
        """
        Calculate the Normalized Google Distance (NGD) between two vectors.

        NGD is a measure of similarity derived from the number of hits returned by the Google search engine for a given set of keywords.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Normalized Google Distance between the two vectors.

        Notes
        -----
            When used for comparing two probability density functions (pdfs),
            Google distance equals half of Cityblock distance.

        References
        ----------
            1. Lee & Rashid (2008) Information Technology, ITSim 2008.
               doi:10.1109/ITSIM.2008.4631601.
        """
        u, v = np.asarray(u), np.asarray(v)
        x = float(np.sum(u))
        y = float(np.sum(v))
        summin = float(np.sum(np.minimum(u, v)))
        return (max([x, y]) - summin) / ((x + y) - min([x, y]))

    def gower(self, u, v):
        """
        Calculate the Gower distance between two vectors.

        The Gower distance equals the Cityblock distance divided by the vector length.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Gower distance between the two vectors.

        References
        ----------
            1. Gower JC. (1971) General Coefficient of Similarity
               and Some of Its Properties, Biometrics 27, 857-874.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.sum(np.abs(u - v)) / u.size

    #### NEEDS CHECKING ####
    # def harmonicmean(self, u, v):
    #     """
    #     Harmonic mean distance.
    #     Notes:
    #         Added by SC.
    #     """
    #     u,v = np.asarray(u), np.asarray(v)
    #     return 1 - 2.*np.sum(u*v/(u+v))
    #########

    def hellinger(self, u, v):
        """
        Calculate the Hellinger distance between two vectors.

        The Hellinger distance is a measure of similarity between two probability distributions.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Hellinger distance between the two vectors.

        Notes
        -----
            This implementation produces values two times larger than values
            obtained by Hellinger distance described in Wikipedia and also
            in https://gist.github.com/larsmans/3116927.

        References
        ----------
           1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
              Measures between Probability Density Functions. International
              Journal of Mathematical Models and Methods in Applied Sciences.
              1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.sqrt(2 * np.sum((np.sqrt(u) - np.sqrt(v)) ** 2))

    def inner(self, u, v):
        """
        Calculate the inner product distance between two vectors.

        The inner product distance is a measure of similarity between two vectors, based on their inner product.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The inner product distance between the two vectors.

        Notes
        -----
            Added by SC.
        """
        u, v = np.asarray(u), np.asarray(v)
        return 1 - np.dot(u, v)

    def jaccard(self, u, v):
        """
        Calculate the Jaccard distance between two vectors.

        The Jaccard distance measures dissimilarity between sample sets.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Jaccard distance between the two vectors.

        References
        ----------
           1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
              Measures between Probability Density Functions. International
              Journal of Mathematical Models and Methods in Applied Sciences.
              1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        uv = np.dot(u, v)
        return 1 - (uv / (np.dot(u, u) + np.dot(v, v) - uv))

    def jeffreys(self, u, v):
        """
        Calculate the Jeffreys divergence between two vectors.

        The Jeffreys divergence is a symmetric version of the Kullback-Leibler divergence.

        Parameters
        ----------
        - u, v: Input vectors between which the divergence is to be calculated.

        Returns
        -------
        - The Jeffreys divergence between the two vectors.

        References
        ----------
            1. Jeffreys H (1946) An Invariant Form for the Prior Probability
               in Estimation Problems. Proc.Roy.Soc.Lon., Ser. A 186, 453-461.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        # Add epsilon to zeros in vectors to avoid division
        # by 0 and/or log of 0. Alternatively, zeros in the
        # vectors could be ignored or masked (see below).
        # u = ma.masked_where(u == 0, u)
        # v = ma.masked_where(v == 0, u)
        u = np.where(u == 0, self.epsilon, u)
        v = np.where(v == 0, self.epsilon, v)
        return np.sum((u - v) * np.log(u / v))

    def jensenshannon_divergence(self, u, v):
        """
        Calculate the Jensen-Shannon divergence between two vectors.

        The Jensen-Shannon divergence is a symmetric and finite measure of similarity between two probability distributions.

        Parameters
        ----------
        - u, v: Input vectors between which the divergence is to be calculated.

        Returns
        -------
        - The Jensen-Shannon divergence between the two vectors.

        References
        ----------
            1. Lin J. (1991) Divergence measures based on the Shannon entropy.
               IEEE Transactions on Information Theory, 37(1):145–151.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        Comments:
            Equals Jensen difference in Sung-Hyuk (2007):
            u = np.where(u==0, self.epsilon, u)
            v = np.where(v==0, self.epsilon, v)
            el1 = (u * np.log(u) + v * np.log(v)) / 2
            el2 = (u + v)/2
            el3 = np.log(el2)
            return np.sum(el1 - el2 * el3)
        """
        u, v = np.asarray(u), np.asarray(v)
        u = np.where(u == 0, self.epsilon, u)
        v = np.where(v == 0, self.epsilon, v)
        dl = u * np.log(2 * u / (u + v))
        dr = v * np.log(2 * v / (u + v))
        return (np.sum(dl) + np.sum(dr)) / 2

    def jensen_difference(self, u, v):
        """
        Calculate the Jensen difference between two vectors.

        The Jensen difference is considered similar to the Jensen-Shannon divergence.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Jensen difference between the two vectors.

        Notes
        -----
            1. Equals half of Topsøe distance
            2. Equals squared jensenshannon_distance.


        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        u = np.where(u == 0, self.epsilon, u)
        v = np.where(v == 0, self.epsilon, v)
        el1 = (u * np.log(u) + v * np.log(v)) / 2
        el2 = (u + v) / 2
        return np.sum(el1 - el2 * np.log(el2))

    def k_divergence(self, u, v):
        """
        Calculate the K divergence between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the divergence is to be calculated.

        Returns
        -------
        - The K divergence between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        u = np.where(u == 0, self.epsilon, u)
        v = np.where(v == 0, self.epsilon, v)
        return np.sum(u * np.log(2 * u / (u + v)))

    def kl_divergence(self, u, v):
        """
        Calculate the Kullback-Leibler divergence between two vectors.

        The Kullback-Leibler divergence measures the difference between two probability distributions.

        Parameters
        ----------
        - u, v: Input vectors between which the divergence is to be calculated.

        Returns
        -------
        - The Kullback-Leibler divergence between the two vectors.

        References
        ----------
            1. Kullback S, Leibler RA (1951) On information and sufficiency.
               Ann. Math. Statist. 22:79–86
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        u = np.where(u == 0, self.epsilon, u)
        v = np.where(v == 0, self.epsilon, v)
        return np.sum(u * np.log(u / v))

    def kulczynski(self, u, v):
        """
        Calculate the Kulczynski distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Kulczynski distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.sum(np.abs(u - v)) / np.sum(np.minimum(u, v))

    def kumarjohnson(self, u, v):
        """
        Calculate the Kumar-Johnson distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Kumar-Johnson distance between the two vectors.

        References
        ----------
            1. Kumar P, Johnson A. (2005) On a symmetric divergence measure
               and information inequalities, Journal of Inequalities in pure
               and applied Mathematics. 6(3).
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        uvmult = u * v
        with np.errstate(divide="ignore", invalid="ignore"):
            numer = np.power(u**2 - v**2, 2)
            denom = 2 * np.power(uvmult, 3 / 2)
            return np.sum(np.where(uvmult != 0, numer / denom, 0))

    def lorentzian(self, u, v):
        """
        Calculate the Lorentzian distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Lorentzian distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.

        Notes
        -----
            One (1) is added to guarantee the non-negativity property and to
            eschew the log of zero.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.sum(np.log(np.abs(u - v) + 1))

    def cityblock(self, u, v):
        """
        Calculate the Cityblock (Manhattan) distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Cityblock distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.

        Synonyms:
            City block distance
            Manhattan distance
            Rectilinear distance
            Taxicab norm

        Notes
        -----
            Cityblock distance between two probability density functions
            (pdfs) equals:
            1. Non-intersection distance multiplied by 2.
            2. Gower distance multiplied by vector length.
            3. Bray-Curtis distance multiplied by 2.
            4. Google distance multiplied by 2.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.sum(np.abs(u - v))

    def marylandbridge(self, u, v):
        """
        Calculate the Maryland Bridge distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Maryland Bridge distance between the two vectors.

        References
        ----------
            1. Deza M, Deza E (2009) Encyclopedia of Distances.
               Springer-Verlag Berlin Heidelberg. 1-590.
        """
        u, v = np.asarray(u), np.asarray(v)
        uvdot = np.dot(u, v)
        return 1 - (uvdot / np.dot(u, u) + uvdot / np.dot(v, v)) / 2

    def matusita(self, u, v):
        """
        Calculate the Matusita distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Matusita distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.

        Notes
        -----
            Equals square root of Squared-chord distance.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.sqrt(np.sum((np.sqrt(u) - np.sqrt(v)) ** 2))

    def max_symmetric_chisq(self, u, v):
        """
        Calculate the maximum symmetric chi-square distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The maximum symmetric chi-square distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        return max(self.neyman_chisq(u, v), self.pearson_chisq(u, v))

    def min_symmetric_chisq(self, u, v):
        """
        Calculate the minimum symmetric chi-square distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The minimum symmetric chi-square distance between the two vectors.

        Notes
        -----
            Added by SC.
        """
        u, v = np.asarray(u), np.asarray(v)
        return min(self.neyman_chisq(u, v), self.pearson_chisq(u, v))

    def meehl(self, u, v):
        """
        Calculate the Meehl distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Meehl distance between the two vectors.

        Notes
        -----
            Added by SC.

        References
        ----------
            1. Deza M. and Deza E. (2013) Encyclopedia of Distances.
               Berlin, Heidelberg: Springer Berlin Heidelberg.
               https://doi.org/10.1007/978-3-642-30958-8.
        """
        u, v = np.asarray(u), np.asarray(v)

        xi = u[:-1]
        yi = v[:-1]
        xiplus1 = np.roll(u, 1)[:-1]
        yiplus1 = np.roll(v, 1)[:-1]

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nansum((xi - yi - xiplus1 + yiplus1) ** 2)

    def minkowski(self, u, v, p=2):
        """
        Calculate the Minkowski distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.
        - p: The order of the norm of the difference.

        Returns
        -------
        - The Minkowski distance between the two vectors.

        Notes
        -----
            When p goes to infinite, the Chebyshev distance is derived.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.linalg.norm(u - v, ord=p)

    def motyka(self, u, v):
        """
        Calculate the Motyka distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Motyka distance between the two vectors.

        Notes
        -----
            The distance between identical vectors is not equal to 0 but 0.5.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.sum(np.maximum(u, v)) / np.sum(u + v)

    def neyman_chisq(self, u, v):
        """
        Calculate the Neyman chi-square distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Neyman chi-square distance between the two vectors.

        References
        ----------
            1. Neyman J (1949) Contributions to the theory of the chi^2 test.
               In Proceedings of the First Berkley Symposium on Mathematical
               Statistics and Probability.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sum(np.where(u != 0, (u - v) ** 2 / u, 0))

    def nonintersection(self, u, v):
        """
        Calculate the Nonintersection distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Nonintersection distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        Notes
        -----
            When used for comparing two probability density functions (pdfs),
            Nonintersection distance equals half of Cityblock distance.
        """
        u, v = np.asarray(u), np.asarray(v)
        return 1 - np.sum(np.minimum(u, v))

    def pearson_chisq(self, u, v):
        """
        Calculate the Pearson chi-square divergence between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the divergence is to be calculated.

        Returns
        -------
        - The Pearson chi-square divergence between the two vectors.

        References
        ----------
            1. Pearson K. (1900) On the Criterion that a given system of
               deviations from the probable in the case of correlated system
               of variables is such that it can be reasonable supposed to have
               arisen from random sampling, Phil. Mag. 50, 157-172.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        Notes
        -----
            Pearson chi-square divergence is asymmetric.
        """
        u, v = np.asarray(u), np.asarray(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sum(np.where(v != 0, (u - v) ** 2 / v, 0))

    def penroseshape(self, u, v):
        """
        Calculate the Penrose shape distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Penrose shape distance between the two vectors.

        References
        ----------
            1. Deza M, Deza E (2009) Encyclopedia of Distances.
               Springer-Verlag Berlin Heidelberg. 1-590.
        """
        u, v = np.asarray(u), np.asarray(v)
        umu = np.mean(u)
        vmu = np.mean(v)
        return np.sqrt(np.sum(((u - umu) - (v - vmu)) ** 2))

    def prob_chisq(self, u, v):
        """
        Calculate the Probabilistic chi-square distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Probabilistic chi-square distance between the two vectors.

        Notes
        -----
            Added by SC.
        """
        u, v = np.asarray(u), np.asarray(v)
        uvsum = u + v
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2 * np.sum(np.where(uvsum != 0, (u - v) ** 2 / uvsum, 0))

    def ruzicka(self, u, v):
        """
        Calculate the Ruzicka distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Ruzicka distance between the two vectors.

        Notes
        -----
            Added by SC.
        """
        u, v = np.asarray(u), np.asarray(v)
        den = np.sum(np.maximum(u, v))

        return 1 - np.sum(np.minimum(u, v)) / den

    def sorensen(self, u, v):
        """
        Calculate the Sorensen distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Sorensen distance between the two vectors.

        Notes
        -----
            The Sorensen distance equals the Manhattan distance divided by the sum of the two vectors.

            Added by SC.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.sum(np.abs(u - v)) / np.sum(u + v)

    def soergel(self, u, v):
        """
        Calculate the Soergel distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Soergel distance between the two vectors.

        Notes
        -----
            Equals Tanimoto distance.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.sum(np.abs(u - v)) / np.sum(np.maximum(u, v))

    def squared_chisq(self, u, v):
        """
        Calculate the Squared chi-square distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Squared chi-square distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        uvsum = u + v
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sum(np.where(uvsum != 0, (u - v) ** 2 / uvsum, 0))

    def squaredchord(self, u, v):
        """
        Calculate the Squared-chord distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Squared-chord distance between the two vectors.

        References
        ----------
            1. Gavin DG et al. (2003) A statistical approach to evaluating
               distance metrics and analog assignments for pollen records.
               Quaternary Research 60:356–367.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        Notes
        -----
            Equals to squared Matusita distance.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.sum((np.sqrt(u) - np.sqrt(v)) ** 2)

    def squared_euclidean(self, u, v):
        """
        Calculate the Squared Euclidean distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Squared Euclidean distance between the two vectors.

        References
        ----------
            1. Gavin DG et al. (2003) A statistical approach to evaluating
               distance metrics and analog assignments for pollen records.
               Quaternary Research 60:356–367.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        Notes
        -----
            Equals to squared Euclidean distance.
        """
        u, v = np.asarray(u), np.asarray(v)
        return np.dot((u - v), (u - v))

    def taneja(self, u, v):
        """
        Calculate the Taneja distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Taneja distance between the two vectors.

        References
        ----------
            1. Taneja IJ. (1995), New Developments in Generalized Information
               Measures, Chapter in: Advances in Imaging and Electron Physics,
               Ed. P.W. Hawkes, 91, 37-135.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        u = np.where(u == 0, self.epsilon, u)
        v = np.where(v == 0, self.epsilon, v)
        uvsum = u + v
        return np.sum((uvsum / 2) * np.log(uvsum / (2 * np.sqrt(u * v))))

    def tanimoto(self, u, v):
        """
        Calculate the Tanimoto distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Tanimoto distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        Notes
        -----
            Equals Soergel distance.
        """
        u, v = np.asarray(u), np.asarray(v)
        # return np.sum(abs(u-v)) / np.sum(np.maximum(u, v))
        usum = np.sum(u)
        vsum = np.sum(v)
        minsum = np.sum(np.minimum(u, v))
        return (usum + vsum - 2 * minsum) / (usum + vsum - minsum)

    def topsoe(self, u, v):
        """
        Calculate the Topsøe distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Topsøe distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        Notes
        -----
            Equals two times Jensen-Shannon divergence.
        """
        u, v = np.asarray(u), np.asarray(v)
        u = np.where(u == 0, self.epsilon, u)
        v = np.where(v == 0, self.epsilon, v)
        dl = u * np.log(2 * u / (u + v))
        dr = v * np.log(2 * v / (u + v))
        return np.sum(dl + dr)

    def vicis_symmetric_chisq(self, u, v):
        """
        Calculate the Vicis Symmetric chi-square distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Vicis Symmetric chi-square distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307
        """
        u, v = np.asarray(u), np.asarray(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            u_v = (u - v) ** 2
            uvmin = np.minimum(u, v) ** 2
            return np.sum(np.where(uvmin != 0, u_v / uvmin, 0))

    def vicis_wave_hedges(self, u, v):
        """
        Calculate the Vicis-Wave Hedges distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Vicis-Wave Hedges distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u, v = np.asarray(u), np.asarray(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            u_v = abs(u - v)
            uvmin = np.minimum(u, v)
            return np.sum(np.where(uvmin != 0, u_v / uvmin, 0))

    def wave_hedges(self, u, v):
        """
        Calculate the Wave Hedges distance between two vectors.

        Parameters
        ----------
        - u, v: Input vectors between which the distance is to be calculated.

        Returns
        -------
        - The Wave Hedges distance between the two vectors.

        References
        ----------
            1. Sung-Hyuk C (2007) Comprehensive Survey on Distance/Similarity
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307
        """
        u, v = np.asarray(u), np.asarray(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            u_v = abs(u - v)
            uvmax = np.maximum(u, v)
            return np.sum(np.where(((u_v != 0) & (uvmax != 0)), u_v / uvmax, 0))
