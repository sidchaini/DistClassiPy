<h1 align="center">
<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sidchaini/DistClassiPy/main/docs/_static/logo-dark.svg" width="300">
  <img alt="DistClassiPy Logo" src="https://raw.githubusercontent.com/sidchaini/DistClassiPy/main/docs/_static/logo.svg" width="300">
</picture>
</h1>
<br>

[![PyPI](https://img.shields.io/pypi/v/distclassipy?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/distclassipy/)
[![Installs](https://img.shields.io/pypi/dm/distclassipy.svg?label=PyPI%20downloads)](https://pypi.org/project/distclassipy/)
[![Codecov](https://codecov.io/gh/sidchaini/distclassipy/branch/main/graph/badge.svg)](https://codecov.io/gh/sidchaini/distclassipy)
[![License - GPL-3](https://img.shields.io/pypi/l/distclassipy.svg)](https://github.com/sidchaini/distclassipy/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![arXiv](https://img.shields.io/badge/arXiv-astro--ph%2F2403.12120-red)](https://arxiv.org/abs/2403.12120) 
[![ascl:2403.002](https://img.shields.io/badge/ascl-2403.002-blue.svg?colorB=262255)](https://ascl.net/2403.002)

<!-- [![Paper](https://img.shields.io/badge/DOI-10.1038%2Fs41586--020--2649--2-blue)](
https://doi.org/10.1038/s41586-020-2649-2) -->

A python package for a distance-based classifier which can use several different distance metrics.

## Installation
To install DistClassiPy, run the following command:
```bash
pip install distclassipy
```

## Usage
Here's a quick example to get you started with DistClassiPy:
```python
import distclassipy as dcpy
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)
# Example usage of DistanceMetricClassifier
clf = dcpy.DistanceMetricClassifier()
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0]], metric="canberra"))

# Example usage of EnsembleDistanceClassifier
ensemble_clf = dcpy.EnsembleDistanceClassifier(feat_idx=0)
ensemble_clf.fit(X, y)
print(ensemble_clf.predict(X))
```

## Features
- **Distance Metric-Based Classification**: Utilizes a variety of distance metrics for classification.
- **Customizable for Scientific Goals**: Allows fine-tuning based on scientific objectives by selecting appropriate distance metrics and features, enhancing both computational efficiency and model performance.
- **Interpretable Results**: Offers improved interpretability of classification outcomes by directly using distance metrics and feature importance, making it ideal for scientific applications.
- **Efficient and Scalable**: Demonstrates lower computational requirements compared to traditional methods like Random Forests, making it suitable for large datasets.
- **Open Source and Accessible**: Available as an open-source Python package on PyPI, encouraging broad application in astronomy and beyond.
- **(NEW) Ensemble Distance Classification**: Leverages an ensemble approach to use different distance metrics for each quantile, improving classification performance across diverse data distributions.
- **(NEW) Expanded Distance Metrics**: DistClassiPy now offers 43 built-in distance metrics, an increase from the previous 18. Additionally, users can still define and use custom distance metrics as needed.

## Documentation

For more detailed information about the package and its functionalities, please refer to the [official documentation](https://sidchaini.github.io/DistClassiPy/).

## Contributing
Contributions are welcome! If you have suggestions for improvements or bug fixes, please feel free to open an [issue](https://github.com/sidchaini/DistClassiPy/issues) or submit a [pull request](https://github.com/sidchaini/DistClassiPy/pulls).

## License
DistClassiPy is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). See the LICENSE file for more details.

## Citation

If you use DistClassiPy in your research or project, please consider citing the paper:
> Chaini, S., Mahabal, A., Kembhavi, A., & Bianco, F. B. (2024). Light Curve Classification with DistClassiPy: a new distance-based classifier. Astronomy and Computing. https://doi.org/10.1016/j.ascom.2024.100850.

### Bibtex


```bibtex
@ARTICLE{2024A&C....4800850C,
       author = {{Chaini}, S. and {Mahabal}, A. and {Kembhavi}, A. and {Bianco}, F.~B.},
        title = "{Light curve classification with DistClassiPy: A new distance-based classifier}",
      journal = {Astronomy and Computing},
     keywords = {Variable stars (1761), Astronomy data analysis (1858), Open source software (1866), Astrostatistics (1882), Classification (1907), Light curve classification (1954), Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics, Computer Science - Machine Learning},
         year = 2024,
        month = jul,
       volume = {48},
          eid = {100850},
        pages = {100850},
          doi = {10.1016/j.ascom.2024.100850},
archivePrefix = {arXiv},
       eprint = {2403.12120},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024A&C....4800850C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```
  

<!-- You can also find citation information in the [CITATION.cff](https://github.com/sidchaini/DistClassiPy/CITATION.cff) file. -->


## Authors
Siddharth Chaini, Ashish Mahabal, Ajit Kembhavi and Federica B. Bianco.
