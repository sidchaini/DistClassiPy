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
clf = dcpy.DistanceMetricClassifier(metric="canberra")
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0]]))
```

## Features
- **Distance Metric-Based Classification**: Utilizes a variety of distance metrics for classification.
- **Customizable for Scientific Goals**: Allows fine-tuning based on scientific objectives by selecting appropriate distance metrics and features, enhancing both computational efficiency and model performance.
- **Interpretable Results**: Offers improved interpretability of classification outcomes by directly using distance metrics and feature importance, making it ideal for scientific applications.
- **Efficient and Scalable**: Demonstrates lower computational requirements compared to traditional methods like Random Forests, making it suitable for large datasets
- **Open Source and Accessible**: Available as an open-source Python package on PyPI, encouraging broad application in astronomy and beyond

## Documentation

For more detailed information about the package and its functionalities, please refer to the [official documentation](https://sidchaini.github.io/DistClassiPy/).

## Contributing
Contributions are welcome! If you have suggestions for improvements or bug fixes, please feel free to open an [issue](https://github.com/sidchaini/DistClassiPy/issues) or submit a [pull request](https://github.com/sidchaini/DistClassiPy/pulls).

## License
DistClassiPy is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). See the LICENSE file for more details.

## Citation

If you use DistClassiPy in your research or project, please consider citing the paper:
> Chaini, S., Mahabal, A., Kembhavi, A., & Bianco, F. B. (2024). Light Curve Classification with DistClassiPy: a new distance-based classifier. arXiv. https://doi.org/10.48550/arXiv.2403.12120

### Bibtex


```bibtex
@ARTICLE{chaini2024light,
       author = {{Chaini}, Siddharth and {Mahabal}, Ashish and {Kembhavi}, Ajit and {Bianco}, Federica B.},
       title = "{Light Curve Classification with DistClassiPy: a new distance-based classifier}",
       journal = {arXiv e-prints},
       keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics, Computer Science - Machine Learning},
       year = 2024,
       month = mar,
       eid = {arXiv:2403.12120},
       pages = {arXiv:2403.12120},
       archivePrefix = {arXiv},
       eprint = {2403.12120},
       primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240312120C},
       adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
  

<!-- You can also find citation information in the [CITATION.cff](https://github.com/sidchaini/DistClassiPy/CITATION.cff) file. -->


## Authors
Siddharth Chaini, Ashish Mahabal, Ajit Kembhavi and Federica B. Bianco.
