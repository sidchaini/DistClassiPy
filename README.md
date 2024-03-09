<h1 align="center">
<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo-dark.svg" width="300">
  <img alt="Pandas Logo" src="docs/_static/logo.svg" width="300">
</picture>
</h1>
<br>

[![PyPI](https://img.shields.io/pypi/v/distclassipy?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/distclassipy/)
[![Installs](https://img.shields.io/pypi/dm/distclassipy.svg?label=PyPI%20downloads)](https://pypi.org/project/distclassipy/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

clf = dcpy.DistanceMetricClassifier()
# Add your data and labels
clf.fit(data, labels)
# Predict new instances
predictions = clf.predict(new_data)
```

## Features
- Multiple distance metrics support
- Easy integration with existing data processing pipelines
- Efficient and scalable for large datasets

## Documentation

For more detailed information about the package and its functionalities, please refer to the [official documentation](https://sidchaini.github.io/DistClassiPy/).

## Contributing
Contributions are welcome! If you have suggestions for improvements or bug fixes, please feel free to open an [issue](https://github.com/sidchaini/DistClassiPy/issues) or submit a [pull request](https://github.com/sidchaini/DistClassiPy/pulls).

## License
DistClassiPy is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). See the LICENSE file for more details.

## Citation

If you use DistClassiPy in your research or project, please consider citing the paper:
> Light Curve Classification with DistClassiPy: a new distance-based classifier (submitted)

### Bibtex

```bibtex
@ARTICLE{Chaini2024,
       author = {{Chaini}, S. and {Mahabal}, A. and {Kembhavi}, A. and {Bianco}, F.~B.},
        title = "{Light Curve Classification with DistClassiPy: a new distance-based classifier}",
      journal = {Submitted to A&C},
    %  keywords = {},
         year = 2024,
      %   month = ,
      %  volume = {},
      %     eid = {},
      %   pages = {},
      %     doi = {},
      %  adsurl = {},
      % adsnote = {}
}
```

<!-- You can also find citation information in the [CITATION.cff](https://github.com/sidchaini/DistClassiPy/CITATION.cff) file. -->


## Authors
Siddharth Chaini (E-mail: sidchaini@gmail.com), Ashish Mahabal, Ajit Kembhavi and Federica B. Bianco.