# DistClassiPy
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

If you use DistClassiPy in your research or project, please consider citing the package. You can find citation information in the [CITATION.cff](https://github.com/sidchaini/DistClassiPy/CITATION.cff) file.


## Contact
For any queries, please reach out to Siddharth Chaini at sidchaini@gmail.com.
