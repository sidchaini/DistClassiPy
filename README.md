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

## Contributing
Contributions are welcome! Please check our [GitHub issues](https://github.com/sidchaini/DistClassiPy/issues) for ways to contribute or open a pull request.

## License
DistClassiPy is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). See the LICENSE file for more details.

## Contact
For any queries, please reach out to Siddharth Chaini at sidchaini@gmail.com.
