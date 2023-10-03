import unittest
import numpy as np
from distclassipy import classifier

class TestClassifier(unittest.TestCase):
    """
    Unit test for the Classifier class.
    """
    def setUp(self):
        """
        Set up the test case.
        """
        self.clf = classifier.Classifier()

    def test_fit(self):
        """
        Test the fit method of the Classifier class.
        """
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        self.clf.fit(X, y)
        self.assertEqual(self.clf.X_.tolist(), X.tolist())
        self.assertEqual(self.clf.y_.tolist(), y.tolist())

    def test_predict(self):
        """
        Test the predict method of the Classifier class.
        """
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        self.clf.fit(X, y)
        predictions = self.clf.predict(X)
        self.assertEqual(predictions.tolist(), y.tolist())

    def test_predict_and_analyse(self):
        """
        Test the predict_and_analyse method of the Classifier class.
        """
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        self.clf.fit(X, y)
        analysis = self.clf.predict_and_analyse(X)
        self.assertIsInstance(analysis, dict)

    def test_calculate_confidence(self):
        """
        Test the calculate_confidence method of the Classifier class.
        """
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        self.clf.fit(X, y)
        confidence = self.clf.calculate_confidence(X)
        self.assertIsInstance(confidence, np.ndarray)

if __name__ == '__main__':
    """
    Main entry point for the test module.
    """
    unittest.main()

