import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

actual = numpy.array([2, 1, 4, 1, 2, 1])
predicted = numpy.array([2, 1, 4, 1, 2, 1])

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

cm_display.plot()
plt.show()