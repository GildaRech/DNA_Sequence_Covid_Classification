# DNA_Sequence_Covid_Classification
This work is a sequence classification, we predict whether a DNA sequence belongs to the SARS-CoV-2 (Covid-19).

<ol> 
<li>**Data description** </li>
<ul>
<li> Xtr.csv - the training sequences.</li>
<li> Xte.csv - the test sequences.</li>
<li> Ytr.csv - the sequence labels of the training sequences indicating
Covid-19 DNA (1) or not (0).</li>
<li> Xtr_vectors- the training vectors which contain the encoding vector
of each DNA sequencing reads from training.</li>
<li> Xte_vectors- the testing vectors which contain the encoding vector of
each DNA sequencing reads from testing.</li>
<li> Each row of Xtr.csv represents a sequence.</li>
<li>Xte.csv contains 1000
test sequences, for which you need to predict the labels.</li>
<li>Ytr.csv
contains the labels corresponding to the training data, in the same
format as a submission file.
</ul>
<br>
<li> **Pre-processing**</li>
For features use the vectors dataset which are already encoded
but still we need to convert the values 0 to -1. Therefore transforming
 {0,1} classification problem to {-1,1} classification problem.
To do that we computed : $y=2y-1$
<br><br>
<li>**Models**</li>
We used many models to make classification like Logistic Ridge
Regression (Newton-Ralphson ,SGD), Kernel Logistic Regression,
Kernel Ridge Regression, Kernel SVM and HardMarginSVM.
We got the best accuracy with using Kernel SVM using RBF kernel.
Our model performs an accuracy of **100%** for training and **97%** for validation.
We got this result by looking for the good hyper-parameters
<ul>
<li>kernel = 'rbf'</li>
<li>sigma =0.1407035175879397 </li>
<li>degree = 2 </li>
<li>C = 10.0 </li>
<li>tol = 1e-4 </li>
</ul>
</ol>
