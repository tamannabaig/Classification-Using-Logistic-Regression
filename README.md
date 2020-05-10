# Classification-Using-Logistic-Regression
### Problem Description and Data Set
This Project deals with identifying whether a fish is TigerFish0 or TigerFish1. This time you are to use a logistic regression approach.  The first line in the data file contains a single integer indicating how many sets of labelled data you have to work with. Each line after that contains three tab-separated entries. The first is a float representing the body length in centimeters, followed by a float representing the dorsal fin length in centimeters, then an integer identifying the fish as either TigerFish0 (with a 0) or TigerFish1 (with a 1).
### Assignment
Using what you have learned in class and the notes, develop (from scratch in Python, no using a Logistic Regression library function!) a Hypothesis function that will predict type of fish given unseen data. You are free to use any model variation and any testing or training approach we have discussed.

What to Turn In Via Canvas
A pdf file that includes:
• Problem Description <br />
• Description of your data set along with a plot of the data.<br />
• A description of your model and testing procedure, including<br />
o Initial values that you chose for your weights, alpha, and the initial value for J.<br />
o Final values for alpha, your weights, how many iterations your learning algorithm went through and your final value of J on your training set.<br />
o Include a plot of J (vertical axis) vs. number of iterations (horizontal axis).<br />
o If you did feature scaling, describe what you did.<br />
o Value of J on your test set.<br />
• A confusion matrix showing your results on the test set.<br />
• A description of your final results that includes accuracy, precision, recall and F1 values.<br />
• A comparison of your results with a logistic regression approach as compared to your previous k-nearest neighbor approach (including kNN confusion matrix, accuracy, precision, recall and F1 values.<br />

The python file that repeatedly prompts for the body length and dorsal fin length (in centimeters) of a candidate fish and prints out the type to the screen. The program should terminate when you enter zero for both values.
