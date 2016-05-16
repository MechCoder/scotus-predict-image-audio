This heavily borrows from the IPython notebooks of the work of Katz et al. listed here
https://github.com/mjbommar/scotus-predict/e

There are a couple of minor modifications to the script.

* Remove the targets corresponding to the labels marked as "-1" or from which we could
  not predict the outcome from data. 
* Fix the random state of the different sklearn models used.
