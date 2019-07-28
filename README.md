
Automatic Induction of Neural Network Decision Tree Algorithms
==============================================================

This repository contains code for "Automatic Induction of Neural Network Decision Tree Algorithms". To appear in Computing Conference 2019. Advances in Intelligent Systems and Computing.

For latest update in this realm of research please see: https://github.com/chappers/TreeGrad

Preprint: 
* https://arxiv.org/abs/1811.10735

If you use this code please cite via:

```
@inproceedings{siu2018automatic,
  title={Automatic Induction of Neural Network Decision Tree Algorithms},
  author={Siu, Chapman},
  booktitle={2019 Computing Conference},
  year={2019},
  organization={Springer International Publishing}
}
```

Usage
-----

You will need to change number rounds to 10, and number of epochs to 20 for comparison with paper -

```sh
python auto_induction.py
```

In the repository, the definition of the layers can be found in `decision_tree.py`.
