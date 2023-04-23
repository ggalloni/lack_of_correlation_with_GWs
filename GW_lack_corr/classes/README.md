# Classes

Here are stored the two main classes of the package: `Settings` and `State`. The former
contains all the settings of the analysis, regardless of the particular run. The latter
contains the state of the pipeline while the analysis is going. In this way, I can
carry the results from one script to another, without having to recompute them, or
redifine quantities, etc.