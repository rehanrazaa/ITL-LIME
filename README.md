# ITL-LIME: Instance-Based Transfer Learning for Enhancing Local Explanations in Low-Resource Data Settings

A novel Instance-based Transfer Learning LIME framework (ITL-LIME) that enhances explanation fidelity and stability in dataconstrained environments. ITL-LIME introduces instance transfer learning into the LIME framework by leveraging relevant real instances from a related source domain to aid the explanation process
in the target domain. Specifically, it employ clustering to partition the source domain into clusters with representative prototypes. Instead of generating random perturbations, ITL-LIME method retrieves pertinent real source instances from the source cluster whose prototype is most similar to the target instance. 
These are then combined with the target instance’s neighboring real instances. To define a compact locality, ITL-LIME further construct a contrastive learning based encoder as a weighting mechanism to assign weights to the instances from the combined set based on their proximity to the target instance. 
Finally, these weighted source and target instances are used to train the surrogate model for explanation purposes. 


## SCARF Github


## Using the code
Have a look at the LICENSE.

## Citation
If you find our work helpful in your research, please cite it as:
