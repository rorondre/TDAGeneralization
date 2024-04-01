# plot pipeline

Assume the results are in the folder FOLDER (e.g. final_results/2024-03-21)

 - 1. python analysis/merge_json.py FOLDER
 - 2. python analysis/plot_E_alpha.py FOLDER/all_results.json

The plots are saved in FOLDER/figures



Takeout of first results:
 - setup works, even with VIT
 - we have pseudo quantities, BUT only with small proportion, TODO: increase it, change eval procedure, ie use the eval function with a proportion (if proportion . 0.1, eval on tensors may have memory issues)
 - ph_dim is not working so far, magnitude and $E_\alpha$ are promising but are we observing really what we want.

TODO
 - 0. Fix PH dim
 - 1. what is said above
 - 2. non-pseudo quantities: in particular be able to freeze layers
 - 3. new plots: plot mag in terms of t (Rayna), and $E_\alpha$ wrt $\alpha$, kendall   (Ben)
 - 4. Curvature
 - 5. understand how to set t and $\alpha$
 - 6. In the future: other models and datasets
