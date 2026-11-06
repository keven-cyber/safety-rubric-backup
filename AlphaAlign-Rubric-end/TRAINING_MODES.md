# Training modes

The maintained comparison configuration uses `p1_earlytarget`, selected before the cross-model experiments. Run it through `scripts/train_rubricevolve_agentic_eight.sh p1_earlytarget` or select a model/method using `scripts/train_cross_model_twelve_p1.sh`.

The native launchers distinguish fresh training, checkpoint continuation and evaluation-only recovery. Do not equate a new fresh run with checkpoint restoration. Configure dataset/workspace/model paths explicitly. The code preserves original validation evidence and checkpoint selection; see the root README for current settings.

Historical wrapper variants that are dependencies of the retained launch chain remain available. Unreferenced backup modules and unrelated benchmark tasks are omitted from this distribution.
