# Prodigy + ScheduleFree
*Eliminating hyperparameters, one commit at a time.*

**Current status:** Experimental

## Installation
```
pip install prodigy-plus-schedule-free
```
If you'd like to try the v2.0.0 beta, use the following instead:
```
pip install prodigy-plus-schedule-free==2.0.0b1
```
Please note v2.0.0 **includes breaking changes**. Do not use it to resume training runs on older versions! For more details, [see the dev branch](https://github.com/LoganBooker/prodigy-plus-schedule-free/tree/dev).

## Usage
```python
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
optimizer = ProdigyPlusScheduleFree(model.parameters(), lr=1.0, betas=(0.9, 0.99), beta3=None, 
                                    weight_decay=0.0, weight_decay_by_lr=True, 
				    use_bias_correction=False, d0=1e-6, d_coef=1.0, 
				    prodigy_steps=0, use_speed=False, eps=1e-8, 
				    split_groups=True, split_groups_mean=True,
 				    factored=True, factored_fp32=True, fused_back_pass=False,
                                    use_stableadamw=True, use_muon_pp=False, use_cautious=False,
				    use_grams=False, use_adopt=False, use_focus=False,
                                    stochastic_rounding=True)
```

As with the reference implementation of schedule-free, a constant scheduler should be used, along with the appropriate
calls to `optimizer.train()` and `optimizer.eval()`. See the schedule-free documentation for more details: https://github.com/facebookresearch/schedule_free

## TLDR
The default settings should "just work", but there are a few configurations you can try to improve things.

### Gradient scaling/clipping
By default, the optimiser uses StableAdamW to scale parameter updates, which negates the need to use external gradient scaling or clipping. However, this can also hamper Prodigy's
ability to adapt the stepsize. While the optimiser includes internal logic to mostly mitigate this, you can set `use_stableadamw=False` and use external gradient clipping instead.

### Training multiple networks
Try setting `split_groups_mean=False` to force the optimiser to use per-group learning rates. If the model fails to learn, or learns too slowly, set `use_speed=True` as well. Finally,
you can use just `split_groups=False` by itself to revert to the default Prodigy behaviour of combined learning rate calculations.

### Turning off Prodigy
Earlier versions of the optimiser recommended setting `prodigy_steps` equal to 5-25% of your total step count, but this should not be necessary with recent updates. That said,
you can still use the setting to make sure the LR does not change after a certain step, and free any memory used by Prodigy for adapting the step size.

## Details
An optimiser based on Prodigy that includes schedule-free logic and much, much lower memory usage, the aim being to remove the need to set any hyperparameters. Of course,
that's never the case with any optimiser, but hopefully, this comes close!

Hyperparameters eliminated: Learning rate (Prodigy), LR scheduler (ScheduleFree), epsilon (Adam-atan2, optional, not enabled by default).

Based on code from:
* https://github.com/facebookresearch/schedule_free
* https://github.com/konstmish/prodigy

Incorporates improvements from these pull requests (credit to https://github.com/dxqbYD, https://github.com/sangoi-exe and https://github.com/nhamanasu):
* https://github.com/konstmish/prodigy/pull/23
* https://github.com/konstmish/prodigy/pull/22
* https://github.com/konstmish/prodigy/pull/20
* https://github.com/facebookresearch/schedule_free/pull/54

If you do use another scheduler, linear or cosine is preferred, as a restarting scheduler can confuse Prodigy's adaptation logic.

Leave `lr` set to 1 unless you encounter instability. Do not use with gradient clipping, as this can hamper the
ability for the optimiser to predict stepsizes. Gradient clipping/normalisation is already handled in the following configurations:

1) `use_stableadamw=True,eps=1e8` (or any reasonable positive epsilon. This is the default.)
2) `eps=None` (Adam-atan2, scale invariant. Will disable StableAdamW if enabled.)

By default, `split_groups` and `split_groups_mean` are set to `True`, so each parameter group will have its own `d` values, however,
they will all use the harmonic mean for the dynamic learning rate. To make each group use its own dynamic LR, set `split_groups_mean=False`.
To use the reference Prodigy behaviour where all groups are combined, set `split_groups=False`. 

The optimiser uses low-rank approximations for the second moment, much like Adafactor. There should be little to no difference 
in training performance, but your mileage may vary. If you encounter problems, you can try disabling factorisation by 
setting `factored=False`. If you're training in bfloat16, and need to squeeze out every last drop of memory, you can also set `factored_fp32=False`, which
will make the factored second moment use the same precision as the weights, rather than float32 (to maximise stability).

The optimiser also supports [fused backward pass](https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html) to significantly lower
gradient memory usage. The `fused_back_pass` argument must be set to `True` so the optimiser knows not to perform the regular step. Please note however that 
your training scripts / UI of choice *must* support the feature for generic optimisers -- as of January 2025, popular trainers such as OneTrainer and Kohya 
hard-code which optimisers have fused backward pass support, and so this optimiser's fused pass will not work out of the box with them.

In some scenarios, it can be advantageous to freeze Prodigy's adaptive stepsize after a certain number of steps. This
can be controlled via the `prodigy_steps` settings. [It's been suggested that all Prodigy needs to do is achieve "escape velocity"](https://arxiv.org/pdf/2409.20325)
in terms of finding a good LR, which it usually achieves after ~25% of training, though this is very dependent on batch size and epochs. 

This setting can be particularly helpful when training diffusion models, which have very different gradient behaviour than what most optimisers are tuned for. 
Prodigy in particular will increase the LR forever if it is not stopped or capped in some way (usually via a decaying LR scheduler). Even if you don't need
to cap LR growth, the optimiser will free all Prodigy-specific state memory once `prodigy_steps` is exceeded, which may improve performance where memory
usage is on the borderline.

## Experimental features

**Adam-atan2:** `eps=None`. Outlined in [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872), 
you can use atan2 in place of the regular division plus epsilon found in most Adam-style optimisers. This makes updates scale-invariant, and removes the need 
to tweak the epsilon. Disabled by default.

**Muon:** `use_muon_pp=True`. This changes the fundamental behaviour of the optimiser for compatible parameters from AdamW to SGD
with a quasi-second moment based on the RMS of the updates. [As explained by Keller Jordan](https://x.com/kellerjordan0/status/1844782418676339059), and demonstrated 
(in various forms) by optimisers such as Shampoo, SOAP and Jordan's Muon, applying preconditioning to the gradient can improve convergence. However, 
this approach may not work in some situations (small batch sizes, fine-tuning) and as such, is disabled by default.

**C-Optim:** `use_cautious=True`. Outlined in [Cautious Optimizers: Improving Training with One Line of Code](https://arxiv.org/pdf/2411.16085). 
Applies a simple modification to parameter updates that promotes values that are aligned with the current gradient. This should result in faster convergence. While not 1:1 compatible with schedule-free, [the implementation by nhamanasu](https://github.com/facebookresearch/schedule_free/pull/54) does work, though improvements may be limited.

**Grams:** `use_grams=True`. Described in [Grams: Gradient Descent with Adaptive Momentum Scaling](https://arxiv.org/abs/2412.17107). 
In a similar vein to C-Optim, the parameter update is modified to separate the update direction from momentum. Thanks to [gesen2egee for the pull request](https://github.com/LoganBooker/prodigy-plus-schedule-free/pull/5).

**ADOPT:** `use_adopt=True`. A partial implementation of [ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate](https://arxiv.org/abs/2411.02853), as we only update the second moment after the parameter update, so as to exclude the current gradient. Disabled by default.

**OrthoGrad:** `use_orthograd=True`. Updates weights using the component of the gradient that is orthogonal to the current weight direction, as described in [Grokking at the Edge of Numerical Stability](https://arxiv.org/pdf/2501.04697). Can help prevent overfitting and improve generalisation. Ignored
for parameters using Muon.

**FOCUS:** `use_focus=True`. Modifies the update step to better handle noise at large step sizes. From [FOCUS: First-Order Concentrated Update Scheme](https://arxiv.org/abs/2501.12243). This method is incompatible with factorisation (which will increase state memory usage), Muon and Adam-atan2. 
Additionally, Prodigy modifies the second moment updates when `d` changes, which may limit the benefits of this method.

**SPEED:** `use_speed=True`. Something of my own creation I've dubbed "Signed Prodigy with ExponEntial D", or SPEED. Prodigy is very
dependent on the magnitude of weights, updates and the gradient, which makes it very difficult to apply other types of optimisations to it. This is my attempt to
decouple Prodigy's LR adaptation from these magnitudes by using just the sign instead, along with a capped growth rate.

## Prodigy FAQ
**Q: Why doesn't Prodigy ever lower the learning rate?**

The original Prodigy's aim is not to act as a combined learning rate calculator and scheduler. It's meant to ballpark a good learning rate, and leave LR decay to your preferred
scheduler (usually cosine). Prodigy + ScheduleFree does combine the two, but it doesn't adjust the LR directly -- in simple terms, it uses a smaller and smaller portion of the averaged 
updates as training goes on, roughly approximating a 1/t schedule. 

Looking at `d` alone tells only parts of the story; this is just the LR Prodigy has calculated, minus any internal modifications. A better metric is observing the norm of the weights, 
you'll see their rate of growth decrease significantly over time, reflecting the small tail of a traditional LR schedule.

**Q: Why isn't Prodigy increasing the LR?**

If Prodigy fails to increase the LR over an extended period (say 100 or more steps), and you're not using bias correction, non-constant LR scheduler or warmup, this usually indicates one of the following:
1. You haven't set the optimiser's `lr` argument to 1. For compatibility with external LR schedulers, the optimiser will multiple the LR you provide with the adaptive one, so if you forget to change this when switching optimisers, the LR will be tiny.
2. The ideal LR is less than `d0` (Prodigy's initial LR guess). Try setting `d0` to a lower value, such as 1e-7 or 1e-8. If this doesn't help, you can also try setting `d_coef=2` (or higher), or `use_speed=True`.
3. External gradient clipping is enabled. This optimiser handles gradient scaling already, so turn off any external clipping/scaling. Alternatively, you can use external scaling, and disable the internal one via `use_stableadamw=False`.

## MNIST results
Generated from the [MNIST example in the schedule-free repository](https://github.com/facebookresearch/schedule_free/tree/main/examples/mnist), using the default settings.
```
Prodigy LR: 0.000832
Test set: Average loss: 0.0472, Accuracy: 9836/10000 (98.36%)
Test set: Average loss: 0.0345, Accuracy: 9879/10000 (98.79%)
Test set: Average loss: 0.0305, Accuracy: 9905/10000 (99.05%)
Test set: Average loss: 0.0295, Accuracy: 9912/10000 (99.12%)
Test set: Average loss: 0.0296, Accuracy: 9916/10000 (99.16%)
Test set: Average loss: 0.0295, Accuracy: 9921/10000 (99.21%)
Test set: Average loss: 0.0305, Accuracy: 9916/10000 (99.16%)
Test set: Average loss: 0.0300, Accuracy: 9915/10000 (99.15%)
Test set: Average loss: 0.0305, Accuracy: 9917/10000 (99.17%)
Test set: Average loss: 0.0310, Accuracy: 9919/10000 (99.19%)
Test set: Average loss: 0.0326, Accuracy: 9923/10000 (99.23%)
Test set: Average loss: 0.0338, Accuracy: 9928/10000 (99.28%)
Test set: Average loss: 0.0345, Accuracy: 9925/10000 (99.25%)
Test set: Average loss: 0.0354, Accuracy: 9925/10000 (99.25%)
```
