# Prodigy + Schedule-Free
*Eliminating hyperparameters, one commit at a time.*

**Current status:** Experimental

## Changes in v2.0.0
* Schedule-Free can be disabled using `use_schedulefree=False`. This reverts the optimiser to straight Prodigy, while keeping per-group learning rates and the rest of the features of the optimiser (StableAdamW, factorisation, and so on). In this mode, it is best paired with a decaying LR scheduler.
* Changed `split_groups_mean` to `False` so full, per-group stepsize adaptation is active by default.
* The Prodigy implementation adjusted to more closely match the original.
* StableAdamW use a soft scaling formula based on the square root of the RMS. This should result in more accurate LR adjustments.
* SPEED has been completely reworked, and should be more stable and perform better on a wide range of tasks. Personally, I now prefer it over base Prodigy.
* Removed Muon. It never really worked correctly when combined with Schedule-Free and Prodigy.
* Removed the "confidence" learning rate limiter, which ended up being too aggressive for non-SDXL training and fine-tuning.
* Added a limiter to d growth to prevent over-estimated LRs when gradients and EMAs are still stabilising. It can be disabled via `d_limiter=False`.
* Added logging group parameter `effective_lr`. This value is for reporting only; rather than using `d * lr`, you can track `d * effective_lr`. This provides a closer approximation of the LR when Schedule-Free is on. Once the LR has settled, `d * effective_lr` should be close to `(1 - beta1) * d * lr`.
* Sufficied to say, you should not resume training started with older versions of the optimiser with this one. It will break.

## Installation
For the most recent release:
```
pip install prodigy-plus-schedule-free
```
Please note v2.0.0 **includes breaking changes**. Do not use it to resume training runs on older versions! Please check the changelog above for more details. For the previous release (v1.9.2), use:
```
pip install prodigy-plus-schedule-free==1.9.2
```

## Usage
```python
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
optimizer = ProdigyPlusScheduleFree(model.parameters(), lr=1.0, betas=(0.9, 0.99), beta3=None, 
                 					weight_decay=0.0, weight_decay_by_lr=True, d0=1e-6, d_coef=1.0,
							        d_limiter=True, prodigy_steps=0, schedulefree_c=0, eps=1e-8,
									split_groups=True, split_groups_mean=False,
                 					factored=True, factored_fp32=True, use_bias_correction=False,
                 					use_stableadamw=True, use_schedulefree=True, use_speed=False,
                 					stochastic_rounding=True, fused_back_pass=False,
                 					use_cautious=False, use_grams=False, use_adopt=False,
									use_orthograd=False, use_focus=False)
```

> [!IMPORTANT]
> As with the reference implementation of Schedule-Free, a constant scheduler should be used, along with the appropriate calls to `optimizer.train()` and `optimizer.eval()`. See the Schedule-Free documentation for more details: https://github.com/facebookresearch/schedule_free

> [!TIP]
> [Recent research suggests](https://arxiv.org/pdf/2507.09846) `betas=(0.95, 0.99)` works better in most situations for Schedule-Free. For now, the default remains `betas=(0.9, 0.99)`.

## TLDR
The default settings should "just work", but there are a few configurations you can try to improve things.

### Gradient scaling/clipping
By default, the optimiser uses StableAdamW to scale parameter updates, which reduces the need for external gradient scaling or clipping. However, this can also hamper Prodigy's ability to adapt the stepsize. While the optimiser includes internal logic to mostly mitigate this, you can try `eps=None` to use Adam-atan2 instead, or set `use_stableadamw=False` and use external gradient clipping.

### Training multiple networks
Unlike reference Prodigy, this optimiser will adjust the stepsize per parameter group, allowing one to train multiple networks at the same time. To train all groups together like the original Prodigy, set `split_groups=False`.

> [!TIP]
> As of v2.0.0, `split_groups_mean` is `False` by default, so full, per-group training is always active. Set `split_groups_mean=True` to replicate the behaviour of older versions.

### Turning off Prodigy
Earlier versions of the optimiser recommended setting `prodigy_steps` equal to 5-25% of your total step count, but this should not be necessary with recent updates. That said, you can still use the setting to make sure the LR does not change after a certain step, and free any memory used by Prodigy for adapting the step size.

## Details
An optimiser based on Prodigy that includes Schedule-Free logic and much, much lower memory usage, the aim being to remove the need to set any hyperparameters. Of course, that's never the case with any optimiser, but hopefully, this comes close!

Hyperparameters eliminated: Learning rate (Prodigy), LR scheduler (Schedule-Free), epsilon (Adam-atan2, optional, not enabled by default).

Based on code from:
* https://github.com/facebookresearch/schedule_free
* https://github.com/konstmish/prodigy

Incorporates improvements from these pull requests (credit to https://github.com/dxqbYD, https://github.com/sangoi-exe and https://github.com/nhamanasu):
* https://github.com/konstmish/prodigy/pull/23
* https://github.com/konstmish/prodigy/pull/22
* https://github.com/konstmish/prodigy/pull/20
* https://github.com/facebookresearch/schedule_free/pull/54

If you do use another scheduler, linear or cosine is preferred, as a restarting scheduler can confuse Prodigy's adaptation logic.

Leave `lr` set to 1 unless you encounter instability. Do not use with gradient clipping, as this can hamper the ability for the optimiser to predict stepsizes. Gradient clipping/normalisation is already handled in the following configurations:

1) `use_stableadamw=True,eps=1e8` (or any reasonable positive epsilon. This is the default.)
2) `eps=None` (Adam-atan2, scale invariant. Will disable StableAdamW if enabled.)

The optimiser uses low-rank approximations for the second moment, much like Adafactor. There should be little to no difference in training performance, but your mileage may vary. If you encounter problems, you can try disabling factorisation by setting `factored=False`. If you're training in bfloat16, and need to squeeze out every last drop of memory, you can also set `factored_fp32=False`, which will make the factored second moment use the same precision as the weights, rather than float32 (to maximise stability).

The optimiser also supports [fused backward pass](https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html) to significantly lower gradient memory usage. The `fused_back_pass` argument must be set to `True` so the optimiser knows not to perform the regular step. Please note however that your training scripts / UI of choice *must* support the feature for generic optimisers -- as of May 2025, Kohya hard-codes which optimisers have fused backward pass support, and so this optimiser's fused pass will not work out of the box with it.

In some scenarios, it can be advantageous to freeze Prodigy's adaptive stepsize after a certain number of steps. This can be controlled via the `prodigy_steps` settings. [It's been suggested that all Prodigy needs to do is achieve "escape velocity"](https://arxiv.org/pdf/2409.20325) in terms of finding a good LR, which it usually achieves after ~25% of training, though this is very dependent on batch size and epochs.

This setting can be particularly helpful when training diffusion models, which have very different gradient behaviour than what most optimisers are tuned for. Prodigy in particular will increase the LR forever if it is not stopped or capped in some way (usually via a decaying LR scheduler). Even if you don't need to cap LR growth, the optimiser will free all Prodigy-specific state memory once `prodigy_steps` is exceeded, which may improve performance where memory usage is on the borderline.

## Experimental features
**Adam-atan2:** `eps=None`. Outlined in [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872), you can use atan2 in place of the regular division plus epsilon found in most Adam-style optimisers. This makes updates scale-invariant, and removes the need to tweak the epsilon. Disabled by default.

**C-Optim:** `use_cautious=True`. Outlined in [Cautious Optimizers: Improving Training with One Line of Code](https://arxiv.org/pdf/2411.16085). Applies a simple modification to parameter updates that promotes values that are aligned with the current gradient. This should result in faster convergence. While not 1:1 compatible with Schedule-Free, [the implementation by nhamanasu](https://github.com/facebookresearch/schedule_free/pull/54) does work, though improvements may be limited.

**Grams:** `use_grams=True`. Described in [Grams: Gradient Descent with Adaptive Momentum Scaling](https://arxiv.org/abs/2412.17107). In a similar vein to C-Optim, the parameter update is modified to separate the update direction from momentum. Thanks to [gesen2egee for the pull request](https://github.com/LoganBooker/prodigy-plus-schedule-free/pull/5).

**ADOPT:** `use_adopt=True`. A partial implementation of [ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate](https://arxiv.org/abs/2411.02853), as we only update the second moment after the parameter update, so as to exclude the current gradient. Disabled by default.

**OrthoGrad:** `use_orthograd=True`. Updates weights using the component of the gradient that is orthogonal to the current weight direction, as described in [Grokking at the Edge of Numerical Stability](https://arxiv.org/pdf/2501.04697). Can help prevent overfitting and improve generalisation.

**FOCUS:** `use_focus=True`. Modifies the update step to better handle noise at large step sizes. From [FOCUS: First-Order Concentrated Update Scheme](https://arxiv.org/abs/2501.12243). This method is incompatible with factorisation (which will increase state memory usage), Muon and Adam-atan2. Additionally, Prodigy modifies the second moment updates when `d` changes, which may limit the benefits of this method.

**SPEED:** `use_speed=True`. Something of my own creation I've dubbed _Simplified Prodigy with rElativE D_. It replaces Prodigy's numerator/denominator ratio with a momentum-based estimate of directional progress. SPEED uses less memory, is scale-insensitive, and can be a better choice when training multiple networks, however, it can be unstable when used with weight decay or for extremely long training runs (where it's recommended to use `prodigy_steps`).

> [!NOTE]
> If `use_schedulefree=False`, all experimental features are implemented as per their reference implementations.

## Prodigy FAQ
**Q: Why doesn't Prodigy ever lower the learning rate?**

The original Prodigy's aim is not to act as a combined learning rate calculator and scheduler. It's meant to ballpark a good learning rate, and leave LR decay to your preferred scheduler (usually cosine). Prodigy + Schedule-Free does combine the two, but it doesn't adjust the LR directly -- in simple terms, it uses a smaller and smaller portion of the averaged updates as training goes on, roughly approximating a 1/t schedule. 

Looking at `d` alone tells only parts of the story; this is just the LR Prodigy has calculated, minus any internal modifications. A better metric is observing the norm of the weights, you'll see their rate of growth decrease significantly over time, reflecting the small tail of a traditional LR schedule. You can also log `group['effective_lr'] * group['d']`, which gives a much more accurate representation of Schedule-Free's LR.

**Q: Why isn't Prodigy increasing the LR?**

If Prodigy fails to increase the LR over an extended period (say 100 or more steps), and you're not using bias correction, non-constant LR scheduler or warmup, this usually indicates one of the following:
1. You haven't set the optimiser's `lr` argument to 1. For compatibility with external LR schedulers, the optimiser will multiple the LR you provide with the adaptive one, so if you forget to change this when switching optimisers, the LR will be tiny.
2. The ideal LR is less than `d0` (Prodigy's initial LR guess). Try setting `d0` to a lower value, such as 1e-7 or 1e-8. If this doesn't help, you can also try setting `d_coef=2` (or higher), or `use_speed=True`.
3. The value for `d0` is too conservative and starving Prodigy. Try raising `d0` to 1e-5 or 1e-4.
4. External gradient clipping is enabled. This optimiser handles gradient scaling already, so turn off any external clipping/scaling. Alternatively, you can use external scaling, and disable the internal one via `use_stableadamw=False`.
5. Set `d_limiter=False`. The growth limiter should never prevent the LR from increasing, but it's possible your training scenario requires faster adjustments.

## MNIST results
Generated from the [MNIST example in the Schedule-Free repository](https://github.com/facebookresearch/schedule_free/tree/main/examples/mnist), using the default settings.
```
Prodigy LR: 0.000862
Test set: Average loss: 0.0456, Accuracy: 9849/10000 (98.49%)
Test set: Average loss: 0.0347, Accuracy: 9881/10000 (98.81%)
Test set: Average loss: 0.0324, Accuracy: 9898/10000 (98.98%)
Test set: Average loss: 0.0308, Accuracy: 9911/10000 (99.11%)
Test set: Average loss: 0.0299, Accuracy: 9913/10000 (99.13%)
Test set: Average loss: 0.0285, Accuracy: 9919/10000 (99.19%)
Test set: Average loss: 0.0289, Accuracy: 9922/10000 (99.22%)
Test set: Average loss: 0.0300, Accuracy: 9925/10000 (99.25%)
Test set: Average loss: 0.0306, Accuracy: 9924/10000 (99.24%)
Test set: Average loss: 0.0319, Accuracy: 9927/10000 (99.27%)
Test set: Average loss: 0.0339, Accuracy: 9925/10000 (99.25%)
Test set: Average loss: 0.0349, Accuracy: 9928/10000 (99.28%)
Test set: Average loss: 0.0366, Accuracy: 9924/10000 (99.24%)
Test set: Average loss: 0.0377, Accuracy: 9926/10000 (99.26%)
```
With `use_speed=True`:
```
Prodigy LR: 0.002582
Test set: Average loss: 0.0401, Accuracy: 9861/10000 (98.61%)
Test set: Average loss: 0.0309, Accuracy: 9908/10000 (99.08%)
Test set: Average loss: 0.0276, Accuracy: 9916/10000 (99.16%)
Test set: Average loss: 0.0259, Accuracy: 9928/10000 (99.28%)
Test set: Average loss: 0.0258, Accuracy: 9930/10000 (99.30%)
Test set: Average loss: 0.0268, Accuracy: 9931/10000 (99.31%)
Test set: Average loss: 0.0288, Accuracy: 9926/10000 (99.26%)
Test set: Average loss: 0.0305, Accuracy: 9927/10000 (99.27%)
Test set: Average loss: 0.0309, Accuracy: 9934/10000 (99.34%)
Test set: Average loss: 0.0309, Accuracy: 9932/10000 (99.32%)
Test set: Average loss: 0.0323, Accuracy: 9933/10000 (99.33%)
Test set: Average loss: 0.0337, Accuracy: 9934/10000 (99.34%)
Test set: Average loss: 0.0345, Accuracy: 9932/10000 (99.32%)
Test set: Average loss: 0.0352, Accuracy: 9933/10000 (99.33%)
```
