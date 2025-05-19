import torch
from .core_optimiser import CoreOptimiser

class ProdigyPlusScheduleFree(CoreOptimiser):
    r"""
    An optimiser based on Prodigy and Schedule-Free. Has additional improvements in the form of optional StableAdamW 
    gradient scaling and Adam-atan2 updates, per parameter group adaptation, lower memory utilisation and fused back pass support.

    The optimiser is designed for bfloat16 and/for float32. Other dtypes may work, but are unsupported.

    Based on code from:
    https://github.com/facebookresearch/schedule_free
    https://github.com/konstmish/prodigy

    Incorporates improvements from these pull requests (credit to https://github.com/dxqbYD and https://github.com/sangoi-exe):
    https://github.com/konstmish/prodigy/pull/23
    https://github.com/konstmish/prodigy/pull/22
    https://github.com/konstmish/prodigy/pull/20

    As with the reference implementation of Schedule-Free, a constant scheduler should be used, along with the appropriate
    calls to `train()` and `eval()`. See the Schedule-Free documentation for more details: https://github.com/facebookresearch/schedule_free
    
    If you do use another scheduler, linear or cosine is preferred, as a restarting scheduler can confuse Prodigy's adaptation logic.

    Leave `lr` set to 1 unless you encounter instability. Do not use with gradient clipping, as this can hamper the
    ability for the optimiser to predict stepsizes. Gradient clipping/normalisation is already handled in the following configurations:
    
    1) `use_stableadamw=True,eps=1e8` (or any reasonable positive epsilon)
    2) `eps=None` (Adam-atan2, scale invariant. Will disable StableAdamW if enabled.)

    By default, `split_groups=True`, so each parameter group will have its own `d` values. To use the reference Prodigy behaviour 
    where all groups are combined, set `split_groups=False`.
    
    In some scenarios, it can be advantageous to freeze Prodigy's adaptive stepsize after a certain number of steps. This
    can be controlled via the `prodigy_steps` settings. This will also free any Prodigy-specific memory used by the optimiser 
    (though with all the memory-related improvements, this should not be significant unless you're training very large models).

    Arguments:
        params (iterable):
            Iterable of parameters to optimise or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the Prodigy learning rate.
            (default: 1.0)
        betas (Tuple[float, float], optional): 
            Coefficients used for computing running averages of gradient and its square. For Schedule-Free, it can be worth
            experimenting with 0.95-0.98 for beta1.
            (default: (0.9, 0.99))
        beta3 (float):
            Coefficient for computing the Prodigy stepsize using running averages. If set to None, uses the value sqrt(beta2).
            (default: None).
        weight_decay (float):
            Decoupled weight decay. To also stop decay from being multiplied by the learning rate, set weight_decay_by_lr=False.
            (default: 0).
        weight_decay_by_lr (boolean):
            By default, weight decay is multiplied by the adaptive learning rate (as per the PyTorch implementation of AdamW). Disabling this 
            feature will stop decay being multiplied by the LR. Its effect will be stronger and less sensitive to training dynamics.
            (default: True)
        d0 (float):
            Initial estimate for Prodigy. Should not require adjustment, but can be increased to 1e-5 or 1e-4 if the optimiser struggles to converge.
            (default: 1e-6).
        d_coef (float):
            Coefficient in the expression for the estimate of d. Values such as 0.5 and 2.0 typically work as well. Changing this parameter 
            is the preferred way to tune the method.
            (default: 1.0)
        d_limiter (boolean):
            Limits the growth of d_hat each step, which can help prevent over-estimated learning rates in early training when gradients
            and EMAs are still stabilising. Stepsize adjustments will have a longer warmup period, but should end up more accurate. Does
            not affect SPEED, which has a built-in limiter.
            (default: True)
        prodigy_steps (int):
            If greater than zero, disable Prodigy's stepsize adjustments after the specified optimiser step and release all state memory 
            required by Prodigy.
            (default: 0)
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. If set to None,
            Adam-atan2 is used instead. This removes the need for epsilon tuning, but may not work well in all situations.
            (default: 1e-8).
        split_groups (boolean):
            Calculate d for each parameter group individually. For example, if training a text encoder beside a Unet. Note this can have a 
            significant impact on training dynamics. Set to False for original Prodigy behaviour, where d is calculated as a single value
            across all parameter groups.
            (default: True)
        split_groups_mean (boolean):
            When split_groups is True, the dynamic learning rate for each group is calculated as: 
                'harmonic mean of d across all groups * per-group LR'
            instead of:
                'per-group d * per-group LR'.
            This provides similar behaviour to the original Prodigy, with the benefit that each group can use its own group LR
            with a more stable d. This can be good if one or more networks struggle to increase their LR when trained together.
            If split_groups is False, this value has no effect.
            (default: False)
        factored (boolean):
            Use factored approximation of the second moment, similar to Adafactor. Reduces memory usage. Disable
            if training results in NaNs or the learning rate fails to grow.
            (default: True)
        factored_fp32 (boolean):
            If False, use the dtype of the gradient for the factored second moment. Because factorisation is an approximation, its dtype
            is forced to float32 by default to avoid stability issues. However, if you're training in low precision for short durations, 
            enabling this can slightly reduce memory usage. Ignored if factored is False.
            (default: True)
        use_bias_correction (boolean):
            Use the RAdam variant of Schedule-Free (https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/radam_schedulefree.py).
            This combines bias correction with automatic warmup. Please note this will significantly dampen Prodigy's adaptive stepsize
            calculations -- it can take up to 10 times longer to start adjusting the learning rate. This can be mitigated somewhat by enabling
            SPEED (use_speed=True).
            (default: False).
        use_stableadamw (boolean):
            Scales parameter updates by their root-mean-square (RMS), in essence identical to Adafactor's update scaling.
            Set to False if the adaptive learning rate never improves or is over-estimated.
            (default: True)
        use_schedulefree (boolean):
            Use the Schedule-Free version of the optimiser. If set to False, the optimiser will use a modified version of the
            reference Prodigy implementation and may require the use of an external LR schedule (cosine is recommended).
            (default: True).
        use_speed (boolean):
            Highly experimental. Simplified Prodigy with rElativE D. Replaces Prodigy's numerator/denominator ratio with a 
            momentum-based estimate of directional progress. SPEED uses less memory, is scale-insensitive, and can be 
            a better choice when training multiple networks, however, it can be unstable when used with weight decay.
            (default: False)
        stochastic_rounding (boolean):
            Use stochastic rounding for bfloat16 weights (https://github.com/pytorch/pytorch/issues/120376). Brings
            bfloat16 training performance closer to that of float32.
            (default: True)
        fused_back_pass (boolean):
            Stops the optimiser from running the normal step method. Set to True if using fused backward pass. Really only
            needed for scripts and UIs that call the regular step method even when using fused backward pass (OneTrainer).
            (default: False)
        use_cautious (boolean):
            Experimental. Perform "cautious" updates, as proposed in https://arxiv.org/pdf/2411.16085. Modifies
            the update to isolate and boost values that align with the current gradient. Note that we do not have
            access to a first moment, so this deviates from the paper (we apply the mask directly to the update).
            May have a limited effect.
            (default: False)
        use_grams (boolean):
            Experimental. Perform "grams" updates, as proposed in https://arxiv.org/abs/2412.17107. Modifies 
            the update using sign operations that align with the current gradient. Note that we do not have
            access to a first moment, so this deviates from the paper (we apply the sign directly to the update).
            May have a limited effect.
            (default: False)
        use_adopt (boolean):
            Experimental. Performs a modified step where the second moment is updated after the parameter update,
            so as not to include the current gradient in the denominator. This is a partial implementation of ADOPT 
            (https://arxiv.org/abs/2411.02853), as we don't have a first moment to use for the update.
            (default: False)
        use_orthograd (boolean):
            Experimental. Updates weights using the component of the gradient that is orthogonal to the current 
            weight direction, as described in "Grokking at the Edge of Numerical Stability" (https://arxiv.org/pdf/2501.04697).
            Can help prevent overfitting and improve generalisation.
            (default: False)
        use_focus (boolean):
            Experimental. Modifies the update step to better handle noise at large step sizes. From 
            "FOCUS: First-Order Concentrated Update Scheme" (https://arxiv.org/abs/2501.12243). This method is
            incompatible with factorisation and Adam-atan2.
            (default: False)
    """
    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.99), beta3=None,
                 weight_decay=0.0,
                 weight_decay_by_lr=True,
                 d0=1e-6, d_coef=1.0,
                 d_limiter=True,
                 prodigy_steps=0,
                 eps=1e-8,
                 split_groups=True,
                 split_groups_mean=False,
                 factored=True,
                 factored_fp32=True,
                 use_bias_correction=False,
                 use_stableadamw=True,
                 use_schedulefree=True,
                 use_speed=False,
                 stochastic_rounding=True,
                 fused_back_pass=False,
                 use_cautious=False,
                 use_grams=False,
                 use_adopt=False,
                 use_orthograd=False,
                 use_focus=False,
                 **kwargs):

        super().__init__(params=params, lr=lr,
                        betas=betas, beta3=beta3,
                        weight_decay=weight_decay,
                        weight_decay_by_lr=weight_decay_by_lr,
                        d0=d0, d_coef=d_coef,
                        d_limiter=d_limiter,
                        prodigy_steps=prodigy_steps,
                        eps=eps,
                        split_groups=split_groups,
                        split_groups_mean=split_groups_mean,
                        factored=factored,
                        factored_fp32=factored_fp32,
                        use_bias_correction=use_bias_correction,
                        use_stableadamw=use_stableadamw,
                        use_schedulefree=use_schedulefree,
                        use_speed=use_speed,
                        stochastic_rounding=stochastic_rounding,
                        fused_back_pass=fused_back_pass,
                        use_cautious=use_cautious,
                        use_grams=use_grams,
                        use_adopt=use_adopt,
                        use_orthograd=use_orthograd,
                        use_focus=use_focus,
                        **kwargs)

    @torch.no_grad()
    def set_train_mode(self, train):
        for group in (g for g in self.param_groups if g['use_schedulefree'] and g['train_mode'] != train):
            beta1, _, _ = self.get_betas(group)
            w = 1 - beta1 if train else 1 - 1 / beta1
            for p in group['params']:
                z = self.state[p].get('z')
                if z is not None:
                    p.lerp_(end=z.to(device=p.device), weight=w)
            group['train_mode'] = train

    def eval(self):
        self.set_train_mode(False)

    def train(self):
        self.set_train_mode(True)

    @torch.no_grad()
    def initialise_state(self, p, group):
        state, needs_init = self.initialise_state_internal(p, group)

        if needs_init:
            if group['use_schedulefree']:
                state['z'] = p.detach().clone(memory_format=torch.preserve_format)
            else:
                state['exp_avg'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format).detach()

        return state

    @torch.no_grad()
    def update_params(self, y, z, update, group, dlr):
        beta1, _, _ = self.get_betas(group)
        decay = self.get_weight_decay(group)

        weight = dlr ** 2
        weight_sum = group['running_weight_sum'] = group.get('weight_sum', 0) + weight
        ckp1 = weight / weight_sum if weight_sum else 0

        xy_step = 1 - beta1 * (1 - ckp1)
        group['effective_lr'] = group['lr'] * xy_step

        cautious, grams = group['use_cautious'], group['use_grams']

        y_wd = None
        if decay != 0: # Weight decay at Y.
            if group['weight_decay_by_lr']:
                update.add_(y, alpha=decay)
            else:
                y_wd = y.clone().detach()

        if cautious or grams:
            u = (y - z).mul_(ckp1).add_(update, alpha=dlr * xy_step)
            z.sub_(update, alpha=dlr)

            if cautious:
                # "Cautious Optimizer (C-Optim): Improving Training with One Line of Code": https://github.com/kyleliang919/c-optim
                # ScheduleFree implementation by nhamanasu: https://github.com/facebookresearch/schedule_free/pull/54
                mask = update.mul_(u).sign_().clamp_min_(0)
                mask.div_(mask.mean().clamp_min(1e-3))
                u.mul_(mask)
            elif grams:
                # "Grams: Gradient Descent with Adaptive Momentum Scaling": https://arxiv.org/abs/2412.17107
                u.abs_().mul_(update.sign_())

            y.sub_(u)
            del u
        else:
            y.lerp_(end=z, weight=ckp1)
            z.sub_(update, alpha=dlr)
            y.sub_(update, alpha=dlr * xy_step)

        if y_wd is not None: # Apply decoupled LR decay.
            z.sub_(y_wd, alpha=decay)
            y.sub_(y_wd, alpha=decay * xy_step)
            del y_wd

    @torch.no_grad()
    def step_param_prodigy(self, p, group):
        k = group['k']
        use_adopt = group['use_adopt']
        use_bias_correction = group['use_bias_correction']
        stochastic = group['stochastic_rounding']
        beta1, beta2, _ = self.get_betas(group)

        state = self.initialise_state(p, group)

        y = p.float()

        grad = p.grad.to(dtype=torch.float32, copy=True)
        dlr = self.get_dlr(group)

        if use_bias_correction:
            dlr, beta2, _ = self.get_bias_correction(dlr, beta2, k)

        update = None

        if use_adopt and k == 1:
            self.update_second_moment(state, group, grad, 0, y, return_denom=False)
            del grad
        else:
            denom = self.update_second_moment(state, group, grad, beta2, y, denom_before_update=use_adopt)

            if use_adopt:
                clamp_range = k ** 0.25
                grad = self.update_(grad, denom, group, y).clamp_(-clamp_range, clamp_range)

            exp_avg = self.update_first_moment(state, group, grad, beta1)

            if group['use_cautious']:
                mask = grad.mul(exp_avg).sign_().clamp_min_(0)
                mask.div_(mask.mean().clamp(min=1e-3))
                grad.mul_(exp_avg)
                del mask
            elif group['use_grams']:
                mask = exp_avg.abs()
                grad.sign_().mul_(mask)
                del mask
            else:
                grad.copy_(exp_avg)

            update = grad if use_adopt else self.update_(grad, denom, group, y)
            del denom

        if update is not None:
            if group['use_orthograd']:
                update = self.orthograd_(y, update)

            if group['use_stableadamw']:
                update = self.rms_clip_(update)

            self.update_prodigy(state, group, p.grad, p)

            decay = self.get_weight_decay(group)
            if decay != 0:
                if group['weight_decay_by_lr']:
                    decay *= dlr
                y.mul_(1 - decay)

            y.sub_(update, alpha=dlr)

            self.smart_copy(p, y, stochastic, True)

            del update

    @torch.no_grad()
    def step_param_schedulefree(self, p, group):
        if not group['train_mode']:
            raise Exception("Not in train mode!")

        k = group['k']
        use_adopt = group['use_adopt']
        use_bias_correction = group['use_bias_correction']
        stochastic = group['stochastic_rounding']
        _, beta2, _ = self.get_betas(group)

        state = self.initialise_state(p, group)

        z_state = state['z']
        y, z = p.float(), z_state.float()

        grad = p.grad.to(dtype=torch.float32, copy=True)
        dlr = self.get_dlr(group)

        if use_bias_correction:
            dlr, beta2, rho_t = self.get_bias_correction(dlr, beta2, k)

        update = None

        if use_adopt and k == 1:
            self.update_second_moment(state, group, grad, 0, z, return_denom=False)
            del grad
        else:
            denom = self.update_second_moment(state, group, grad, beta2, z, denom_before_update=use_adopt)

            if use_bias_correction and rho_t <= 4.0:
                update = grad
            else:
                grad.mul_(group['d'])
                update = self.update_(grad, denom, group, z)
            del denom

        if update is not None:
            if group['use_orthograd']:
                update = self.orthograd_(z, update)

            if group['use_stableadamw']:
                update = self.rms_clip_(update)

            self.update_prodigy(state, group, p.grad, z_state)
            self.update_params(y, z, update, group, dlr)

            self.smart_copy(p, y, stochastic, True)
            self.smart_copy(z_state, z, stochastic, True)

            del update

    @torch.no_grad()
    def step_param(self, p, group):
        self.on_start_step()

        if p.grad is not None:
            if group['use_schedulefree']:
                self.step_param_schedulefree(p, group)
            else:
                self.step_param_prodigy(p, group)

        self.on_end_step()