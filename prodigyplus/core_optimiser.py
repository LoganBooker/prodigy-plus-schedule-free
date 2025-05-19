import torch
from statistics import harmonic_mean

class CoreOptimiser(torch.optim.Optimizer):
    VERSION = (2, 0, 0)

    def __init__(self, params, **kwargs):
        if not 0.0 < kwargs['d0']:
            raise ValueError("Invalid d0 value: {}".format(kwargs['d0']))
        if not 0.0 < kwargs['lr']:
            raise ValueError("Invalid learning rate: {}".format(kwargs['lr']))
        if kwargs['weight_decay'] < 0:
            raise ValueError("Invalid weight decay: {}".format(kwargs['weight_decay']))
        if kwargs['prodigy_steps'] < 0:
            raise ValueError("Invalid Prodigy steps: {}".format(kwargs['prodigy_steps']))
        if kwargs['eps'] is not None and not 0.0 < kwargs['eps']:
            raise ValueError("Invalid epsilon value: {}".format(kwargs['eps']))
        if not 0.0 <= kwargs['betas'][0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(kwargs['betas'][0]))
        if kwargs['betas'][1] is not None and not 0.0 <= kwargs['betas'][1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(kwargs['betas'][1]))
        if kwargs['beta3'] is not None and not 0.0 <= kwargs['beta3'] < 1.0:
            raise ValueError("Invalid beta3 parameter: {}".format(kwargs['beta3']))

        self.try_hook_kohya_fbp()

        if kwargs['eps'] is None:
            self.log(f"'eps' is None, Adam-atan2 enabled.")
            if kwargs['use_stableadamw']:
                self.log(f"'use_stableadamw' has been disabled (mutually exclusive with Adam-atan2).")
                kwargs['use_stableadamw'] = False

        # SPEED expects (mostly) unmodified weights during training to determine LR. If weight growth is dampened too much, 
        # SPEED can massively overestimate the LR. This only seems to be a problem in certain edge cases; higher weight decay 
        # should be fine for the most part, but warn the user anyway.
        if kwargs['use_speed'] and kwargs['weight_decay'] > 0:
            self.log(f"WARNING: Weight decay with 'use_speed' detected! If training becomes unstable, try lower values.")

        if kwargs['use_cautious'] and kwargs['use_grams']:
            self.log(f"'use_grams' has been disabled (mutually exclusive with 'use_cautious').")
            kwargs['use_grams'] = False

        if kwargs['use_focus']:
            if kwargs['factored']:
                self.log(f"'factored' has been disabled (incompatible with 'use_focus').")
                kwargs['factored'] = False
            if kwargs['eps'] is None:
                self.log(f"Adam-atan2 ('eps=None') has been disabled (incompatible with 'use_focus').")
                # We skip the Adam-atan2 branch entirely when FOCUS is enabled.

        defaults = dict(kwargs)

        defaults['optimiser_version'] = CoreOptimiser.VERSION
        defaults['effective_lr'] = defaults['lr']
        defaults['d'] = defaults['d_prev'] = defaults['d0']
        defaults['d_denom'] = defaults['d_numerator'] = 0
        defaults['prev_d_numerator'] = defaults['max_d_numerator'] = 0
        defaults['shared_d'] = None
        defaults['is_global_group'] = False
        defaults['train_mode'] = True
        defaults['k'] = 1

        super().__init__(params, defaults)

        global_group = self.get_global_group()

        if global_group['split_groups'] and len(self.param_groups) == 1:
            self.log(f"Optimiser contains single param_group -- 'split_groups' has been disabled.")
            global_group['split_groups'] = False

        self.parameters_to_process = None

        self.print_dtype_warning = True
        self.print_version_check = True

        # Use tensors to keep everything on device during parameter loop.
        for group in (self.param_groups if global_group['split_groups'] else [global_group]):
            p = group['params'][0]
            group['running_d_numerator'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
            group['running_d_denom'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)

    @torch.no_grad()
    def eval(self):
        pass

    @torch.no_grad()
    def train(self):
        pass

    def get_global_group(self):
        if getattr(self, '_global_group', None) is None:
            self._global_group = next(
                (g for g in self.param_groups if g.get('is_global_group', False)),
                self.param_groups[0]
            )
            self._global_group['is_global_group'] = True
        return self._global_group

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def supports_fused_back_pass(self):
        return True

    def log(self, message):
        print(f"[{self.__class__.__name__}] {message}")

    @torch.no_grad()
    def get_sliced_tensor(self, tensor, slice_p=11):
        return tensor.ravel()[::slice_p]

    @torch.no_grad()
    def get_running_values_for_group(self, group):
        if not group['split_groups']:
            group = self.get_global_group()

        p = group['params'][0]
        numerator, denom = group['running_d_numerator'], group['running_d_denom']
        if numerator.device != p.device:
            group['running_d_numerator'] = numerator = numerator.to(p.device)
        if denom.device != p.device:
            group['running_d_denom'] = denom = denom.to(p.device)
        return numerator, denom

    # Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
    def orthograd_(self, p, grad):
        if p.norm(2) <= 1e-30:
            return grad

        G_shape = grad.shape
        w = p.view(-1)
        g = grad.view(-1)
        g_norm = g.norm(2)

        proj = torch.dot(w, g) / torch.dot(w, w).add(1e-30)
        g_orth = g.sub_(w, alpha=proj)
        g_orth_scaled = g_orth.mul_(g_norm / g_orth.norm(2).add(1e-30))

        return g_orth_scaled.view(G_shape)

    # Implementation by Nerogar. From: https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    def copy_stochastic_(self, target, source):
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))

    def smart_copy(self, target, source, stochastic_rounding, smart_delete_source):
        if target is source:
            return

        if stochastic_rounding and target.dtype == torch.bfloat16 and source.dtype == torch.float32:
            self.copy_stochastic_(target, source)
        else:
            target.copy_(source)

        if smart_delete_source:
            del source

    # Modified Adafactor factorisation implementation by Ross Wightman 
    # https://github.com/huggingface/pytorch-image-models/pull/2320
    @torch.no_grad()
    def factored_dims(self,
        shape,
        factored,
        min_dim_size_to_factor):
        r"""Whether to use a factored second moment estimator.
        This function returns a tuple with the two largest axes to reduce over.
        If all dimensions have size < min_dim_size_to_factor, return None.
        Args:
        shape: an input shape
        factored: whether to use factored second-moment estimator for > 2d vars.
        min_dim_size_to_factor: only factor accumulator if all array dimensions are greater than this size.
        Returns:
        None or a tuple of ints
        """
        if not factored or len(shape) < 2:
            return None
        if all(dim < min_dim_size_to_factor for dim in shape):
            return None
        sorted_dims = sorted(((x, i) for i, x in enumerate(shape)))
        return int(sorted_dims[-2][1]), int(sorted_dims[-1][1])

    @torch.no_grad()
    def initialise_state(self, p, group):
        raise Exception("Not implemented!")

    @torch.no_grad()
    def initialise_state_internal(self, p, group):
        state = self.state[p]
        needs_init = len(state) == 0

        if needs_init:
            grad = p.grad

            if getattr(self, "print_dtype_warning", True) and (p.dtype not in {torch.bfloat16, torch.float32} or grad.dtype not in {torch.bfloat16, torch.float32}):
                self.log(f"Unsupported parameter dtype! The optimiser is designed for bf16 and fp32 training only. Other dtypes may produce unexpected results.")
                self.print_dtype_warning = False

            dtype = torch.bfloat16 if grad.dtype == torch.float32 else grad.dtype
            sliced_data = self.get_sliced_tensor(p)

            # NOTE: We don't initialise z/exp_avg here -- subclass needs to do that.
            factored_dims = self.factored_dims(
                grad.shape,
                factored=group['factored'] and not group['use_focus'],
                min_dim_size_to_factor=32
            )

            if factored_dims is not None:
                dc, dr = factored_dims
                row_shape = list(grad.shape)
                row_shape[dr] = 1
                col_shape = list(grad.shape)
                col_shape[dc] = 1

                factored_dtype = torch.float32 if group['factored_fp32'] else grad.dtype
                state["exp_avg_sq_row"] = torch.zeros(row_shape, dtype=factored_dtype, device=p.device).detach()
                state["exp_avg_sq_col"] = torch.zeros(col_shape, dtype=factored_dtype, device=p.device).detach()
                state["exp_avg_sq_metadata"] = (dr, dc)
            else:
                state['exp_avg_sq'] = torch.zeros_like(grad, memory_format=torch.preserve_format).detach()

            # If the initial weights are zero, don't bother storing them.
            if p.any() > 0:
                state['p0'] = sliced_data.to(dtype=dtype, memory_format=torch.preserve_format, copy=True).detach()
            else:
                state['p0'] = torch.tensor(0.0, dtype=dtype, device=p.device)

            if not group['use_speed']:
                state['s'] = torch.zeros_like(sliced_data, memory_format=torch.preserve_format, dtype=dtype).detach()

        return state, needs_init

    def get_betas(self, group):
        beta1, beta2 = group['betas']
        beta3 = group['beta3']
        return beta1, beta2, beta3 if beta3 is not None else beta2 ** 0.5

    def get_weight_decay(self, group):
        return group['weight_decay']

    def get_bias_correction(self, dlr, beta2, k):
        beta2_t = beta2 ** k
        bias_correction2 = 1 - beta2_t

        # maximum length of the approximated SMA
        rho_inf = 2 / (1 - beta2) - 1
        # compute the length of the approximated SMA
        rho_t = rho_inf - 2 * k * beta2_t / bias_correction2
        rect = (
            ((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
            if rho_t > 4.0
            else 0.0
        )
        dlr *= rect
        beta2 = 1 - (1 - beta2) / (1 - beta2_t)

        return (dlr, beta2, rho_t)

    @torch.no_grad()
    def update_d_stats_and_reset(self, group):
        k = group['k']
        prodigy_steps = group['prodigy_steps']
        
        if prodigy_steps > 0 and k >= prodigy_steps:
            return

        d, d0 = group['d'], group['d0']
        _, _, beta3 = self.get_betas(group)

        running_d_numerator, running_d_denom = self.get_running_values_for_group(group)

        max_d_numerator = group['max_d_numerator']
        d_numerator = group['d_numerator']
        d_numerator *= beta3

        d_numerator_item = running_d_numerator.item()
        d_denom_item = running_d_denom.item()

        # Prevent the accumulation of negative values in the numerator in early training.
        # We still allow negative updates once progress starts being made, as this is 
        # important for regulating the adaptive stepsize.
        if d > d0 or d_numerator_item > 0:
            d_numerator += d_numerator_item

        group['prev_d_numerator'] = group['d_numerator']
        group['max_d_numerator'] = max(d_numerator, max_d_numerator)
        group['d_numerator'] = d_numerator
        group['d_denom'] = d_denom_item

        running_d_numerator.zero_()
        running_d_denom.zero_()

    @torch.no_grad()
    def calculate_d(self, group):
        k = group['k']
        prodigy_steps = group['prodigy_steps']

        if prodigy_steps > 0 and k >= prodigy_steps:
            return

        d, d_prev, d_coef = group['d'], group['d_prev'], group['d_coef']
        d_numerator, d_denom = group['d_numerator'], group['d_denom']
        prev_d_numerator, max_d_numerator = group['prev_d_numerator'], group['max_d_numerator']

        if group['use_speed']:
            if d_numerator >= max_d_numerator and d_numerator > 0 and prev_d_numerator > 0:
                d_power = 0.5 * (d_prev / d) ** 2
                d_hat = min(2 ** d_power, (d_numerator / prev_d_numerator) ** 0.5)
                d = max(d, d * d_hat)
        elif d_denom > 0:
            d_hat = d_numerator / d_denom
            if group['d_limiter']:
                d_hat = min(d * (2 ** 0.25), d_hat)
            d = max(d, d_hat * d_coef)

        group['d_prev'] = group['d']
        group['d'] = d

    def on_start_step(self):
        if not self.parameters_to_process:
            # Optimiser hasn't run yet (or is starting a new step), so initialise.
            self.parameters_to_process = sum(len(group['params']) for group in self.param_groups)

            # Check if resuming training started on an older version.
            if getattr(self, "print_version_check", True):
                global_group = self.get_global_group()
                version = global_group.get('optimiser_version', None)
                if version != self.VERSION:
                    expected, actual = '.'.join(map(str, self.VERSION)), '.'.join(map(str, version)) if version else '<= 1.9.2'
                    self.log(f"Optimiser version mismatch or missing: expected {expected}, got {actual}. Training resume is not supported between versions!")
                self.print_version_check = False

    def on_end_step(self):
        self.parameters_to_process -= 1

        if self.parameters_to_process == 0:
            global_group = self.get_global_group()

            # Update d for next optimiser step.
            if global_group['split_groups']:
                for i, group in enumerate(self.param_groups):
                    if group['prodigy_steps'] > 0 and group['k'] == group['prodigy_steps']:
                        self.log(f"Prodigy stepsize adaptation disabled after {group['k']} steps for param_group {i}.")

                    self.update_d_stats_and_reset(group)
                    self.calculate_d(group)

                if global_group['split_groups_mean']:
                    d_mean = harmonic_mean(group['d'] for group in self.param_groups)
                    for group in self.param_groups:
                        group['shared_d'] = d_mean
            else:
                # When groups aren't split, calculate d for the first group (which collects stats for all groups in non-split mode), 
                # then copy to all other groups.
                self.update_d_stats_and_reset(global_group)
                self.calculate_d(global_group)

                for i, group in enumerate(self.param_groups):
                    if group['prodigy_steps'] > 0 and group['k'] == group['prodigy_steps']:
                        self.log(f"Prodigy stepsize adaptation disabled after {group['k']} steps for param_group {i}.")

                    for key in ['d', 'd_prev', 'd_numerator', 'd_denom', 'prev_d_numerator', 'max_d_numerator']:
                        group[key] = global_group[key]

            # Update other group values unrelated to d.
            for group in self.param_groups:
                group['weight_sum'] = group.get('running_weight_sum', 0)
                group['k'] += 1

    def get_dlr(self, group):
        shared_d = group.get('shared_d', None)
        dlr = (shared_d if group['split_groups'] and group['split_groups_mean'] and shared_d is not None else group['d']) * group['lr']
        return dlr

    def update_prodigy(self, state, group, grad, data):
        k = group['k']
        prodigy_steps = group['prodigy_steps']

        if prodigy_steps <= 0 or k < prodigy_steps:
            d_update = (group['d'] ** 2) / (group['d0'] ** 0.5)

            sliced_grad, sliced_data = self.get_sliced_tensor(grad), self.get_sliced_tensor(data)
            running_d_numerator, running_d_denom = self.get_running_values_for_group(group)

            x0_minus = state['p0'] - sliced_data
            x0_dot = torch.dot(sliced_grad, x0_minus)

            if group['use_speed']:
                beta = 1 - (k ** -0.75)
                x0_dot = state.get('exp_avg_x0_dot', x0_dot) * beta + x0_dot * (1 - beta)
                state['exp_avg_x0_dot'] = x0_dot.item()

                # Always normalise by largest displacement. This is mainly to keep the numerator stable
                # when weight decay is applied.
                x0_minus_l1_norm = x0_minus.abs().sum().clamp_min(state.get('max_x0_minus_l1_norm', 1e-8))
                state['max_x0_minus_l1_norm'] = x0_minus_l1_norm.item()

                x0_minus_l2_norm = x0_minus.norm().clamp_min(state.get('max_x0_minus_l2_norm', 1e-8))
                state['max_x0_minus_l2_norm'] = x0_minus_l2_norm.item()

                # Penalise d as displacement increases. This helps prevent pathologic d growth.
                d_update = d_update ** (group['d_coef'] ** 0.125)
                d_update /= x0_minus_l2_norm.add(1).square()

                # d_denom is unused by SPEED, so use it instead for logging l2.
                running_d_denom.add_(x0_minus_l2_norm)

                # Normalise.
                x0_dot /= x0_minus_l1_norm
            else:
                _, _, beta3 = self.get_betas(group)
                running_d_denom.add_(state['s'].mul_(beta3).add_(sliced_grad, alpha=d_update).abs().sum())

            running_d_numerator.add_(x0_dot, alpha=d_update)

            del x0_minus
        else:
            # Free the memory used by Prodigy, as we no longer need it.
            if 's' in state:
                s = state.pop('s')
                del s
            if 'p0' in state:
                p0 = state.pop('p0')
                del p0

    def update_(self, num, denom, group, w):
        if group['use_focus']:
            # FOCUS: First Order Concentrated Updating Scheme: https://arxiv.org/pdf/2501.12243
            gamma = 0.1

            # Original form.
            # update = torch.sign(num) + gamma * torch.sign(w - denom)

            denom = denom.sub_(w).sign_().mul_(-gamma)
            update = num.sign_().add_(denom)
        else:
            d, eps = group['d'], group['eps']

            if eps is None:
                # Adam-atan2. Use atan2 rather than epsilon and division 
                # for parameter updates (https://arxiv.org/abs/2407.05872).
                # Has the nice property of "clipping" the gradient as well.

                # b = 1, a = 1 / arctan(1) = ~1.27. Page 35 of paper. This
                # clips most updates to ~2.0
                a = 1.2732395
                update = num.atan2_(denom).mul_(a)
            else:
                update = num.div_(denom.add_(d * eps))

        return update

    def get_denom(self, state, group):
        if group['use_focus']:
            denom = state['exp_avg_sq'].clone()
        elif 'exp_avg_sq_metadata' in state:
            row_var, col_var = state["exp_avg_sq_row"], state["exp_avg_sq_col"]
            dr, dc = state["exp_avg_sq_metadata"]

            reduce_dc = dc - 1 if dc > dr else dc
            row_col_mean = row_var.mean(dim=reduce_dc, keepdim=True).add_(1e-30)
            denom = (row_var.div(row_col_mean) * col_var).sqrt_()
        else:
            denom = state['exp_avg_sq'].sqrt()

        return denom
   
    def update_first_moment(self, state, group, grad, beta1):
        d_k, exp_avg = group['d'], state['exp_avg']

        return exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1) * d_k)

    def update_second_moment(self, state, group, grad, beta2, w, return_denom=True, denom_before_update=False):
        d_k, denom = group['d'] ** 2, None

        if return_denom and denom_before_update:
            denom = self.get_denom(state, group)

        # Adam EMA updates
        if group['use_focus']:
            state['exp_avg_sq'].mul_(beta2).add_(w, alpha=(1 - beta2) * d_k)
        elif 'exp_avg_sq_metadata' in state:
            row_var, col_var = state["exp_avg_sq_row"], state["exp_avg_sq_col"]
            dr, dc = state["exp_avg_sq_metadata"]

            row_var.mul_(beta2).add_(
                grad.norm(dim=dr, keepdim=True).square_().mul_(1 / grad.shape[dr]),
                alpha=(1 - beta2) * d_k
            )
            col_var.mul_(beta2).add_(
                grad.norm(dim=dc, keepdim=True).square_().mul_(1 / grad.shape[dc]),
                alpha=(1 - beta2) * d_k
            )
        else:
            state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=(1 - beta2) * d_k)

        if return_denom and denom is None:
            denom = self.get_denom(state, group)

        return denom

    def get_rms(self, tensor, eps=1e-8):
        return tensor.norm(2).div(tensor.numel() ** 0.5).clamp_min(eps)

    def rms_clip_(self, tensor):
        return tensor.mul_(self.get_rms(tensor, 1.0).rsqrt())

    def try_hook_kohya_fbp(self):
        self.kohya_original_patch_adafactor_fused = None

        try:
            # Import and patching will fail if not Kohya.
            import library.adafactor_fused

            # Get the original method so we can restore it later.
            self.kohya_original_patch_adafactor_fused = library.adafactor_fused.patch_adafactor_fused

            # Define the override.
            def prodigy_patch_adafactor_fused(optimizer):
                unwrapped_optimiser = None
                if hasattr(optimizer, "optimizer"):
                    # If the optimiser is wrapped, forward the calls to the actual optimiser.
                    def _step(self, *args, **kwargs):
                        return self.optimizer.step(*args, **kwargs)

                    def _step_param(self, *args, **kwargs):
                        return self.optimizer.step_param(*args, **kwargs)

                    optimizer.step = _step.__get__(optimizer)
                    optimizer.step_param = _step_param.__get__(optimizer)
                    unwrapped_optimiser = optimizer.optimizer
                else:
                    unwrapped_optimiser = optimizer
               
                self.log(f"Kohya pipeline detected with fused backward pass. Gradient hook patch successful.")
                library.adafactor_fused.patch_adafactor_fused = unwrapped_optimiser.kohya_original_patch_adafactor_fused # Restore the original method.

                unwrapped_optimiser.fused_back_pass = True
                unwrapped_optimiser.kohya_original_patch_adafactor_fused = None

            # Patch the method.
            library.adafactor_fused.patch_adafactor_fused = prodigy_patch_adafactor_fused
        except:
            pass

    def try_unhook_kohya_fbp(self):
        if self.kohya_original_patch_adafactor_fused is None:
            return

        try:
            # Import and patching will fail if not Kohya.
            import library.adafactor_fused

            # User did not opt for fused backward pass, so remove our hook.
            library.adafactor_fused.patch_adafactor_fused = self.kohya_original_patch_adafactor_fused
        except:
            pass

        self.kohya_original_patch_adafactor_fused = None

    @torch.no_grad()
    def step_param(self, p, group):
        raise Exception("Not implemented!")            

    @torch.no_grad()
    def step_parameter(self, p, group, i):
        self.step_param(p, group)

    @torch.no_grad()
    def step(self, closure=None):
        self.try_unhook_kohya_fbp()

        if self.get_global_group()['fused_back_pass']:
            return

        """Performs a single optimisation step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for param_group in self.param_groups:
            for p in param_group["params"]:
                self.step_param(p, param_group)

        return loss