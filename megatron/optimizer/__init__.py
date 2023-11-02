# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD

from megatron import get_args

from .distrib_optimizer import DistributedOptimizer
from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer

from collections import defaultdict

def get_param_groups(modules,
                     no_weight_decay_cond,
                     scale_lr_cond,
                     lr_mult):
    """creates param groups based on weight decay condition (regularized vs non regularized)
       and learning rate scale condition (args.lr vs lr_mult * args.lr)
       scale_lr_cond is used during finetuning where head of the network requires a scaled
       version of the base learning rate. 
    """
    wd_no_scale_lr = []
    wd_scale_lr = []
    no_wd_no_scale_lr = []
    no_wd_scale_lr = []
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # do not regularize biases nor Norm parameters
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_no_scale_lr.append(param)
            elif not no_wd and scale_lr:
                wd_scale_lr.append(param)
            elif no_wd and not scale_lr:
                no_wd_no_scale_lr.append(param)
            else:
                no_wd_scale_lr.append(param)

    param_groups = []
    if len(wd_no_scale_lr):
        param_groups.append({'params': wd_no_scale_lr, 'wd_mult': 1.0, 'lr_mult': 1.0})
    if len(wd_scale_lr):
        param_groups.append({'params': wd_scale_lr, 'wd_mult': 1.0, 'lr_mult': lr_mult})
    if len(no_wd_no_scale_lr):
        param_groups.append({'params': no_wd_no_scale_lr, 'wd_mult': 0.0, 'lr_mult': 1.0})
    if len(no_wd_scale_lr):
        param_groups.append({'params': no_wd_scale_lr, 'wd_mult': 0.0, 'lr_mult': lr_mult})

    return param_groups


def add_mup_multipliers(param_groups, optimizer_type, decoupled_wd=False):
    """
    Reimplementation of MuAdam and MuSGD optimisers from the mup library (https://github.com/microsoft/mup/blob/main/mup/optim.py) 
    
    Adds to the param_groups the `mup_lr_mult` and `mup_wd_mult` multipliers which rescale learning rate and weight decay 
        according to the model size scaling. The multipliers are to be applied in the step() function of OptimizerParamScheduler().
    """
    new_param_groups = []

    if optimizer_type == 'adam' or optimizer_type == 'adan':
        for param_group in param_groups:

            # new group template with a copy of everything except params
            def new_group():
                new_g = {k:v for k, v in param_group.items() if k != 'params'} 
                new_g['params'] = []
                return new_g

            # create separate groups for matrix- and vector-like params
            matrix_like_p = defaultdict(new_group) # key is width_mult
            vector_like_p = new_group()

            # split params in the current group based on the infshape dimension 
            for p in param_group['params']:
                assert hasattr(p, 'infshape'), (
                    f'A parameter with shape {p.shape} does not have `infshape` attribute. '
                    'Did you forget to call `mup.set_base_shapes` on the model?')
                if p.infshape.ninf() == 2:
                    matrix_like_p[p.infshape.width_mult()]['params'].append(p)
                elif p.infshape.ninf() > 2:
                    raise NotImplementedError('more than 2 inf dimensions')
                else:
                    vector_like_p['params'].append(p)

            # for matrix-like params, add LR multiplier 1/width_mult (Table 8, row 4 in the muP paper) 
            # and WD multiplier if optimiser is not WD decoupled (see https://github.com/microsoft/mup/issues/1)
            for width_mult, group in matrix_like_p.items():
                group['mup_lr_mult'] = 1/width_mult
                if not decoupled_wd: 
                    group['mup_wd_mult'] = width_mult
                else:
                    group['mup_wd_mult'] = 1.0

            # for vector-like and scalar params, no multipliers
            vector_like_p['mup_lr_mult'] = 1.0
            vector_like_p['mup_wd_mult'] = 1.0

            # add to new param groups
            new_param_groups.extend(list(matrix_like_p.values()) + [vector_like_p])

    elif optimizer_type == 'sgd':
        for param_group in param_groups:

            # new group template with a copy of everything except params
            def new_group():
                new_g = {k:v for k, v in param_group.items() if k != 'params'} 
                new_g['params'] = []
                return new_g

            vector_like_p = defaultdict(new_group) # key is width mult
            matrix_like_p = defaultdict(new_group) # key is fan_in/out ratio
            fixed_p = new_group()

            # split parameters based on the infshape dimension 
            for p in param_group['params']:
                assert hasattr(p, 'infshape'), (
                    f'A parameter with shape {p.shape} does not have `infshape` attribute. '
                    'Did you forget to call `mup.set_base_shapes` on the model?')
                if p.infshape.ninf() == 1:
                    vector_like_p[p.infshape.width_mult()]['params'].append(p)
                elif p.infshape.ninf() == 2:
                    matrix_like_p[p.infshape.fanin_fanout_mult_ratio()]['params'].append(p) # keep original implementation, according to the paper should always equal 1 
                elif p.infshape.ninf() > 2:
                    raise NotImplementedError('more than 2 inf dimensions')
                else:
                    fixed_p['params'].append(p)

            # for vector-like params, add LR multiplier width_mult (Table 8, row 3 in the muP paper)
            for width_mult, group in vector_like_p.items():
                group['mup_lr_mult'] = width_mult
                if not decoupled_wd:
                    group['mup_wd_mult'] = 1/width_mult
                else:
                    group['mup_wd_mult'] = 1.0

            # for matrix-like params, add LR multiplier of 1/shape_ratio == 1 (Table 8, row 3 in the muP paper)
            for shape_ratio, group in matrix_like_p.items():
                assert shape_ratio == 1.0, 'fanin_fanout_mult_ratio() is not equal to 1'
                group['mup_lr_mult'] = 1/shape_ratio
                if not decoupled_wd:
                    group['mup_wd_mult'] = shape_ratio
                else:
                    group['mup_wd_mult'] = 1.0

            # for scalar params, no multipliers
            fixed_p['mup_lr_mult'] = 1.0
            fixed_p['mup_wd_mult'] = 1.0

            new_param_groups.extend(list(matrix_like_p.values()) + \
                                    list(vector_like_p.values()) + [fixed_p])
    else:
        raise NotImplementedError(f'optimizer type {optimizer_type} not supported for muP')

    return new_param_groups


def get_megatron_optimizer(model,
                           no_weight_decay_cond=None,
                           scale_lr_cond=None,
                           lr_mult=1.0,
                           use_mup=False):
    args = get_args()

    # Base optimizer.
    param_groups = get_param_groups(model,
                                    no_weight_decay_cond,
                                    scale_lr_cond,
                                    lr_mult)

    if use_mup:
        if args.optimizer == 'adam' and args.weight_decay != 0:
            print('[NB] weight decay parameter is not transferable with muP for the Adam optimizer! \
                    If FusedAdam with the default configuration is used, it corresponds to AdamW which is muP-supported.')
        param_groups = add_mup_multipliers(param_groups, args.optimizer)
        
    if args.optimizer == 'adam':
        optimizer = Adam(param_groups,
                         lr=args.lr,
                         weight_decay=args.weight_decay,
                         betas=(args.adam_beta1, args.adam_beta2),
                         eps=args.adam_eps)
    elif args.optimizer == 'sgd':
        optimizer = SGD(param_groups,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        momentum=args.sgd_momentum)
    elif args.optimizer == 'adan':
        try:
            from adan import Adan
        except ImportError:
            raise ImportError(
                'to use the Adan optimizer, please execute '
                '`python3 -m pip install '
                'git+https://github.com/sail-sg/Adan.git`'
            )
        optimizer = Adan(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adan_beta1, args.adan_beta2, args.adan_beta3),
            eps=args.adam_eps,
            no_prox=args.adan_no_prox,
            foreach=args.adan_foreach,
            fused=args.adan_fused,
        )
    else:
        raise Exception('{} optimizer is not supported.'.format(
            args.optimizer))

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == 'local':
        params_have_main_grad = True

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if args.fp16 or args.bf16 or args.use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)

        # Dynamic loss scale.
        else:
            if args.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=args.initial_loss_scale,
                    min_scale=args.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=args.loss_scale_window,
                    hysteresis=args.hysteresis)

        # Megatron optimizer.
        opt_ty = DistributedOptimizer \
            if args.use_distributed_optimizer else \
            Float16OptimizerWithFloat16Params
        return opt_ty(optimizer,
                      args.clip_grad,
                      args.log_num_zeros_in_grad,
                      params_have_main_grad,
                      args.use_contiguous_buffers_in_local_ddp,
                      args.fp16,
                      args.bf16,
                      args.params_dtype,
                      grad_scaler,
                      model)

    # FP32.
    return FP32Optimizer(optimizer, args.clip_grad,
                         args.log_num_zeros_in_grad,
                         params_have_main_grad,
                         args.use_contiguous_buffers_in_local_ddp,
                         model)
