from collections import defaultdict
REGISTRY = defaultdict(dict)

def register_hyper_para(domain, task):
    def register_hyper_para4domain_task(fn):
        if task in REGISTRY[domain]:
            raise ValueError('can not register hyper-parameter twice for {} {}'.format(domain, task))
        REGISTRY[domain][task] = fn
        return fn

    return register_hyper_para4domain_task

def update_args(args):
    if args.task_name in REGISTRY[args.domain_name]:
        REGISTRY[args.domain_name][args.task_name](args)
    else:
        default_hyper_para(args)

@register_hyper_para('default', 'default')
def default_hyper_para(args):
    args.action_repeat = getattr(args, 'action_repeat', 4)
    args.critic_lr = getattr(args, 'critic_lr', 1e-3)
    args.actor_lr = getattr(args, 'actor_lr', 1e-3)
    args.encoder_lr = getattr(args, 'encoder_lr', 1e-3)
    args.batch_size = getattr(args, 'batch_size', 128)
    if args.domain_name == 'cheetah':
        args.batch_size = 512
    args.num_train_steps = getattr(args, 'num_train_steps', 4e6)
    args.num_train_steps = int(args.num_train_steps//args.action_repeat)
    args.adam_warmup_step = getattr(args, 'adam_warmup_step', 6e3)
    args.adam_warmup_step = int(args.adam_warmup_step*4/args.action_repeat)
    args.eval_freq = int(args.eval_freq*4/args.action_repeat)


@register_hyper_para('finger', 'spin')
def finger_spin(args):
    args.action_repeat = getattr(args, 'action_repeat', 2)
    args.num_train_steps = getattr(args, 'num_train_steps', 2e6)
    default_hyper_para(args)

@register_hyper_para('walker', 'walk')
def walker_walk(args):
    finger_spin(args)

@register_hyper_para('cartpole', 'swingup')
def cartpole_swingup(args):
    args.action_repeat = 8
    default_hyper_para(args)

@register_hyper_para('cheetah', 'run')
def cheetah_run(args):
    args.critic_lr = getattr(args, 'critic_lr', 2e-4)
    args.actor_lr = getattr(args, 'actor_lr', 2e-4)
    args.encoder_lr = getattr(args, 'encoder_lr', 2e-4)
    default_hyper_para(args)
