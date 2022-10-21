# Cambricon PyTorch Model Migration Report
## Cambricon PyTorch Changes
| No. |  File  |  Description  |
| 1 | avg_checkpoints.py:14 | add "import torch_mlu" |
| 2 | benchmark.py:14 | add "import torch_mlu" |
| 3 | benchmark.py:36 | change "if getattr(torch.cuda.amp, 'autocast') is not None:" to "if getattr(torch.mlu.amp, 'autocast') is not None: " |
| 4 | benchmark.py:133 | change "def cuda_timestamp(sync=False, device=None):" to "def mlu_timestamp(sync=False, device=None): " |
| 5 | benchmark.py:135 | change "torch.cuda.synchronize(device=device)" to "torch.mlu.synchronize(device=device) " |
| 6 | benchmark.py:188 | change "self, model_name, detail=False, device='cuda', torchscript=False, precision='float32'," to "self, model_name, detail=False, device='mlu', torchscript=False, precision='float32', " |
| 7 | benchmark.py:195 | change "self.amp_autocast = torch.cuda.amp.autocast if self.use_amp else suppress" to "self.amp_autocast = torch.mlu.amp.autocast if self.use_amp else suppress " |
| 8 | benchmark.py:223 | change "if 'cuda' in self.device:" to "if 'mlu' in self.device: " |
| 9 | benchmark.py:224 | change "self.time_fn = partial(cuda_timestamp, device=self.device)" to "self.time_fn = partial(mlu_timestamp, device=self.device) " |
| 10 | benchmark.py:237 | change "def __init__(self, model_name, device='cuda', torchscript=False, **kwargs):" to "def __init__(self, model_name, device='mlu', torchscript=False, **kwargs): " |
| 11 | benchmark.py:306 | change "def __init__(self, model_name, device='cuda', torchscript=False, **kwargs):" to "def __init__(self, model_name, device='mlu', torchscript=False, **kwargs): " |
| 12 | benchmark.py:426 | change "def __init__(self, model_name, device='cuda', profiler='', **kwargs):" to "def __init__(self, model_name, device='mlu', profiler='', **kwargs): " |
| 13 | benchmark.py:477 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 14 | clean_checkpoint.py:10 | add "import torch_mlu" |
| 15 | inference.py:13 | add "import torch_mlu" |
| 16 | inference.py:81 | change "model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()" to "model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).mlu() " |
| 17 | inference.py:83 | change "model = model.cuda()" to "model = model.mlu() " |
| 18 | inference.py:104 | change "input = input.cuda()" to "input = input.mlu() " |
| 19 | train.py:26 | add "import torch_mlu" |
| 20 | train.py:50 | change "if getattr(torch.cuda.amp, 'autocast') is not None:" to "if getattr(torch.mlu.amp, 'autocast') is not None: " |
| 21 | train.py:334 | change "args.device = 'cuda:0'" to "args.device = 'mlu:0' " |
| 22 | train.py:338 | change "args.device = 'cuda:%d' % args.local_rank" to "args.device = 'mlu:%d' % args.local_rank " |
| 23 | train.py:339 | change "torch.cuda.set_device(args.local_rank)" to "torch.mlu.set_device(args.local_rank) " |
| 24 | train.py:340 | change "torch.distributed.init_process_group(backend='nccl', init_method='env://')" to "torch.distributed.init_process_group(backend='cncl', init_method='env://') " |
| 25 | train.py:402 | change "model.cuda()" to "model.mlu() " |
| 26 | train.py:435 | change "amp_autocast = torch.cuda.amp.autocast" to "amp_autocast = torch.mlu.amp.autocast " |
| 27 | train.py:455 | change "# Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper" to "# Important to create EMA model after mlu(), DP wrapper, and AMP but before SyncBN and DDP wrapper " |
| 28 | train.py:586 | change "train_loss_fn = train_loss_fn.cuda()" to "train_loss_fn = train_loss_fn.mlu() " |
| 29 | train.py:587 | change "validate_loss_fn = nn.CrossEntropyLoss().cuda()" to "validate_loss_fn = nn.CrossEntropyLoss().mlu() " |
| 30 | train.py:681 | change "input, target = input.cuda(), target.cuda()" to "input, target = input.mlu(), target.mlu() " |
| 31 | train.py:712 | change "torch.cuda.synchronize()" to "torch.mlu.synchronize() " |
| 32 | train.py:778 | change "input = input.cuda()" to "input = input.mlu() " |
| 33 | train.py:779 | change "target = target.cuda()" to "target = target.mlu() " |
| 34 | train.py:804 | change "torch.cuda.synchronize()" to "torch.mlu.synchronize() " |
| 35 | validate.py:16 | add "import torch_mlu" |
| 36 | validate.py:35 | change "if getattr(torch.cuda.amp, 'autocast') is not None:" to "if getattr(torch.mlu.amp, 'autocast') is not None: " |
| 37 | validate.py:129 | change "amp_autocast = torch.cuda.amp.autocast" to "amp_autocast = torch.mlu.amp.autocast " |
| 38 | validate.py:166 | change "model = model.cuda()" to "model = model.mlu() " |
| 39 | validate.py:176 | change "criterion = nn.CrossEntropyLoss().cuda()" to "criterion = nn.CrossEntropyLoss().mlu() " |
| 40 | validate.py:216 | change "input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).cuda()" to "input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).mlu() " |
| 41 | validate.py:223 | change "target = target.cuda()" to "target = target.mlu() " |
| 42 | validate.py:224 | change "input = input.cuda()" to "input = input.mlu() " |
| 43 | validate.py:319 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 44 | convert/convert_from_mxnet.py:7 | add "import torch_mlu" |
| 45 | convert/convert_nest_flax.py:9 | add "import torch_mlu" |
| 46 | tests/test_layers.py:2 | add "import torch_mlu" |
| 47 | tests/test_models.py:2 | add "import torch_mlu" |
| 48 | tests/test_optim.py:11 | add "import torch_mlu" |
| 49 | tests/test_optim.py:38 | change "if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():" to "if y.is_mlu and bias.is_mlu and y.get_device() != bias.get_device(): " |
| 50 | tests/test_optim.py:39 | change "y = y.cuda(bias.get_device())" to "y = y.mlu(bias.get_device()) " |
| 51 | tests/test_optim.py:64 | change "i = input_cuda if weight.is_cuda else input" to "i = input_mlu if weight.is_mlu else input " |
| 52 | tests/test_optim.py:103 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |
| 53 | tests/test_optim.py:106 | change "input_cuda = Variable(input.data.float().cuda())" to "input_mlu = Variable(input.data.float().mlu()) " |
| 54 | tests/test_optim.py:107 | change "weight_cuda = Variable(weight.data.float().cuda(), requires_grad=True)" to "weight_mlu = Variable(weight.data.float().mlu(), requires_grad=True) " |
| 55 | tests/test_optim.py:108 | change "bias_cuda = Variable(bias.data.float().cuda(), requires_grad=True)" to "bias_mlu = Variable(bias.data.float().mlu(), requires_grad=True) " |
| 56 | tests/test_optim.py:109 | change "optimizer_cuda = constructor(weight_cuda, bias_cuda)" to "optimizer_mlu = constructor(weight_mlu, bias_mlu) " |
| 57 | tests/test_optim.py:110 | change "fn_cuda = functools.partial(fn_base, optimizer_cuda, weight_cuda, bias_cuda)" to "fn_mlu = functools.partial(fn_base, optimizer_mlu, weight_mlu, bias_mlu) " |
| 58 | tests/test_optim.py:114 | change "optimizer_cuda.load_state_dict(state_dict_c)" to "optimizer_mlu.load_state_dict(state_dict_c) " |
| 59 | tests/test_optim.py:121 | change "optimizer_cuda.step(fn_cuda)" to "optimizer_mlu.step(fn_mlu) " |
| 60 | tests/test_optim.py:122 | change "torch_tc.assertEqual(weight, weight_cuda)" to "torch_tc.assertEqual(weight, weight_mlu) " |
| 61 | tests/test_optim.py:123 | change "torch_tc.assertEqual(bias, bias_cuda)" to "torch_tc.assertEqual(bias, bias_mlu) " |
| 62 | tests/test_optim.py:157 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |
| 63 | tests/test_optim.py:160 | change "torch.randn(10, 5).cuda()," to "torch.randn(10, 5).mlu(), " |
| 64 | tests/test_optim.py:161 | change "torch.randn(10).cuda()," to "torch.randn(10).mlu(), " |
| 65 | tests/test_optim.py:162 | change "torch.randn(5).cuda()," to "torch.randn(5).mlu(), " |
| 66 | timm/data/dataset.py:7 | add "import torch_mlu" |
| 67 | timm/data/distributed_sampler.py:2 | add "import torch_mlu" |
| 68 | timm/data/loader.py:69 | change "self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)" to "self.mean = torch.tensor([x * 255 for x in mean]).mlu().view(1, 3, 1, 1) " |
| 69 | timm/data/loader.py:70 | change "self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)" to "self.std = torch.tensor([x * 255 for x in std]).mlu().view(1, 3, 1, 1) " |
| 70 | timm/data/loader.py:82 | change "stream = torch.cuda.Stream()" to "stream = torch.mlu.Stream() " |
| 71 | timm/data/loader.py:86 | change "with torch.cuda.stream(stream):" to "with torch.mlu.stream(stream): " |
| 72 | timm/data/loader.py:87 | change "next_input = next_input.cuda(non_blocking=True)" to "next_input = next_input.mlu(non_blocking=True) " |
| 73 | timm/data/loader.py:88 | change "next_target = next_target.cuda(non_blocking=True)" to "next_target = next_target.mlu(non_blocking=True) " |
| 74 | timm/data/loader.py:101 | change "torch.cuda.current_stream().wait_stream(stream)" to "torch.mlu.current_stream().wait_stream(stream) " |
| 75 | timm/data/mixup.py:14 | add "import torch_mlu" |
| 76 | timm/data/mixup.py:17 | change "def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):" to "def one_hot(x, num_classes, on_value=1., off_value=0., device='mlu'): " |
| 77 | timm/data/mixup.py:22 | change "def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):" to "def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='mlu'): " |
| 78 | timm/data/random_erasing.py:10 | add "import torch_mlu" |
| 79 | timm/data/random_erasing.py:13 | change "def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):" to "def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='mlu'): " |
| 80 | timm/data/random_erasing.py:48 | change "mode='const', min_count=1, max_count=None, num_splits=0, device='cuda'):" to "mode='const', min_count=1, max_count=None, num_splits=0, device='mlu'): " |
| 81 | timm/data/transforms.py:1 | add "import torch_mlu" |
| 82 | timm/data/transforms_factory.py:8 | add "import torch_mlu" |
| 83 | timm/data/parsers/parser_tfds.py:10 | add "import torch_mlu" |
| 84 | timm/loss/asymmetric_loss.py:1 | add "import torch_mlu" |
| 85 | timm/loss/binary_cross_entropy.py:7 | add "import torch_mlu" |
| 86 | timm/loss/cross_entropy.py:6 | add "import torch_mlu" |
| 87 | timm/loss/jsd.py:1 | add "import torch_mlu" |
| 88 | timm/models/beit.py:25 | add "import torch_mlu" |
| 89 | timm/models/byobnet.py:32 | add "import torch_mlu" |
| 90 | timm/models/cait.py:13 | add "import torch_mlu" |
| 91 | timm/models/coat.py:14 | add "import torch_mlu" |
| 92 | timm/models/convit.py:25 | add "import torch_mlu" |
| 93 | timm/models/convit.py:172 | change "distances = distances.to('cuda')" to "distances = distances.to('mlu') " |
| 94 | timm/models/convnext.py:15 | add "import torch_mlu" |
| 95 | timm/models/crossvit.py:29 | add "import torch_mlu" |
| 96 | timm/models/cspnet.py:15 | add "import torch_mlu" |
| 97 | timm/models/densenet.py:9 | add "import torch_mlu" |
| 98 | timm/models/dla.py:10 | add "import torch_mlu" |
| 99 | timm/models/dpn.py:13 | add "import torch_mlu" |
| 100 | timm/models/efficientnet.py:41 | add "import torch_mlu" |
| 101 | timm/models/efficientnet_blocks.py:6 | add "import torch_mlu" |
| 102 | timm/models/features.py:16 | add "import torch_mlu" |
| 103 | timm/models/ghostnet.py:10 | add "import torch_mlu" |
| 104 | timm/models/helpers.py:12 | add "import torch_mlu" |
| 105 | timm/models/hrnet.py:14 | add "import torch_mlu" |
| 106 | timm/models/hub.py:8 | add "import torch_mlu" |
| 107 | timm/models/inception_resnet_v2.py:5 | add "import torch_mlu" |
| 108 | timm/models/inception_v3.py:6 | add "import torch_mlu" |
| 109 | timm/models/inception_v4.py:5 | add "import torch_mlu" |
| 110 | timm/models/levit.py:31 | add "import torch_mlu" |
| 111 | timm/models/mlp_mixer.py:45 | add "import torch_mlu" |
| 112 | timm/models/mobilenetv3.py:12 | add "import torch_mlu" |
| 113 | timm/models/nasnet.py:7 | add "import torch_mlu" |
| 114 | timm/models/nest.py:23 | add "import torch_mlu" |
| 115 | timm/models/nfnet.py:25 | add "import torch_mlu" |
| 116 | timm/models/pit.py:20 | add "import torch_mlu" |
| 117 | timm/models/pnasnet.py:11 | add "import torch_mlu" |
| 118 | timm/models/res2net.py:7 | add "import torch_mlu" |
| 119 | timm/models/resnest.py:9 | add "import torch_mlu" |
| 120 | timm/models/resnet.py:13 | add "import torch_mlu" |
| 121 | timm/models/resnetv2.py:34 | add "import torch_mlu" |
| 122 | timm/models/rexnet.py:13 | add "import torch_mlu" |
| 123 | timm/models/selecsls.py:14 | add "import torch_mlu" |
| 124 | timm/models/swin_transformer.py:20 | add "import torch_mlu" |
| 125 | timm/models/tnt.py:10 | add "import torch_mlu" |
| 126 | timm/models/tresnet.py:10 | add "import torch_mlu" |
| 127 | timm/models/twins.py:18 | add "import torch_mlu" |
| 128 | timm/models/vgg.py:8 | add "import torch_mlu" |
| 129 | timm/models/visformer.py:11 | add "import torch_mlu" |
| 130 | timm/models/vision_transformer.py:31 | add "import torch_mlu" |
| 131 | timm/models/vision_transformer_hybrid.py:19 | add "import torch_mlu" |
| 132 | timm/models/vovnet.py:16 | add "import torch_mlu" |
| 133 | timm/models/xcit.py:17 | add "import torch_mlu" |
| 134 | timm/models/layers/activations.py:9 | add "import torch_mlu" |
| 135 | timm/models/layers/activations_jit.py:13 | add "import torch_mlu" |
| 136 | timm/models/layers/activations_me.py:12 | add "import torch_mlu" |
| 137 | timm/models/layers/adaptive_avgmax_pool.py:12 | add "import torch_mlu" |
| 138 | timm/models/layers/attention_pool2d.py:13 | add "import torch_mlu" |
| 139 | timm/models/layers/blur_pool.py:9 | add "import torch_mlu" |
| 140 | timm/models/layers/bottleneck_attn.py:19 | add "import torch_mlu" |
| 141 | timm/models/layers/cbam.py:10 | add "import torch_mlu" |
| 142 | timm/models/layers/cond_conv2d.py:12 | add "import torch_mlu" |
| 143 | timm/models/layers/conv2d_same.py:5 | add "import torch_mlu" |
| 144 | timm/models/layers/create_attn.py:5 | add "import torch_mlu" |
| 145 | timm/models/layers/create_norm_act.py:12 | add "import torch_mlu" |
| 146 | timm/models/layers/drop.py:17 | add "import torch_mlu" |
| 147 | timm/models/layers/evo_norm.py:12 | add "import torch_mlu" |
| 148 | timm/models/layers/halo_attn.py:21 | add "import torch_mlu" |
| 149 | timm/models/layers/inplace_abn.py:1 | add "import torch_mlu" |
| 150 | timm/models/layers/lambda_layer.py:23 | add "import torch_mlu" |
| 151 | timm/models/layers/linear.py:3 | add "import torch_mlu" |
| 152 | timm/models/layers/mixed_conv2d.py:8 | add "import torch_mlu" |
| 153 | timm/models/layers/non_local_attn.py:7 | add "import torch_mlu" |
| 154 | timm/models/layers/norm.py:3 | add "import torch_mlu" |
| 155 | timm/models/layers/norm_act.py:3 | add "import torch_mlu" |
| 156 | timm/models/layers/selective_kernel.py:7 | add "import torch_mlu" |
| 157 | timm/models/layers/pool2d_same.py:5 | add "import torch_mlu" |
| 158 | timm/models/layers/split_attn.py:9 | add "import torch_mlu" |
| 159 | timm/models/layers/weight_init.py:1 | add "import torch_mlu" |
| 160 | timm/models/layers/space_to_depth.py:1 | add "import torch_mlu" |
| 161 | timm/models/layers/split_batchnorm.py:14 | add "import torch_mlu" |
| 162 | timm/models/layers/std_conv.py:19 | add "import torch_mlu" |
| 163 | timm/optim/adabelief.py:2 | add "import torch_mlu" |
| 164 | timm/optim/adafactor.py:12 | add "import torch_mlu" |
| 165 | timm/optim/adahessian.py:6 | add "import torch_mlu" |
| 166 | timm/optim/adamp.py:11 | add "import torch_mlu" |
| 167 | timm/optim/adamw.py:8 | add "import torch_mlu" |
| 168 | timm/optim/lamb.py:56 | add "import torch_mlu" |
| 169 | timm/optim/lars.py:13 | add "import torch_mlu" |
| 170 | timm/optim/lookahead.py:7 | add "import torch_mlu" |
| 171 | timm/optim/madgrad.py:15 | add "import torch_mlu" |
| 172 | timm/optim/nadam.py:3 | add "import torch_mlu" |
| 173 | timm/optim/nvnovograd.py:8 | add "import torch_mlu" |
| 174 | timm/optim/optim_factory.py:6 | add "import torch_mlu" |
| 175 | timm/optim/optim_factory.py:120 | change "assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'" to "assert has_apex and torch.mlu.is_available(), 'APEX and CUDA required for fused optimizers' " |
| 176 | timm/optim/radam.py:6 | add "import torch_mlu" |
| 177 | timm/optim/rmsprop_tf.py:10 | add "import torch_mlu" |
| 178 | timm/optim/sgdp.py:11 | add "import torch_mlu" |
| 179 | timm/scheduler/cosine_lr.py:10 | add "import torch_mlu" |
| 180 | timm/scheduler/multistep_lr.py:5 | add "import torch_mlu" |
| 181 | timm/scheduler/plateau_lr.py:7 | add "import torch_mlu" |
| 182 | timm/scheduler/poly_lr.py:10 | add "import torch_mlu" |
| 183 | timm/scheduler/scheduler.py:3 | add "import torch_mlu" |
| 184 | timm/scheduler/step_lr.py:8 | add "import torch_mlu" |
| 185 | timm/scheduler/tanh_lr.py:10 | add "import torch_mlu" |
| 186 | timm/utils/__init__.py:4 | change "from .cuda import ApexScaler, NativeScaler" to "from .mlu import ApexScaler, NativeScaler " |
| 187 | timm/utils/agc.py:18 | add "import torch_mlu" |
| 188 | timm/utils/checkpoint_saver.py:13 | add "import torch_mlu" |
| 189 | timm/utils/clip_grad.py:1 | add "import torch_mlu" |
| 190 | timm/utils/cuda.py:5 | add "import torch_mlu" |
| 191 | timm/utils/cuda.py:40 | change "self._scaler = torch.cuda.amp.GradScaler()" to "self._scaler = torch.mlu.amp.GradScaler() " |
| 192 | timm/utils/jit.py:5 | add "import torch_mlu" |
| 193 | timm/utils/distributed.py:5 | add "import torch_mlu" |
| 194 | timm/utils/model.py:7 | add "import torch_mlu" |
| 195 | timm/utils/model_ema.py:9 | add "import torch_mlu" |
| 196 | timm/utils/random.py:3 | add "import torch_mlu" |
