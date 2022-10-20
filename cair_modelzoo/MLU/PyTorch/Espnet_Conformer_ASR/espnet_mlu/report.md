# Cambricon PyTorch Model Migration Report
## Cambricon PyTorch Changes
| No. |  File  |  Description  |
| ---- |  ----  |  ----  |
| 1 | egs2/TEMPLATE/asr1/pyscripts/utils/extract_xvectors.py:13 | add "import torch_mlu" |
| 2 | egs2/TEMPLATE/asr1/pyscripts/utils/extract_xvectors.py:33 | change "parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")" to "parser.add_argument("--device", type=str, default="mlu:0", help="Inference device") " |
| 3 | egs2/TEMPLATE/asr1/pyscripts/utils/extract_xvectors.py:62 | change "if torch.cuda.is_available() and ("cuda" in args.device):" to "if torch.mlu.is_available() and ("mlu" in args.device): " |
| 4 | egs2/TEMPLATE/asr1/pyscripts/utils/plot_sinc_filters.py:20 | add "import torch_mlu" |
| 5 | egs2/TEMPLATE/enh1/scripts/utils/calculate_speech_metrics.py:11 | add "import torch_mlu" |
| 6 | egs2/TEMPLATE/ssl1/pyscripts/dump_km_label.py:9 | add "import torch_mlu" |
| 7 | egs2/TEMPLATE/ssl1/pyscripts/feature_loader.py:18 | add "import torch_mlu" |
| 8 | egs2/TEMPLATE/ssl1/pyscripts/feature_loader.py:63 | change "self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")" to "self.device = torch.device("mlu:0" if torch.mlu.is_available() else "cpu") " |
| 9 | egs2/TEMPLATE/ssl1/pyscripts/sklearn_km.py:21 | add "import torch_mlu" |
| 10 | egs2/l3das22/enh1/local/metric.py:16 | add "import torch_mlu" |
| 11 | egs2/lrs2/lipreading1/local/feature_extract/video_processing.py:6 | add "import torch_mlu" |
| 12 | egs2/lrs2/lipreading1/local/feature_extract/video_processing.py:96 | change "device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")" to "device = torch.device("mlu:0" if torch.mlu.is_available() else "cpu") " |
| 13 | egs2/lrs2/lipreading1/local/feature_extract/video_processing.py:117 | change "device="cuda:0" if torch.cuda.is_available() else "cpu"," to "device="mlu:0" if torch.mlu.is_available() else "cpu", " |
| 14 | egs2/lrs2/lipreading1/local/feature_extract/models/pretrained.py:5 | add "import torch_mlu" |
| 15 | espnet/asr/asr_utils.py:13 | add "import torch_mlu" |
| 16 | espnet/asr/chainer_backend/asr.py:57 | change "# check cuda and cudnn availability" to "# check mlu and cudnn availability " |
| 17 | espnet/asr/chainer_backend/asr.py:58 | change "if not chainer.cuda.available:" to "if not chainer.mlu.available: " |
| 18 | espnet/asr/chainer_backend/asr.py:59 | change "logging.warning("cuda is not available")" to "logging.warning("mlu is not available") " |
| 19 | espnet/asr/chainer_backend/asr.py:60 | change "if not chainer.cuda.cudnn_enabled:" to "if not chainer.mlu.cudnn_enabled: " |
| 20 | espnet/asr/chainer_backend/asr.py:109 | change "chainer.cuda.get_device_from_id(gpu_id).use()" to "chainer.mlu.get_device_from_id(gpu_id).use() " |
| 21 | espnet/asr/pytorch_backend/asr.py:14 | add "import torch_mlu" |
| 22 | espnet/asr/pytorch_backend/asr.py:485 | change "backend="nccl"," to "backend="cncl", " |
| 23 | espnet/asr/pytorch_backend/asr.py:503 | change "# check cuda availability" to "# check mlu availability " |
| 24 | espnet/asr/pytorch_backend/asr.py:504 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |
| 25 | espnet/asr/pytorch_backend/asr.py:505 | change "logging.warning("cuda is not available")" to "logging.warning("mlu is not available") " |
| 26 | espnet/asr/pytorch_backend/asr.py:609 | change "device = torch.device(f"cuda:{localrank}")" to "device = torch.device(f"mlu:{localrank}") " |
| 27 | espnet/asr/pytorch_backend/asr.py:611 | change "device = torch.device("cuda" if args.ngpu > 0 else "cpu")" to "device = torch.device("mlu" if args.ngpu > 0 else "cpu") " |
| 28 | espnet/asr/pytorch_backend/asr.py:1167 | change "torch.cuda.synchronize(device=device)" to "torch.mlu.synchronize(device=device) " |
| 29 | espnet/asr/pytorch_backend/asr.py:1282 | change "model.cuda()" to "model.mlu() " |
| 30 | espnet/asr/pytorch_backend/asr.py:1284 | change "rnnlm.cuda()" to "rnnlm.mlu() " |
| 31 | espnet/asr/pytorch_backend/asr.py:1492 | change "model.cuda()" to "model.mlu() " |
| 32 | espnet/asr/pytorch_backend/asr_init.py:8 | add "import torch_mlu" |
| 33 | espnet/asr/pytorch_backend/asr_mix.py:15 | add "import torch_mlu" |
| 34 | espnet/asr/pytorch_backend/asr_mix.py:136 | change "# check cuda availability" to "# check mlu availability " |
| 35 | espnet/asr/pytorch_backend/asr_mix.py:137 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |
| 36 | espnet/asr/pytorch_backend/asr_mix.py:138 | change "logging.warning("cuda is not available")" to "logging.warning("mlu is not available") " |
| 37 | espnet/asr/pytorch_backend/asr_mix.py:205 | change "device = torch.device("cuda" if args.ngpu > 0 else "cpu")" to "device = torch.device("mlu" if args.ngpu > 0 else "cpu") " |
| 38 | espnet/asr/pytorch_backend/asr_mix.py:590 | change "model.cuda()" to "model.mlu() " |
| 39 | espnet/asr/pytorch_backend/asr_mix.py:592 | change "rnnlm.cuda()" to "rnnlm.mlu() " |
| 40 | espnet/asr/pytorch_backend/recog.py:6 | add "import torch_mlu" |
| 41 | espnet/asr/pytorch_backend/recog.py:149 | change "device = "cuda"" to "device = "mlu" " |
| 42 | espnet/bin/asr_align.py:50 | add "import torch_mlu" |
| 43 | espnet/bin/asr_align.py:213 | change "device = "cuda"" to "device = "mlu" " |
| 44 | espnet/bin/asr_align.py:239 | change ":param device: for inference; one of ['cuda', 'cpu']" to ":param device: for inference; one of ['mlu', 'cpu'] " |
| 45 | espnet/lm/chainer_backend/lm.py:219 | change "with chainer.backends.cuda.get_device_from_id(self._device_id):" to "with chainer.backends.mlu.get_device_from_id(self._device_id): " |
| 46 | espnet/lm/chainer_backend/lm.py:227 | change "with chainer.backends.cuda.get_device_from_id(self._device_id):" to "with chainer.backends.mlu.get_device_from_id(self._device_id): " |
| 47 | espnet/lm/chainer_backend/lm.py:271 | change "xp = chainer.backends.cuda.get_array_module(x)" to "xp = chainer.backends.mlu.get_array_module(x) " |
| 48 | espnet/lm/chainer_backend/lm.py:314 | change "xp = chainer.backends.cuda.get_array_module(x)" to "xp = chainer.backends.mlu.get_array_module(x) " |
| 49 | espnet/lm/chainer_backend/lm.py:342 | change "# check cuda and cudnn availability" to "# check mlu and cudnn availability " |
| 50 | espnet/lm/chainer_backend/lm.py:343 | change "if not chainer.cuda.available:" to "if not chainer.mlu.available: " |
| 51 | espnet/lm/chainer_backend/lm.py:344 | change "logging.warning("cuda is not available")" to "logging.warning("mlu is not available") " |
| 52 | espnet/lm/chainer_backend/lm.py:345 | change "if not chainer.cuda.cudnn_enabled:" to "if not chainer.mlu.cudnn_enabled: " |
| 53 | espnet/lm/chainer_backend/lm.py:395 | change "chainer.cuda.get_device_from_id(gpu_id).use()" to "chainer.mlu.get_device_from_id(gpu_id).use() " |
| 54 | espnet/lm/pytorch_backend/extlm.py:9 | add "import torch_mlu" |
| 55 | espnet/lm/pytorch_backend/lm.py:14 | add "import torch_mlu" |
| 56 | espnet/lm/pytorch_backend/lm.py:79 | change "x = x.cuda(device)" to "x = x.mlu(device) " |
| 57 | espnet/lm/pytorch_backend/lm.py:80 | change "t = t.cuda(device)" to "t = t.mlu(device) " |
| 58 | espnet/lm/pytorch_backend/lm.py:221 | change "# check cuda and cudnn availability" to "# check mlu and cudnn availability " |
| 59 | espnet/lm/pytorch_backend/lm.py:222 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |
| 60 | espnet/lm/pytorch_backend/lm.py:223 | change "logging.warning("cuda is not available")" to "logging.warning("mlu is not available") " |
| 61 | espnet/lm/pytorch_backend/lm.py:277 | change "model.to("cuda")" to "model.to("mlu") " |
| 62 | espnet/mt/pytorch_backend/mt.py:15 | add "import torch_mlu" |
| 63 | espnet/mt/pytorch_backend/mt.py:96 | change "# check cuda availability" to "# check mlu availability " |
| 64 | espnet/mt/pytorch_backend/mt.py:97 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |
| 65 | espnet/mt/pytorch_backend/mt.py:98 | change "logging.warning("cuda is not available")" to "logging.warning("mlu is not available") " |
| 66 | espnet/mt/pytorch_backend/mt.py:140 | change "device = torch.device("cuda" if args.ngpu > 0 else "cpu")" to "device = torch.device("mlu" if args.ngpu > 0 else "cpu") " |
| 67 | espnet/mt/pytorch_backend/mt.py:523 | change "model.cuda()" to "model.mlu() " |
| 68 | espnet/nets/batch_beam_search.py:6 | add "import torch_mlu" |
| 69 | espnet/nets/batch_beam_search_online.py:9 | add "import torch_mlu" |
| 70 | espnet/nets/batch_beam_search_online_sim.py:7 | add "import torch_mlu" |
| 71 | espnet/nets/beam_search.py:7 | add "import torch_mlu" |
| 72 | espnet/nets/beam_search_transducer.py:7 | add "import torch_mlu" |
| 73 | espnet/nets/ctc_prefix_score.py:8 | add "import torch_mlu" |
| 74 | espnet/nets/ctc_prefix_score.py:41 | change "torch.device("cuda:%d" % x.get_device())" to "torch.device("mlu:%d" % x.get_device()) " |
| 75 | espnet/nets/ctc_prefix_score.py:42 | change "if x.is_cuda" to "if x.is_mlu " |
| 76 | espnet/nets/scorer_interface.py:6 | add "import torch_mlu" |
| 77 | espnet/nets/transducer_decoder_interface.py:6 | add "import torch_mlu" |
| 78 | espnet/nets/chainer_backend/ctc.py:7 | change "from chainer import cuda" to "from chainer import mlu " |
| 79 | espnet/nets/chainer_backend/ctc.py:135 | change "self.loss = warp_ctc(y_hat, ilens, [cuda.to_cpu(y.data) for y in ys])[0]" to "self.loss = warp_ctc(y_hat, ilens, [mlu.to_cpu(y.data) for y in ys])[0] " |
| 80 | espnet/nets/chainer_backend/deterministic_embed_id.py:6 | change "from chainer import cuda, function_node, link, variable" to "from chainer import mlu, function_node, link, variable " |
| 81 | espnet/nets/chainer_backend/deterministic_embed_id.py:43 | change "xp = cuda.get_array_module(*inputs)" to "xp = mlu.get_array_module(*inputs) " |
| 82 | espnet/nets/chainer_backend/deterministic_embed_id.py:75 | change "xp = cuda.get_array_module(*inputs)" to "xp = mlu.get_array_module(*inputs) " |
| 83 | espnet/nets/chainer_backend/deterministic_embed_id.py:89 | change "# original code based on cuda elementwise method" to "# original code based on mlu elementwise method " |
| 84 | espnet/nets/chainer_backend/deterministic_embed_id.py:91 | change "cuda.elementwise(" to "mlu.elementwise( " |
| 85 | espnet/nets/chainer_backend/deterministic_embed_id.py:98 | change "cuda.elementwise(" to "mlu.elementwise( " |
| 86 | espnet/nets/chainer_backend/deterministic_embed_id.py:122 | change "xp = cuda.get_array_module(*grads)" to "xp = mlu.get_array_module(*grads) " |
| 87 | espnet/nets/chainer_backend/e2e_asr_transformer.py:280 | change "ys_hat = chainer.backends.cuda.to_cpu(self.ctc.argmax(xs).data)" to "ys_hat = chainer.backends.mlu.to_cpu(self.ctc.argmax(xs).data) " |
| 88 | espnet/nets/chainer_backend/rnn/decoders.py:98 | change "with chainer.backends.cuda.get_device_from_id(self._device_id):" to "with chainer.backends.mlu.get_device_from_id(self._device_id): " |
| 89 | espnet/nets/chainer_backend/rnn/decoders.py:106 | change "with chainer.backends.cuda.get_device_from_id(self._device_id):" to "with chainer.backends.mlu.get_device_from_id(self._device_id): " |
| 90 | espnet/nets/chainer_backend/rnn/encoders.py:8 | change "from chainer import cuda" to "from chainer import mlu " |
| 91 | espnet/nets/chainer_backend/rnn/encoders.py:134 | change "ilens = cuda.to_cpu(ilens)" to "ilens = mlu.to_cpu(ilens) " |
| 92 | espnet/nets/chainer_backend/rnn/training.py:13 | change "from chainer import Variable, cuda, training" to "from chainer import Variable, mlu, training " |
| 93 | espnet/nets/chainer_backend/rnn/training.py:34 | change "with cuda.get_device_from_array(x) as dev:" to "with mlu.get_device_from_array(x) as dev: " |
| 94 | espnet/nets/chainer_backend/rnn/training.py:148 | change "from cupy.cuda import nccl" to "from cupy.mlu import cncl " |
| 95 | espnet/nets/chainer_backend/rnn/training.py:152 | change "self.nccl = nccl" to "self.cncl = cncl " |
| 96 | espnet/nets/chainer_backend/rnn/training.py:160 | change "with cuda.Device(self._devices[0]):" to "with mlu.Device(self._devices[0]): " |
| 97 | espnet/nets/chainer_backend/rnn/training.py:172 | change "null_stream = cuda.Stream.null" to "null_stream = mlu.Stream.null " |
| 98 | espnet/nets/chainer_backend/rnn/training.py:179 | change "self.nccl.NCCL_FLOAT," to "self.cncl.NCCL_FLOAT, " |
| 99 | espnet/nets/chainer_backend/rnn/training.py:180 | change "self.nccl.NCCL_SUM," to "self.cncl.NCCL_SUM, " |
| 100 | espnet/nets/chainer_backend/rnn/training.py:208 | change "gp.data.ptr, gp.size, self.nccl.NCCL_FLOAT, 0, null_stream.ptr" to "gp.data.ptr, gp.size, self.cncl.NCCL_FLOAT, 0, null_stream.ptr " |
| 101 | espnet/nets/chainer_backend/rnn/training.py:242 | change "xp = cuda.cupy if device != -1 else np" to "xp = mlu.cupy if device != -1 else np " |
| 102 | espnet/nets/chainer_backend/transformer/training.py:10 | change "from chainer import cuda" to "from chainer import mlu " |
| 103 | espnet/nets/chainer_backend/transformer/training.py:34 | change "with cuda.get_device_from_array(x) as dev:" to "with mlu.get_device_from_array(x) as dev: " |
| 104 | espnet/nets/chainer_backend/transformer/training.py:145 | change "from cupy.cuda import nccl" to "from cupy.mlu import cncl " |
| 105 | espnet/nets/chainer_backend/transformer/training.py:152 | change "self.nccl = nccl" to "self.cncl = cncl " |
| 106 | espnet/nets/chainer_backend/transformer/training.py:161 | change "with cuda.Device(self._devices[0]):" to "with mlu.Device(self._devices[0]): " |
| 107 | espnet/nets/chainer_backend/transformer/training.py:171 | change "null_stream = cuda.Stream.null" to "null_stream = mlu.Stream.null " |
| 108 | espnet/nets/chainer_backend/transformer/training.py:178 | change "self.nccl.NCCL_FLOAT," to "self.cncl.NCCL_FLOAT, " |
| 109 | espnet/nets/chainer_backend/transformer/training.py:179 | change "self.nccl.NCCL_SUM," to "self.cncl.NCCL_SUM, " |
| 110 | espnet/nets/chainer_backend/transformer/training.py:207 | change "gp.data.ptr, gp.size, self.nccl.NCCL_FLOAT, 0, null_stream.ptr" to "gp.data.ptr, gp.size, self.cncl.NCCL_FLOAT, 0, null_stream.ptr " |
| 111 | espnet/nets/pytorch_backend/ctc.py:5 | add "import torch_mlu" |
| 112 | espnet/nets/pytorch_backend/ctc.py:65 | change "with torch.backends.cudnn.flags(deterministic=True):" to "with torch.backends.mlufusion.flags(deterministic=True): " |
| 113 | espnet/nets/pytorch_backend/e2e_asr.py:15 | add "import torch_mlu" |
| 114 | espnet/nets/pytorch_backend/e2e_asr_maskctc.py:18 | add "import torch_mlu" |
| 115 | espnet/nets/pytorch_backend/e2e_asr_mix.py:19 | add "import torch_mlu" |
| 116 | espnet/nets/pytorch_backend/e2e_asr_mix_transformer.py:24 | add "import torch_mlu" |
| 117 | espnet/nets/pytorch_backend/e2e_asr_mulenc.py:15 | add "import torch_mlu" |
| 118 | espnet/nets/pytorch_backend/e2e_asr_transducer.py:11 | add "import torch_mlu" |
| 119 | espnet/nets/pytorch_backend/e2e_asr_transformer.py:11 | add "import torch_mlu" |
| 120 | espnet/nets/pytorch_backend/e2e_mt.py:14 | add "import torch_mlu" |
| 121 | espnet/nets/pytorch_backend/e2e_mt_transformer.py:11 | add "import torch_mlu" |
| 122 | espnet/nets/pytorch_backend/e2e_st.py:17 | add "import torch_mlu" |
| 123 | espnet/nets/pytorch_backend/e2e_st_transformer.py:11 | add "import torch_mlu" |
| 124 | espnet/nets/pytorch_backend/e2e_tts_fastspeech.py:8 | add "import torch_mlu" |
| 125 | espnet/nets/pytorch_backend/e2e_tts_tacotron2.py:9 | add "import torch_mlu" |
| 126 | espnet/nets/pytorch_backend/e2e_tts_transformer.py:8 | add "import torch_mlu" |
| 127 | espnet/nets/pytorch_backend/e2e_vc_tacotron2.py:10 | add "import torch_mlu" |
| 128 | espnet/nets/pytorch_backend/e2e_vc_transformer.py:8 | add "import torch_mlu" |
| 129 | espnet/nets/pytorch_backend/gtn_ctc.py:7 | add "import torch_mlu" |
| 130 | espnet/nets/pytorch_backend/gtn_ctc.py:85 | change "return torch.mean(loss.cuda() if log_probs.is_cuda else loss)" to "return torch.mean(loss.mlu() if log_probs.is_mlu else loss) " |
| 131 | espnet/nets/pytorch_backend/gtn_ctc.py:108 | change "if grad_output.is_cuda:" to "if grad_output.is_mlu: " |
| 132 | espnet/nets/pytorch_backend/gtn_ctc.py:109 | change "input_grad = input_grad.cuda()" to "input_grad = input_grad.mlu() " |
| 133 | espnet/nets/pytorch_backend/nets_utils.py:9 | add "import torch_mlu" |
| 134 | espnet/nets/pytorch_backend/wavenet.py:13 | add "import torch_mlu" |
| 135 | espnet/nets/pytorch_backend/conformer/contextual_block_encoder_layer.py:8 | add "import torch_mlu" |
| 136 | espnet/nets/pytorch_backend/conformer/encoder.py:9 | add "import torch_mlu" |
| 137 | espnet/nets/pytorch_backend/conformer/encoder_layer.py:10 | add "import torch_mlu" |
| 138 | espnet/nets/pytorch_backend/conformer/swish.py:10 | add "import torch_mlu" |
| 139 | espnet/nets/pytorch_backend/fastspeech/duration_calculator.py:9 | add "import torch_mlu" |
| 140 | espnet/nets/pytorch_backend/fastspeech/duration_predictor.py:9 | add "import torch_mlu" |
| 141 | espnet/nets/pytorch_backend/fastspeech/length_regulator.py:11 | add "import torch_mlu" |
| 142 | espnet/nets/pytorch_backend/frontends/beamformer.py:1 | add "import torch_mlu" |
| 143 | espnet/nets/pytorch_backend/frontends/dnn_beamformer.py:4 | add "import torch_mlu" |
| 144 | espnet/nets/pytorch_backend/frontends/dnn_wpe.py:3 | add "import torch_mlu" |
| 145 | espnet/nets/pytorch_backend/frontends/feature_transform.py:5 | add "import torch_mlu" |
| 146 | espnet/nets/pytorch_backend/frontends/frontend.py:4 | add "import torch_mlu" |
| 147 | espnet/nets/pytorch_backend/frontends/mask_estimator.py:4 | add "import torch_mlu" |
| 148 | espnet/nets/pytorch_backend/lm/default.py:6 | add "import torch_mlu" |
| 149 | espnet/nets/pytorch_backend/lm/seq_rnn.py:3 | add "import torch_mlu" |
| 150 | espnet/nets/pytorch_backend/lm/transformer.py:6 | add "import torch_mlu" |
| 151 | espnet/nets/pytorch_backend/rnn/attentions.py:6 | add "import torch_mlu" |
| 152 | espnet/nets/pytorch_backend/rnn/decoders.py:9 | add "import torch_mlu" |
| 153 | espnet/nets/pytorch_backend/rnn/decoders.py:755 | change "if att_weight > 0.0 and not lpz[0].is_cuda" to "if att_weight > 0.0 and not lpz[0].is_mlu " |
| 154 | espnet/nets/pytorch_backend/rnn/decoders.py:940 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 155 | espnet/nets/pytorch_backend/rnn/encoders.py:5 | add "import torch_mlu" |
| 156 | espnet/nets/pytorch_backend/streaming/segment.py:2 | add "import torch_mlu" |
| 157 | espnet/nets/pytorch_backend/streaming/window.py:1 | add "import torch_mlu" |
| 158 | espnet/nets/pytorch_backend/tacotron2/cbhg.py:9 | add "import torch_mlu" |
| 159 | espnet/nets/pytorch_backend/tacotron2/decoder.py:10 | add "import torch_mlu" |
| 160 | espnet/nets/pytorch_backend/tacotron2/encoder.py:10 | add "import torch_mlu" |
| 161 | espnet/nets/pytorch_backend/transducer/blocks.py:5 | add "import torch_mlu" |
| 162 | espnet/nets/pytorch_backend/transducer/conv1d_nets.py:5 | add "import torch_mlu" |
| 163 | espnet/nets/pytorch_backend/transducer/custom_decoder.py:5 | add "import torch_mlu" |
| 164 | espnet/nets/pytorch_backend/transducer/custom_encoder.py:5 | add "import torch_mlu" |
| 165 | espnet/nets/pytorch_backend/transducer/error_calculator.py:5 | add "import torch_mlu" |
| 166 | espnet/nets/pytorch_backend/transducer/initializer.py:6 | add "import torch_mlu" |
| 167 | espnet/nets/pytorch_backend/transducer/joint_network.py:3 | add "import torch_mlu" |
| 168 | espnet/nets/pytorch_backend/transducer/rnn_decoder.py:5 | add "import torch_mlu" |
| 169 | espnet/nets/pytorch_backend/transducer/rnn_decoder.py:65 | change "self.multi_gpus = torch.cuda.device_count() > 1" to "self.multi_gpus = torch.mlu.device_count() > 1 " |
| 170 | espnet/nets/pytorch_backend/transducer/rnn_encoder.py:15 | add "import torch_mlu" |
| 171 | espnet/nets/pytorch_backend/transducer/transducer_tasks.py:5 | add "import torch_mlu" |
| 172 | espnet/nets/pytorch_backend/transducer/transducer_tasks.py:198 | change "with torch.backends.cudnn.flags(deterministic=True):" to "with torch.backends.mlufusion.flags(deterministic=True): " |
| 173 | espnet/nets/pytorch_backend/transducer/transformer_decoder_layer.py:5 | add "import torch_mlu" |
| 174 | espnet/nets/pytorch_backend/transducer/utils.py:7 | add "import torch_mlu" |
| 175 | espnet/nets/pytorch_backend/transducer/vgg2l.py:5 | add "import torch_mlu" |
| 176 | espnet/nets/pytorch_backend/transformer/add_sos_eos.py:9 | add "import torch_mlu" |
| 177 | espnet/nets/pytorch_backend/transformer/attention.py:12 | add "import torch_mlu" |
| 178 | espnet/nets/pytorch_backend/transformer/contextual_block_encoder_layer.py:9 | add "import torch_mlu" |
| 179 | espnet/nets/pytorch_backend/transformer/decoder.py:12 | add "import torch_mlu" |
| 180 | espnet/nets/pytorch_backend/transformer/decoder_layer.py:9 | add "import torch_mlu" |
| 181 | espnet/nets/pytorch_backend/transformer/dynamic_conv.py:4 | add "import torch_mlu" |
| 182 | espnet/nets/pytorch_backend/transformer/dynamic_conv2d.py:4 | add "import torch_mlu" |
| 183 | espnet/nets/pytorch_backend/transformer/embedding.py:11 | add "import torch_mlu" |
| 184 | espnet/nets/pytorch_backend/transformer/encoder.py:8 | add "import torch_mlu" |
| 185 | espnet/nets/pytorch_backend/transformer/encoder_layer.py:9 | add "import torch_mlu" |
| 186 | espnet/nets/pytorch_backend/transformer/encoder_mix.py:9 | add "import torch_mlu" |
| 187 | espnet/nets/pytorch_backend/transformer/initializer.py:9 | add "import torch_mlu" |
| 188 | espnet/nets/pytorch_backend/transformer/label_smoothing_loss.py:9 | add "import torch_mlu" |
| 189 | espnet/nets/pytorch_backend/transformer/layer_norm.py:9 | add "import torch_mlu" |
| 190 | espnet/nets/pytorch_backend/transformer/lightconv.py:4 | add "import torch_mlu" |
| 191 | espnet/nets/pytorch_backend/transformer/lightconv2d.py:4 | add "import torch_mlu" |
| 192 | espnet/nets/pytorch_backend/transformer/mask.py:6 | add "import torch_mlu" |
| 193 | espnet/nets/pytorch_backend/transformer/mask.py:13 | change ":param str device: "cpu" or "cuda" or torch.Tensor.device" to ":param str device: "cpu" or "mlu" or torch.Tensor.device " |
| 194 | espnet/nets/pytorch_backend/transformer/multi_layer_conv.py:9 | add "import torch_mlu" |
| 195 | espnet/nets/pytorch_backend/transformer/optimizer.py:9 | add "import torch_mlu" |
| 196 | espnet/nets/pytorch_backend/transformer/positionwise_feed_forward.py:9 | add "import torch_mlu" |
| 197 | espnet/nets/pytorch_backend/transformer/repeat.py:9 | add "import torch_mlu" |
| 198 | espnet/nets/pytorch_backend/transformer/subsampling.py:9 | add "import torch_mlu" |
| 199 | espnet/nets/pytorch_backend/transformer/subsampling_without_posenc.py:8 | add "import torch_mlu" |
| 200 | espnet/nets/scorers/ctc.py:4 | add "import torch_mlu" |
| 201 | espnet/nets/scorers/length_bonus.py:4 | add "import torch_mlu" |
| 202 | espnet/nets/scorers/ngram.py:6 | add "import torch_mlu" |
| 203 | espnet/optimizer/pytorch.py:4 | add "import torch_mlu" |
| 204 | espnet/st/pytorch_backend/st.py:12 | add "import torch_mlu" |
| 205 | espnet/st/pytorch_backend/st.py:107 | change "# check cuda availability" to "# check mlu availability " |
| 206 | espnet/st/pytorch_backend/st.py:108 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |
| 207 | espnet/st/pytorch_backend/st.py:109 | change "logging.warning("cuda is not available")" to "logging.warning("mlu is not available") " |
| 208 | espnet/st/pytorch_backend/st.py:155 | change "device = torch.device("cuda" if args.ngpu > 0 else "cpu")" to "device = torch.device("mlu" if args.ngpu > 0 else "cpu") " |
| 209 | espnet/st/pytorch_backend/st.py:608 | change "model.cuda()" to "model.mlu() " |
| 210 | espnet/transform/spec_augment.py:41 | add "import torch_mlu" |
| 211 | espnet/tts/pytorch_backend/tts.py:19 | add "import torch_mlu" |
| 212 | espnet/tts/pytorch_backend/tts.py:258 | change "# check cuda availability" to "# check mlu availability " |
| 213 | espnet/tts/pytorch_backend/tts.py:259 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |
| 214 | espnet/tts/pytorch_backend/tts.py:260 | change "logging.warning("cuda is not available")" to "logging.warning("mlu is not available") " |
| 215 | espnet/tts/pytorch_backend/tts.py:318 | change "device = torch.device("cuda" if args.ngpu > 0 else "cpu")" to "device = torch.device("mlu" if args.ngpu > 0 else "cpu") " |
| 216 | espnet/tts/pytorch_backend/tts.py:585 | change "device = torch.device("cuda" if args.ngpu > 0 else "cpu")" to "device = torch.device("mlu" if args.ngpu > 0 else "cpu") " |
| 217 | espnet/utils/dataset.py:8 | add "import torch_mlu" |
| 218 | espnet/utils/deterministic_utils.py:5 | add "import torch_mlu" |
| 219 | espnet/utils/spec_augment.py:31 | add "import torch_mlu" |
| 220 | espnet/vc/pytorch_backend/vc.py:19 | add "import torch_mlu" |
| 221 | espnet/vc/pytorch_backend/vc.py:258 | change "# check cuda availability" to "# check mlu availability " |
| 222 | espnet/vc/pytorch_backend/vc.py:259 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |
| 223 | espnet/vc/pytorch_backend/vc.py:260 | change "logging.warning("cuda is not available")" to "logging.warning("mlu is not available") " |
| 224 | espnet/vc/pytorch_backend/vc.py:329 | change "device = torch.device("cuda" if args.ngpu > 0 else "cpu")" to "device = torch.device("mlu" if args.ngpu > 0 else "cpu") " |
| 225 | espnet/vc/pytorch_backend/vc.py:582 | change "device = torch.device("cuda" if args.ngpu > 0 else "cpu")" to "device = torch.device("mlu" if args.ngpu > 0 else "cpu") " |
| 226 | espnet2/asr/ctc.py:3 | add "import torch_mlu" |
| 227 | espnet2/asr/espnet_model.py:5 | add "import torch_mlu" |
| 228 | espnet2/asr/maskctc_model.py:7 | add "import torch_mlu" |
| 229 | espnet2/asr/decoder/abs_decoder.py:4 | add "import torch_mlu" |
| 230 | espnet2/asr/decoder/mlm_decoder.py:7 | add "import torch_mlu" |
| 231 | espnet2/asr/decoder/rnn_decoder.py:4 | add "import torch_mlu" |
| 232 | espnet2/asr/decoder/transformer_decoder.py:7 | add "import torch_mlu" |
| 233 | espnet2/asr/encoder/abs_encoder.py:4 | add "import torch_mlu" |
| 234 | espnet2/asr/encoder/conformer_encoder.py:9 | add "import torch_mlu" |
| 235 | espnet2/asr/encoder/contextual_block_conformer_encoder.py:11 | add "import torch_mlu" |
| 236 | espnet2/asr/encoder/contextual_block_transformer_encoder.py:8 | add "import torch_mlu" |
| 237 | espnet2/asr/encoder/hubert_encoder.py:18 | add "import torch_mlu" |
| 238 | espnet2/asr/encoder/hubert_encoder.py:365 | change "if self.use_amp and self.encoder.mask_emb.dtype != torch.cuda.HalfTensor:" to "if self.use_amp and self.encoder.mask_emb.dtype != torch.mlu.HalfTensor: " |
| 239 | espnet2/asr/encoder/longformer_encoder.py:8 | add "import torch_mlu" |
| 240 | espnet2/asr/encoder/rnn_encoder.py:4 | add "import torch_mlu" |
| 241 | espnet2/asr/encoder/transformer_encoder.py:8 | add "import torch_mlu" |
| 242 | espnet2/asr/encoder/vgg_rnn_encoder.py:4 | add "import torch_mlu" |
| 243 | espnet2/asr/encoder/wav2vec2_encoder.py:11 | add "import torch_mlu" |
| 244 | espnet2/asr/frontend/abs_frontend.py:4 | add "import torch_mlu" |
| 245 | espnet2/asr/frontend/default.py:6 | add "import torch_mlu" |
| 246 | espnet2/asr/frontend/fused.py:4 | add "import torch_mlu" |
| 247 | espnet2/asr/frontend/fused.py:87 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 248 | espnet2/asr/frontend/fused.py:88 | change "dev = "cuda"" to "dev = "mlu" " |
| 249 | espnet2/asr/frontend/s3prl.py:9 | add "import torch_mlu" |
| 250 | espnet2/asr/frontend/windowing.py:9 | add "import torch_mlu" |
| 251 | espnet2/asr/postencoder/abs_postencoder.py:4 | add "import torch_mlu" |
| 252 | espnet2/asr/postencoder/hugging_face_transformers_postencoder.py:11 | add "import torch_mlu" |
| 253 | espnet2/asr/preencoder/abs_preencoder.py:4 | add "import torch_mlu" |
| 254 | espnet2/asr/preencoder/linear.py:9 | add "import torch_mlu" |
| 255 | espnet2/asr/preencoder/sinc.py:11 | add "import torch_mlu" |
| 256 | espnet2/asr/specaug/abs_specaug.py:3 | add "import torch_mlu" |
| 257 | espnet2/asr/specaug/specaug.py:18 | change "When using cuda mode, time_warp doesn't have reproducibility" to "When using mlu mode, time_warp doesn't have reproducibility " |
| 258 | espnet2/asr/transducer/beam_search_transducer.py:8 | add "import torch_mlu" |
| 259 | espnet2/asr/transducer/error_calculator.py:5 | add "import torch_mlu" |
| 260 | espnet2/asr/transducer/joint_network.py:3 | add "import torch_mlu" |
| 261 | espnet2/asr/transducer/transducer_decoder.py:5 | add "import torch_mlu" |
| 262 | espnet2/asr/transducer/utils.py:3 | add "import torch_mlu" |
| 263 | espnet2/bin/asr_align.py:14 | add "import torch_mlu" |
| 264 | espnet2/bin/asr_align.py:227 | change "device = "cuda"" to "device = "mlu" " |
| 265 | espnet2/bin/asr_inference.py:10 | add "import torch_mlu" |
| 266 | espnet2/bin/asr_inference.py:422 | change "device = "cuda"" to "device = "mlu" " |
| 267 | espnet2/bin/asr_inference_k2.py:11 | add "import torch_mlu" |
| 268 | espnet2/bin/asr_inference_k2.py:501 | change "device = "cuda"" to "device = "mlu" " |
| 269 | espnet2/bin/asr_inference_maskctc.py:9 | add "import torch_mlu" |
| 270 | espnet2/bin/asr_inference_maskctc.py:209 | change "device = "cuda"" to "device = "mlu" " |
| 271 | espnet2/bin/asr_inference_streaming.py:10 | add "import torch_mlu" |
| 272 | espnet2/bin/asr_inference_streaming.py:406 | change "device = "cuda"" to "device = "mlu" " |
| 273 | espnet2/bin/diar_inference.py:11 | add "import torch_mlu" |
| 274 | espnet2/bin/diar_inference.py:501 | change "device = "cuda"" to "device = "mlu" " |
| 275 | espnet2/bin/enh_inference.py:11 | add "import torch_mlu" |
| 276 | espnet2/bin/enh_inference.py:77 | change "if device == "cuda":" to "if device == "mlu": " |
| 277 | espnet2/bin/enh_inference.py:78 | change "# NOTE(kamo): "cuda" for torch.load always indicates cuda:0" to "# NOTE(kamo): "mlu" for torch.load always indicates mlu:0 " |
| 278 | espnet2/bin/enh_inference.py:80 | change "device = f"cuda:{torch.cuda.current_device()}"" to "device = f"mlu:{torch.mlu.current_device()}" " |
| 279 | espnet2/bin/enh_inference.py:414 | change "device = "cuda"" to "device = "mlu" " |
| 280 | espnet2/bin/enh_scoring.py:8 | add "import torch_mlu" |
| 281 | espnet2/bin/lm_calc_perplexity.py:9 | add "import torch_mlu" |
| 282 | espnet2/bin/lm_calc_perplexity.py:45 | change "device = "cuda"" to "device = "mlu" " |
| 283 | espnet2/bin/mt_inference.py:9 | add "import torch_mlu" |
| 284 | espnet2/bin/mt_inference.py:301 | change "device = "cuda"" to "device = "mlu" " |
| 285 | espnet2/bin/st_inference.py:9 | add "import torch_mlu" |
| 286 | espnet2/bin/st_inference.py:316 | change "device = "cuda"" to "device = "mlu" " |
| 287 | espnet2/bin/st_inference_streaming.py:10 | add "import torch_mlu" |
| 288 | espnet2/bin/st_inference_streaming.py:385 | change "device = "cuda"" to "device = "mlu" " |
| 289 | espnet2/bin/tts_inference.py:15 | add "import torch_mlu" |
| 290 | espnet2/bin/tts_inference.py:350 | change "device = "cuda"" to "device = "mlu" " |
| 291 | espnet2/diar/abs_diar.py:5 | add "import torch_mlu" |
| 292 | espnet2/diar/espnet_model.py:9 | add "import torch_mlu" |
| 293 | espnet2/diar/label_processor.py:1 | add "import torch_mlu" |
| 294 | espnet2/diar/attractor/abs_attractor.py:4 | add "import torch_mlu" |
| 295 | espnet2/diar/attractor/rnn_attractor.py:1 | add "import torch_mlu" |
| 296 | espnet2/diar/decoder/abs_decoder.py:4 | add "import torch_mlu" |
| 297 | espnet2/diar/decoder/linear_decoder.py:1 | add "import torch_mlu" |
| 298 | espnet2/diar/layers/abs_mask.py:5 | add "import torch_mlu" |
| 299 | espnet2/diar/layers/multi_mask.py:7 | add "import torch_mlu" |
| 300 | espnet2/diar/layers/tcn_nomask.py:10 | add "import torch_mlu" |
| 301 | espnet2/diar/separator/tcn_separator_nomask.py:4 | add "import torch_mlu" |
| 302 | espnet2/enh/abs_enh.py:5 | add "import torch_mlu" |
| 303 | espnet2/enh/espnet_enh_s2t_model.py:6 | add "import torch_mlu" |
| 304 | espnet2/enh/espnet_model.py:4 | add "import torch_mlu" |
| 305 | espnet2/enh/decoder/abs_decoder.py:4 | add "import torch_mlu" |
| 306 | espnet2/enh/decoder/conv_decoder.py:1 | add "import torch_mlu" |
| 307 | espnet2/enh/decoder/null_decoder.py:1 | add "import torch_mlu" |
| 308 | espnet2/enh/decoder/stft_decoder.py:1 | add "import torch_mlu" |
| 309 | espnet2/enh/encoder/abs_encoder.py:4 | add "import torch_mlu" |
| 310 | espnet2/enh/encoder/conv_encoder.py:1 | add "import torch_mlu" |
| 311 | espnet2/enh/encoder/null_encoder.py:1 | add "import torch_mlu" |
| 312 | espnet2/enh/encoder/stft_encoder.py:1 | add "import torch_mlu" |
| 313 | espnet2/enh/layers/beamformer.py:4 | add "import torch_mlu" |
| 314 | espnet2/enh/layers/complex_utils.py:4 | add "import torch_mlu" |
| 315 | espnet2/enh/layers/complexnn.py:1 | add "import torch_mlu" |
| 316 | espnet2/enh/layers/dc_crn.py:9 | add "import torch_mlu" |
| 317 | espnet2/enh/layers/dnn_beamformer.py:5 | add "import torch_mlu" |
| 318 | espnet2/enh/layers/dnn_wpe.py:3 | add "import torch_mlu" |
| 319 | espnet2/enh/layers/dpmulcat.py:1 | add "import torch_mlu" |
| 320 | espnet2/enh/layers/dprnn.py:11 | add "import torch_mlu" |
| 321 | espnet2/enh/layers/fasnet.py:10 | add "import torch_mlu" |
| 322 | espnet2/enh/layers/ifasnet.py:10 | add "import torch_mlu" |
| 323 | espnet2/enh/layers/mask_estimator.py:4 | add "import torch_mlu" |
| 324 | espnet2/enh/layers/skim.py:6 | add "import torch_mlu" |
| 325 | espnet2/enh/layers/tcn.py:11 | add "import torch_mlu" |
| 326 | espnet2/enh/layers/tcndenseunet.py:1 | add "import torch_mlu" |
| 327 | espnet2/enh/layers/wpe.py:3 | add "import torch_mlu" |
| 328 | espnet2/enh/loss/criterions/abs_loss.py:3 | add "import torch_mlu" |
| 329 | espnet2/enh/loss/criterions/tf_domain.py:5 | add "import torch_mlu" |
| 330 | espnet2/enh/loss/criterions/time_domain.py:7 | add "import torch_mlu" |
| 331 | espnet2/enh/loss/wrappers/abs_wrapper.py:4 | add "import torch_mlu" |
| 332 | espnet2/enh/loss/wrappers/fixed_order.py:3 | add "import torch_mlu" |
| 333 | espnet2/enh/loss/wrappers/multilayer_pit_solver.py:1 | add "import torch_mlu" |
| 334 | espnet2/enh/loss/wrappers/pit_solver.py:4 | add "import torch_mlu" |
| 335 | espnet2/enh/separator/abs_separator.py:5 | add "import torch_mlu" |
| 336 | espnet2/enh/separator/asteroid_models.py:5 | add "import torch_mlu" |
| 337 | espnet2/enh/separator/conformer_separator.py:4 | add "import torch_mlu" |
| 338 | espnet2/enh/separator/dan_separator.py:5 | add "import torch_mlu" |
| 339 | espnet2/enh/separator/dc_crn_separator.py:4 | add "import torch_mlu" |
| 340 | espnet2/enh/separator/dccrn_separator.py:4 | add "import torch_mlu" |
| 341 | espnet2/enh/separator/dpcl_e2e_separator.py:4 | add "import torch_mlu" |
| 342 | espnet2/enh/separator/dpcl_separator.py:4 | add "import torch_mlu" |
| 343 | espnet2/enh/separator/dprnn_separator.py:4 | add "import torch_mlu" |
| 344 | espnet2/enh/separator/dptnet_separator.py:5 | add "import torch_mlu" |
| 345 | espnet2/enh/separator/fasnet_separator.py:4 | add "import torch_mlu" |
| 346 | espnet2/enh/separator/ineube_separator.py:4 | add "import torch_mlu" |
| 347 | espnet2/enh/separator/neural_beamformer.py:4 | add "import torch_mlu" |
| 348 | espnet2/enh/separator/rnn_separator.py:4 | add "import torch_mlu" |
| 349 | espnet2/enh/separator/skim_separator.py:4 | add "import torch_mlu" |
| 350 | espnet2/enh/separator/svoice_separator.py:5 | add "import torch_mlu" |
| 351 | espnet2/enh/separator/tcn_separator.py:4 | add "import torch_mlu" |
| 352 | espnet2/enh/separator/transformer_separator.py:4 | add "import torch_mlu" |
| 353 | espnet2/fst/lm_rescore.py:5 | add "import torch_mlu" |
| 354 | espnet2/fst/lm_rescore.py:79 | change "device: str = "cuda"," to "device: str = "mlu", " |
| 355 | espnet2/fst/lm_rescore.py:162 | change "device: str = "cuda"," to "device: str = "mlu", " |
| 356 | espnet2/gan_tts/abs_gan_tts.py:9 | add "import torch_mlu" |
| 357 | espnet2/gan_tts/espnet_model.py:9 | add "import torch_mlu" |
| 358 | espnet2/gan_tts/hifigan/hifigan.py:15 | add "import torch_mlu" |
| 359 | espnet2/gan_tts/hifigan/loss.py:12 | add "import torch_mlu" |
| 360 | espnet2/gan_tts/hifigan/residual_block.py:12 | add "import torch_mlu" |
| 361 | espnet2/gan_tts/jets/alignments.py:5 | add "import torch_mlu" |
| 362 | espnet2/gan_tts/jets/generator.py:10 | add "import torch_mlu" |
| 363 | espnet2/gan_tts/jets/jets.py:8 | add "import torch_mlu" |
| 364 | espnet2/gan_tts/jets/length_regulator.py:6 | add "import torch_mlu" |
| 365 | espnet2/gan_tts/jets/loss.py:9 | add "import torch_mlu" |
| 366 | espnet2/gan_tts/joint/joint_text2wav.py:8 | add "import torch_mlu" |
| 367 | espnet2/gan_tts/melgan/melgan.py:14 | add "import torch_mlu" |
| 368 | espnet2/gan_tts/melgan/pqmf.py:11 | add "import torch_mlu" |
| 369 | espnet2/gan_tts/melgan/residual_stack.py:12 | add "import torch_mlu" |
| 370 | espnet2/gan_tts/parallel_wavegan/parallel_wavegan.py:15 | add "import torch_mlu" |
| 371 | espnet2/gan_tts/parallel_wavegan/upsample.py:13 | add "import torch_mlu" |
| 372 | espnet2/gan_tts/style_melgan/style_melgan.py:16 | add "import torch_mlu" |
| 373 | espnet2/gan_tts/style_melgan/tade_res_block.py:12 | add "import torch_mlu" |
| 374 | espnet2/gan_tts/utils/get_random_segments.py:8 | add "import torch_mlu" |
| 375 | espnet2/gan_tts/vits/duration_predictor.py:13 | add "import torch_mlu" |
| 376 | espnet2/gan_tts/vits/flow.py:13 | add "import torch_mlu" |
| 377 | espnet2/gan_tts/vits/generator.py:14 | add "import torch_mlu" |
| 378 | espnet2/gan_tts/vits/loss.py:10 | add "import torch_mlu" |
| 379 | espnet2/gan_tts/vits/posterior_encoder.py:12 | add "import torch_mlu" |
| 380 | espnet2/gan_tts/vits/residual_coupling.py:12 | add "import torch_mlu" |
| 381 | espnet2/gan_tts/vits/text_encoder.py:13 | add "import torch_mlu" |
| 382 | espnet2/gan_tts/vits/transform.py:8 | add "import torch_mlu" |
| 383 | espnet2/gan_tts/vits/vits.py:10 | add "import torch_mlu" |
| 384 | espnet2/gan_tts/vits/monotonic_align/__init__.py:10 | add "import torch_mlu" |
| 385 | espnet2/gan_tts/wavenet/residual_block.py:13 | add "import torch_mlu" |
| 386 | espnet2/gan_tts/wavenet/wavenet.py:14 | add "import torch_mlu" |
| 387 | espnet2/hubert/espnet_model.py:12 | add "import torch_mlu" |
| 388 | espnet2/iterators/chunk_iter_factory.py:5 | add "import torch_mlu" |
| 389 | espnet2/layers/abs_normalize.py:4 | add "import torch_mlu" |
| 390 | espnet2/layers/global_mvn.py:5 | add "import torch_mlu" |
| 391 | espnet2/layers/inversible_interface.py:4 | add "import torch_mlu" |
| 392 | espnet2/layers/label_aggregation.py:3 | add "import torch_mlu" |
| 393 | espnet2/layers/log_mel.py:4 | add "import torch_mlu" |
| 394 | espnet2/layers/mask_along_axis.py:4 | add "import torch_mlu" |
| 395 | espnet2/layers/sinc_conv.py:9 | add "import torch_mlu" |
| 396 | espnet2/layers/stft.py:5 | add "import torch_mlu" |
| 397 | espnet2/layers/stft.py:93 | change "if input.is_cuda or torch.backends.mkl.is_available():" to "if input.is_mlu or torch.backends.mkl.is_available(): " |
| 398 | espnet2/layers/time_warp.py:2 | add "import torch_mlu" |
| 399 | espnet2/layers/utterance_mvn.py:3 | add "import torch_mlu" |
| 400 | espnet2/lm/abs_model.py:4 | add "import torch_mlu" |
| 401 | espnet2/lm/espnet_model.py:3 | add "import torch_mlu" |
| 402 | espnet2/lm/seq_rnn_lm.py:4 | add "import torch_mlu" |
| 403 | espnet2/lm/transformer_lm.py:3 | add "import torch_mlu" |
| 404 | espnet2/main_funcs/average_nbest_models.py:6 | add "import torch_mlu" |
| 405 | espnet2/main_funcs/calculate_all_attentions.py:4 | add "import torch_mlu" |
| 406 | espnet2/main_funcs/collect_stats.py:7 | add "import torch_mlu" |
| 407 | espnet2/main_funcs/collect_stats.py:52 | change "batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")" to "batch = to_device(batch, "mlu" if ngpu > 0 else "cpu") " |
| 408 | espnet2/main_funcs/pack_funcs.py:276 | add "import torch_mlu" |
| 409 | espnet2/mt/espnet_model.py:5 | add "import torch_mlu" |
| 410 | espnet2/mt/frontend/embedding.py:9 | add "import torch_mlu" |
| 411 | espnet2/optimizers/sgd.py:1 | add "import torch_mlu" |
| 412 | espnet2/schedulers/noam_lr.py:5 | add "import torch_mlu" |
| 413 | espnet2/schedulers/warmup_lr.py:4 | add "import torch_mlu" |
| 414 | espnet2/schedulers/warmup_step_lr.py:4 | add "import torch_mlu" |
| 415 | espnet2/st/espnet_model.py:5 | add "import torch_mlu" |
| 416 | espnet2/tasks/abs_task.py:14 | add "import torch_mlu" |
| 417 | espnet2/tasks/abs_task.py:335 | change "default="nccl"," to "default="cncl", " |
| 418 | espnet2/tasks/abs_task.py:1122 | change "device="cuda" if args.ngpu > 0 else "cpu"," to "device="mlu" if args.ngpu > 0 else "cpu", " |
| 419 | espnet2/tasks/abs_task.py:1227 | change "# NOTE(kamo): "cuda" for torch.load always indicates cuda:0" to "# NOTE(kamo): "mlu" for torch.load always indicates mlu:0 " |
| 420 | espnet2/tasks/abs_task.py:1229 | change "map_location=f"cuda:{torch.cuda.current_device()}"" to "map_location=f"mlu:{torch.mlu.current_device()}" " |
| 421 | espnet2/tasks/abs_task.py:1799 | change "device: Device type, "cpu", "cuda", or "cuda:N"." to "device: Device type, "cpu", "mlu", or "mlu:N". " |
| 422 | espnet2/tasks/abs_task.py:1822 | change "if device == "cuda":" to "if device == "mlu": " |
| 423 | espnet2/tasks/abs_task.py:1823 | change "# NOTE(kamo): "cuda" for torch.load always indicates cuda:0" to "# NOTE(kamo): "mlu" for torch.load always indicates mlu:0 " |
| 424 | espnet2/tasks/abs_task.py:1825 | change "device = f"cuda:{torch.cuda.current_device()}"" to "device = f"mlu:{torch.mlu.current_device()}" " |
| 425 | espnet2/tasks/asr.py:6 | add "import torch_mlu" |
| 426 | espnet2/tasks/diar.py:5 | add "import torch_mlu" |
| 427 | espnet2/tasks/enh.py:5 | add "import torch_mlu" |
| 428 | espnet2/tasks/enh_s2t.py:7 | add "import torch_mlu" |
| 429 | espnet2/tasks/gan_tts.py:11 | add "import torch_mlu" |
| 430 | espnet2/tasks/hubert.py:13 | add "import torch_mlu" |
| 431 | espnet2/tasks/lm.py:6 | add "import torch_mlu" |
| 432 | espnet2/tasks/mt.py:6 | add "import torch_mlu" |
| 433 | espnet2/tasks/st.py:6 | add "import torch_mlu" |
| 434 | espnet2/tasks/tts.py:9 | add "import torch_mlu" |
| 435 | espnet2/torch_utils/add_gradient_noise.py:1 | add "import torch_mlu" |
| 436 | espnet2/torch_utils/device_funcs.py:5 | add "import torch_mlu" |
| 437 | espnet2/torch_utils/device_funcs.py:44 | change "- torch.cuda.Tensor" to "- torch.mlu.Tensor " |
| 438 | espnet2/torch_utils/forward_adaptor.py:1 | add "import torch_mlu" |
| 439 | espnet2/torch_utils/get_layer_from_string.py:3 | add "import torch_mlu" |
| 440 | espnet2/torch_utils/initialize.py:7 | add "import torch_mlu" |
| 441 | espnet2/torch_utils/load_pretrained_model.py:4 | add "import torch_mlu" |
| 442 | espnet2/torch_utils/model_summary.py:3 | add "import torch_mlu" |
| 443 | espnet2/torch_utils/pytorch_version.py:1 | add "import torch_mlu" |
| 444 | espnet2/torch_utils/pytorch_version.py:7 | change "f"cuda.available={torch.cuda.is_available()}, "" to "f"mlu.available={torch.mlu.is_available()}, " " |
| 445 | espnet2/torch_utils/recursive_op.py:2 | add "import torch_mlu" |
| 446 | espnet2/torch_utils/set_all_random_seed.py:4 | add "import torch_mlu" |
| 447 | espnet2/train/abs_espnet_model.py:4 | add "import torch_mlu" |
| 448 | espnet2/train/abs_gan_espnet_model.py:9 | add "import torch_mlu" |
| 449 | espnet2/train/collate_fn.py:4 | add "import torch_mlu" |
| 450 | espnet2/train/dataset.py:14 | add "import torch_mlu" |
| 451 | espnet2/train/distributed_utils.py:6 | add "import torch_mlu" |
| 452 | espnet2/train/distributed_utils.py:14 | change "# torch.distributed.Backend: "nccl", "mpi", "gloo", or "tcp"" to "# torch.distributed.Backend: "cncl", "mpi", "gloo", or "tcp" " |
| 453 | espnet2/train/distributed_utils.py:15 | change "dist_backend: str = "nccl"" to "dist_backend: str = "cncl" " |
| 454 | espnet2/train/distributed_utils.py:89 | change "# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html" to "# https://docs.nvidia.com/deeplearning/sdk/cncl-developer-guide/docs/env.html " |
| 455 | espnet2/train/distributed_utils.py:109 | change "torch.cuda.set_device(self.local_rank)" to "torch.mlu.set_device(self.local_rank) " |
| 456 | espnet2/train/gan_trainer.py:13 | add "import torch_mlu" |
| 457 | espnet2/train/gan_trainer.py:124 | change "iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")" to "iterator_stop = torch.tensor(0).to("mlu" if ngpu > 0 else "cpu") " |
| 458 | espnet2/train/gan_trainer.py:137 | change "batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")" to "batch = to_device(batch, "mlu" if ngpu > 0 else "cpu") " |
| 459 | espnet2/train/gan_trainer.py:329 | change "iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")" to "iterator_stop = torch.tensor(0).to("mlu" if ngpu > 0 else "cpu") " |
| 460 | espnet2/train/gan_trainer.py:337 | change "batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")" to "batch = to_device(batch, "mlu" if ngpu > 0 else "cpu") " |
| 461 | espnet2/train/iterable_dataset.py:10 | add "import torch_mlu" |
| 462 | espnet2/train/reporter.py:14 | add "import torch_mlu" |
| 463 | espnet2/train/reporter.py:353 | change "if torch.cuda.is_initialized():" to "if torch.mlu.is_initialized(): " |
| 464 | espnet2/train/reporter.py:355 | change "torch.cuda.max_memory_reserved() / 2**30" to "torch.mlu.max_memory_reserved() / 2**30 " |
| 465 | espnet2/train/reporter.py:358 | change "if torch.cuda.is_available() and torch.cuda.max_memory_cached() > 0:" to "if torch.mlu.is_available() and torch.mlu.max_memory_cached() > 0: " |
| 466 | espnet2/train/reporter.py:359 | change "stats["gpu_cached_mem_GB"] = torch.cuda.max_memory_cached() / 2**30" to "stats["gpu_cached_mem_GB"] = torch.mlu.max_memory_cached() / 2**30 " |
| 467 | espnet2/train/trainer.py:13 | add "import torch_mlu" |
| 468 | espnet2/train/trainer.py:135 | change "map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu"," to "map_location=f"mlu:{torch.mlu.current_device()}" if ngpu > 0 else "cpu", " |
| 469 | espnet2/train/trainer.py:224 | change "[torch.cuda.current_device()]" to "[torch.mlu.current_device()] " |
| 470 | espnet2/train/trainer.py:230 | change "torch.cuda.current_device()" to "torch.mlu.current_device() " |
| 471 | espnet2/train/trainer.py:497 | change "iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")" to "iterator_stop = torch.tensor(0).to("mlu" if ngpu > 0 else "cpu") " |
| 472 | espnet2/train/trainer.py:512 | change "batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")" to "batch = to_device(batch, "mlu" if ngpu > 0 else "cpu") " |
| 473 | espnet2/train/trainer.py:735 | change "iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")" to "iterator_stop = torch.tensor(0).to("mlu" if ngpu > 0 else "cpu") " |
| 474 | espnet2/train/trainer.py:745 | change "batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")" to "batch = to_device(batch, "mlu" if ngpu > 0 else "cpu") " |
| 475 | espnet2/train/trainer.py:799 | change "batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")" to "batch = to_device(batch, "mlu" if ngpu > 0 else "cpu") " |
| 476 | espnet2/tts/abs_tts.py:9 | add "import torch_mlu" |
| 477 | espnet2/tts/espnet_model.py:9 | add "import torch_mlu" |
| 478 | espnet2/tts/fastspeech/fastspeech.py:9 | add "import torch_mlu" |
| 479 | espnet2/tts/fastspeech2/fastspeech2.py:9 | add "import torch_mlu" |
| 480 | espnet2/tts/fastspeech2/loss.py:8 | add "import torch_mlu" |
| 481 | espnet2/tts/fastspeech2/variance_predictor.py:8 | add "import torch_mlu" |
| 482 | espnet2/tts/feats_extract/abs_feats_extract.py:4 | add "import torch_mlu" |
| 483 | espnet2/tts/feats_extract/dio.py:12 | add "import torch_mlu" |
| 484 | espnet2/tts/feats_extract/energy.py:9 | add "import torch_mlu" |
| 485 | espnet2/tts/feats_extract/linear_spectrogram.py:3 | add "import torch_mlu" |
| 486 | espnet2/tts/feats_extract/log_mel_fbank.py:4 | add "import torch_mlu" |
| 487 | espnet2/tts/feats_extract/log_spectrogram.py:3 | add "import torch_mlu" |
| 488 | espnet2/tts/gst/style_encoder.py:8 | add "import torch_mlu" |
| 489 | espnet2/tts/tacotron2/tacotron2.py:9 | add "import torch_mlu" |
| 490 | espnet2/tts/transformer/transformer.py:8 | add "import torch_mlu" |
| 491 | espnet2/tts/utils/duration_calculator.py:10 | add "import torch_mlu" |
| 492 | espnet2/tts/utils/parallel_wavegan_pretrained_vocoder.py:11 | add "import torch_mlu" |
| 493 | espnet2/utils/griffin_lim.py:14 | add "import torch_mlu" |
| 494 | test/test_asr_init.py:10 | add "import torch_mlu" |
| 495 | test/test_asr_init.py:131 | change "def pytorch_prepare_inputs(idim, odim, ilens, olens, is_cuda=False):" to "def pytorch_prepare_inputs(idim, odim, ilens, olens, is_mlu=False): " |
| 496 | test/test_asr_init.py:142 | change "if is_cuda:" to "if is_mlu: " |
| 497 | test/test_asr_init.py:143 | change "xs_pad = xs_pad.cuda()" to "xs_pad = xs_pad.mlu() " |
| 498 | test/test_asr_init.py:144 | change "ys_pad = ys_pad.cuda()" to "ys_pad = ys_pad.mlu() " |
| 499 | test/test_asr_init.py:145 | change "ilens = ilens.cuda()" to "ilens = ilens.mlu() " |
| 500 | test/test_asr_quantize.py:5 | add "import torch_mlu" |
| 501 | test/test_batch_beam_search.py:7 | add "import torch_mlu" |
| 502 | test/test_batch_beam_search.py:71 | change "for device in ("cpu", "cuda")" to "for device in ("cpu", "mlu") " |
| 503 | test/test_batch_beam_search.py:98 | change "if device == "cuda" and not torch.cuda.is_available():" to "if device == "mlu" and not torch.mlu.is_available(): " |
| 504 | test/test_batch_beam_search.py:99 | change "pytest.skip("no cuda device is available")" to "pytest.skip("no mlu device is available") " |
| 505 | test/test_beam_search.py:5 | add "import torch_mlu" |
| 506 | test/test_beam_search.py:126 | change "for device in ("cpu", "cuda")" to "for device in ("cpu", "mlu") " |
| 507 | test/test_beam_search.py:142 | change "if device == "cuda" and not torch.cuda.is_available():" to "if device == "mlu" and not torch.mlu.is_available(): " |
| 508 | test/test_beam_search.py:143 | change "pytest.skip("no cuda device is available")" to "pytest.skip("no mlu device is available") " |
| 509 | test/test_custom_transducer.py:8 | add "import torch_mlu" |
| 510 | test/test_e2e_asr.py:17 | add "import torch_mlu" |
| 511 | test/test_e2e_asr.py:79 | change "def prepare_inputs(mode, ilens=[14, 13], olens=[4, 3], is_cuda=False):" to "def prepare_inputs(mode, ilens=[14, 13], olens=[4, 3], is_mlu=False): " |
| 512 | test/test_e2e_asr.py:87 | change "if is_cuda:" to "if is_mlu: " |
| 513 | test/test_e2e_asr.py:101 | change "if is_cuda:" to "if is_mlu: " |
| 514 | test/test_e2e_asr.py:102 | change "xs_pad = xs_pad.cuda()" to "xs_pad = xs_pad.mlu() " |
| 515 | test/test_e2e_asr.py:103 | change "ilens = ilens.cuda()" to "ilens = ilens.mlu() " |
| 516 | test/test_e2e_asr.py:104 | change "ys_pad = ys_pad.cuda()" to "ys_pad = ys_pad.mlu() " |
| 517 | test/test_e2e_asr.py:111 | change "def convert_batch(batch, backend="pytorch", is_cuda=False, idim=10, odim=5):" to "def convert_batch(batch, backend="pytorch", is_mlu=False, idim=10, odim=5): " |
| 518 | test/test_e2e_asr.py:122 | change "if is_cuda:" to "if is_mlu: " |
| 519 | test/test_e2e_asr.py:123 | change "xs = xs.cuda()" to "xs = xs.mlu() " |
| 520 | test/test_e2e_asr.py:124 | change "ilens = ilens.cuda()" to "ilens = ilens.mlu() " |
| 521 | test/test_e2e_asr.py:125 | change "ys = ys.cuda()" to "ys = ys.mlu() " |
| 522 | test/test_e2e_asr.py:127 | change "if is_cuda:" to "if is_mlu: " |
| 523 | test/test_e2e_asr.py:704 | change "not torch.cuda.is_available() and not chainer.cuda.available, reason="gpu required"" to "not torch.mlu.is_available() and not chainer.mlu.available, reason="gpu required" " |
| 524 | test/test_e2e_asr.py:715 | change "batch = prepare_inputs("pytorch", is_cuda=True)" to "batch = prepare_inputs("pytorch", is_mlu=True) " |
| 525 | test/test_e2e_asr.py:716 | change "model.cuda()" to "model.mlu() " |
| 526 | test/test_e2e_asr.py:718 | change "batch = prepare_inputs("chainer", is_cuda=True)" to "batch = prepare_inputs("chainer", is_mlu=True) " |
| 527 | test/test_e2e_asr.py:728 | change "@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")" to "@pytest.mark.skipif(torch.mlu.device_count() < 2, reason="multi gpu required") " |
| 528 | test/test_e2e_asr.py:741 | change "batch = prepare_inputs("pytorch", is_cuda=True)" to "batch = prepare_inputs("pytorch", is_mlu=True) " |
| 529 | test/test_e2e_asr.py:742 | change "model.cuda()" to "model.mlu() " |
| 530 | test/test_e2e_asr.py:752 | change "with cupy.cuda.Device(device):" to "with cupy.mlu.Device(device): " |
| 531 | test/test_e2e_asr.py:753 | change "batch = prepare_inputs("chainer", is_cuda=True)" to "batch = prepare_inputs("chainer", is_mlu=True) " |
| 532 | test/test_e2e_asr_conformer.py:4 | add "import torch_mlu" |
| 533 | test/test_e2e_asr_maskctc.py:4 | add "import torch_mlu" |
| 534 | test/test_e2e_asr_mulenc.py:16 | add "import torch_mlu" |
| 535 | test/test_e2e_asr_mulenc.py:83 | change "def prepare_inputs(mode, num_encs=2, is_cuda=False):" to "def prepare_inputs(mode, num_encs=2, is_mlu=False): " |
| 536 | test/test_e2e_asr_mulenc.py:101 | change "if is_cuda:" to "if is_mlu: " |
| 537 | test/test_e2e_asr_mulenc.py:102 | change "xs_pad_list = [xs_pad.cuda() for xs_pad in xs_pad_list]" to "xs_pad_list = [xs_pad.mlu() for xs_pad in xs_pad_list] " |
| 538 | test/test_e2e_asr_mulenc.py:103 | change "ilens_list = [ilens.cuda() for ilens in ilens_list]" to "ilens_list = [ilens.mlu() for ilens in ilens_list] " |
| 539 | test/test_e2e_asr_mulenc.py:104 | change "ys_pad = ys_pad.cuda()" to "ys_pad = ys_pad.mlu() " |
| 540 | test/test_e2e_asr_mulenc.py:112 | change "batch, backend="pytorch", is_cuda=False, idim=2, odim=2, num_inputs=2" to "batch, backend="pytorch", is_mlu=False, idim=2, odim=2, num_inputs=2 " |
| 541 | test/test_e2e_asr_mulenc.py:135 | change "if is_cuda:" to "if is_mlu: " |
| 542 | test/test_e2e_asr_mulenc.py:136 | change "xs_list = [xs_list[idx].cuda() for idx in range(num_inputs)]" to "xs_list = [xs_list[idx].mlu() for idx in range(num_inputs)] " |
| 543 | test/test_e2e_asr_mulenc.py:137 | change "ilens_list = [ilens_list[idx].cuda() for idx in range(num_inputs)]" to "ilens_list = [ilens_list[idx].mlu() for idx in range(num_inputs)] " |
| 544 | test/test_e2e_asr_mulenc.py:138 | change "ys = ys.cuda()" to "ys = ys.mlu() " |
| 545 | test/test_e2e_asr_mulenc.py:511 | change "@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")" to "@pytest.mark.skipif(not torch.mlu.is_available(), reason="gpu required") " |
| 546 | test/test_e2e_asr_mulenc.py:524 | change "batch = prepare_inputs("pytorch", num_encs, is_cuda=True)" to "batch = prepare_inputs("pytorch", num_encs, is_mlu=True) " |
| 547 | test/test_e2e_asr_mulenc.py:525 | change "model.cuda()" to "model.mlu() " |
| 548 | test/test_e2e_asr_mulenc.py:530 | change "@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")" to "@pytest.mark.skipif(torch.mlu.device_count() < 2, reason="multi gpu required") " |
| 549 | test/test_e2e_asr_mulenc.py:546 | change "batch = prepare_inputs("pytorch", num_encs, is_cuda=True)" to "batch = prepare_inputs("pytorch", num_encs, is_mlu=True) " |
| 550 | test/test_e2e_asr_mulenc.py:547 | change "model.cuda()" to "model.mlu() " |
| 551 | test/test_e2e_asr_transducer.py:9 | add "import torch_mlu" |
| 552 | test/test_e2e_asr_transducer.py:130 | change "def prepare_inputs(idim, odim, ilens, olens, is_cuda=False):" to "def prepare_inputs(idim, odim, ilens, olens, is_mlu=False): " |
| 553 | test/test_e2e_asr_transducer.py:142 | change "if is_cuda:" to "if is_mlu: " |
| 554 | test/test_e2e_asr_transducer.py:143 | change "feats = feats.cuda()" to "feats = feats.mlu() " |
| 555 | test/test_e2e_asr_transducer.py:144 | change "labels = labels.cuda()" to "labels = labels.mlu() " |
| 556 | test/test_e2e_asr_transducer.py:145 | change "feats_len = feats_len.cuda()" to "feats_len = feats_len.mlu() " |
| 557 | test/test_e2e_asr_transducer.py:256 | change "@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")" to "@pytest.mark.skipif(torch.mlu.device_count() < 2, reason="multi gpu required") " |
| 558 | test/test_e2e_asr_transducer.py:273 | change "model.cuda()" to "model.mlu() " |
| 559 | test/test_e2e_asr_transducer.py:275 | change "batch = prepare_inputs(idim, odim, ilens, olens, is_cuda=True)" to "batch = prepare_inputs(idim, odim, ilens, olens, is_mlu=True) " |
| 560 | test/test_e2e_asr_transducer.py:287 | change "batch = prepare_inputs(idim, odim, ilens, olens, is_cuda=False)" to "batch = prepare_inputs(idim, odim, ilens, olens, is_mlu=False) " |
| 561 | test/test_e2e_asr_transformer.py:6 | add "import torch_mlu" |
| 562 | test/test_e2e_asr_transformer.py:27 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 563 | test/test_e2e_asr_transformer.py:29 | change "f.cuda()" to "f.mlu() " |
| 564 | test/test_e2e_asr_transformer.py:30 | change "assert len(f(x.cuda(), m.cuda())) == 2" to "assert len(f(x.mlu(), m.mlu())) == 2 " |
| 565 | test/test_e2e_compatibility.py:20 | add "import torch_mlu" |
| 566 | test/test_e2e_mt.py:17 | add "import torch_mlu" |
| 567 | test/test_e2e_mt.py:67 | change "def prepare_inputs(mode, ilens=[20, 10], olens=[4, 3], is_cuda=False):" to "def prepare_inputs(mode, ilens=[20, 10], olens=[4, 3], is_mlu=False): " |
| 568 | test/test_e2e_mt.py:81 | change "if is_cuda:" to "if is_mlu: " |
| 569 | test/test_e2e_mt.py:82 | change "xs_pad = xs_pad.cuda()" to "xs_pad = xs_pad.mlu() " |
| 570 | test/test_e2e_mt.py:83 | change "ilens = ilens.cuda()" to "ilens = ilens.mlu() " |
| 571 | test/test_e2e_mt.py:84 | change "ys_pad = ys_pad.cuda()" to "ys_pad = ys_pad.mlu() " |
| 572 | test/test_e2e_mt.py:91 | change "def convert_batch(batch, backend="pytorch", is_cuda=False, idim=5, odim=5):" to "def convert_batch(batch, backend="pytorch", is_mlu=False, idim=5, odim=5): " |
| 573 | test/test_e2e_mt.py:102 | change "if is_cuda:" to "if is_mlu: " |
| 574 | test/test_e2e_mt.py:103 | change "xs = xs.cuda()" to "xs = xs.mlu() " |
| 575 | test/test_e2e_mt.py:104 | change "ilens = ilens.cuda()" to "ilens = ilens.mlu() " |
| 576 | test/test_e2e_mt.py:105 | change "ys = ys.cuda()" to "ys = ys.mlu() " |
| 577 | test/test_e2e_mt.py:362 | change "not torch.cuda.is_available() and not chainer.cuda.available, reason="gpu required"" to "not torch.mlu.is_available() and not chainer.mlu.available, reason="gpu required" " |
| 578 | test/test_e2e_mt.py:370 | change "batch = prepare_inputs("pytorch", is_cuda=True)" to "batch = prepare_inputs("pytorch", is_mlu=True) " |
| 579 | test/test_e2e_mt.py:371 | change "model.cuda()" to "model.mlu() " |
| 580 | test/test_e2e_mt.py:382 | change "@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")" to "@pytest.mark.skipif(torch.mlu.device_count() < 2, reason="multi gpu required") " |
| 581 | test/test_e2e_mt.py:392 | change "batch = prepare_inputs("pytorch", is_cuda=True)" to "batch = prepare_inputs("pytorch", is_mlu=True) " |
| 582 | test/test_e2e_mt.py:393 | change "model.cuda()" to "model.mlu() " |
| 583 | test/test_e2e_mt_transformer.py:9 | add "import torch_mlu" |
| 584 | test/test_e2e_st.py:17 | add "import torch_mlu" |
| 585 | test/test_e2e_st.py:80 | change "mode, ilens=[20, 15], olens_tgt=[4, 3], olens_src=[3, 2], is_cuda=False" to "mode, ilens=[20, 15], olens_tgt=[4, 3], olens_src=[3, 2], is_mlu=False " |
| 586 | test/test_e2e_st.py:97 | change "if is_cuda:" to "if is_mlu: " |
| 587 | test/test_e2e_st.py:98 | change "xs_pad = xs_pad.cuda()" to "xs_pad = xs_pad.mlu() " |
| 588 | test/test_e2e_st.py:99 | change "ilens = ilens.cuda()" to "ilens = ilens.mlu() " |
| 589 | test/test_e2e_st.py:100 | change "ys_pad_tgt = ys_pad_tgt.cuda()" to "ys_pad_tgt = ys_pad_tgt.mlu() " |
| 590 | test/test_e2e_st.py:101 | change "ys_pad_src = ys_pad_src.cuda()" to "ys_pad_src = ys_pad_src.mlu() " |
| 591 | test/test_e2e_st.py:108 | change "def convert_batch(batch, backend="pytorch", is_cuda=False, idim=40, odim=5):" to "def convert_batch(batch, backend="pytorch", is_mlu=False, idim=40, odim=5): " |
| 592 | test/test_e2e_st.py:122 | change "if is_cuda:" to "if is_mlu: " |
| 593 | test/test_e2e_st.py:123 | change "xs = xs.cuda()" to "xs = xs.mlu() " |
| 594 | test/test_e2e_st.py:124 | change "ilens = ilens.cuda()" to "ilens = ilens.mlu() " |
| 595 | test/test_e2e_st.py:125 | change "ys_tgt = ys_tgt.cuda()" to "ys_tgt = ys_tgt.mlu() " |
| 596 | test/test_e2e_st.py:126 | change "ys_src = ys_src.cuda()" to "ys_src = ys_src.mlu() " |
| 597 | test/test_e2e_st.py:555 | change "not torch.cuda.is_available() and not chainer.cuda.available, reason="gpu required"" to "not torch.mlu.is_available() and not chainer.mlu.available, reason="gpu required" " |
| 598 | test/test_e2e_st.py:563 | change "batch = prepare_inputs("pytorch", is_cuda=True)" to "batch = prepare_inputs("pytorch", is_mlu=True) " |
| 599 | test/test_e2e_st.py:564 | change "model.cuda()" to "model.mlu() " |
| 600 | test/test_e2e_st.py:575 | change "@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")" to "@pytest.mark.skipif(torch.mlu.device_count() < 2, reason="multi gpu required") " |
| 601 | test/test_e2e_st.py:585 | change "batch = prepare_inputs("pytorch", is_cuda=True)" to "batch = prepare_inputs("pytorch", is_mlu=True) " |
| 602 | test/test_e2e_st.py:586 | change "model.cuda()" to "model.mlu() " |
| 603 | test/test_e2e_st_conformer.py:8 | add "import torch_mlu" |
| 604 | test/test_e2e_st_transformer.py:8 | add "import torch_mlu" |
| 605 | test/test_e2e_tts_fastspeech.py:15 | add "import torch_mlu" |
| 606 | test/test_e2e_tts_fastspeech.py:317 | change "@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")" to "@pytest.mark.skipif(not torch.mlu.is_available(), reason="gpu required") " |
| 607 | test/test_e2e_tts_fastspeech.py:356 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 608 | test/test_e2e_tts_fastspeech.py:409 | change "@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")" to "@pytest.mark.skipif(torch.mlu.device_count() < 2, reason="multi gpu required") " |
| 609 | test/test_e2e_tts_fastspeech.py:448 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 610 | test/test_e2e_tts_tacotron2.py:12 | add "import torch_mlu" |
| 611 | test/test_e2e_tts_tacotron2.py:201 | change "@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")" to "@pytest.mark.skipif(not torch.mlu.is_available(), reason="gpu required") " |
| 612 | test/test_e2e_tts_tacotron2.py:225 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 613 | test/test_e2e_tts_tacotron2.py:260 | change "@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")" to "@pytest.mark.skipif(torch.mlu.device_count() < 2, reason="multi gpu required") " |
| 614 | test/test_e2e_tts_tacotron2.py:278 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 615 | test/test_e2e_tts_transformer.py:11 | add "import torch_mlu" |
| 616 | test/test_e2e_tts_transformer.py:195 | change "@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")" to "@pytest.mark.skipif(not torch.mlu.is_available(), reason="gpu required") " |
| 617 | test/test_e2e_tts_transformer.py:223 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 618 | test/test_e2e_tts_transformer.py:259 | change "@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")" to "@pytest.mark.skipif(torch.mlu.device_count() < 2, reason="multi gpu required") " |
| 619 | test/test_e2e_tts_transformer.py:287 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 620 | test/test_e2e_vc_tacotron2.py:13 | add "import torch_mlu" |
| 621 | test/test_e2e_vc_tacotron2.py:200 | change "@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")" to "@pytest.mark.skipif(not torch.mlu.is_available(), reason="gpu required") " |
| 622 | test/test_e2e_vc_tacotron2.py:223 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 623 | test/test_e2e_vc_tacotron2.py:258 | change "@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")" to "@pytest.mark.skipif(torch.mlu.device_count() < 2, reason="multi gpu required") " |
| 624 | test/test_e2e_vc_tacotron2.py:275 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 625 | test/test_e2e_vc_transformer.py:12 | add "import torch_mlu" |
| 626 | test/test_e2e_vc_transformer.py:198 | change "@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")" to "@pytest.mark.skipif(not torch.mlu.is_available(), reason="gpu required") " |
| 627 | test/test_e2e_vc_transformer.py:226 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 628 | test/test_e2e_vc_transformer.py:262 | change "@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")" to "@pytest.mark.skipif(torch.mlu.device_count() < 2, reason="multi gpu required") " |
| 629 | test/test_e2e_vc_transformer.py:290 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 630 | test/test_initialization.py:8 | add "import torch_mlu" |
| 631 | test/test_lm.py:6 | add "import torch_mlu" |
| 632 | test/test_lm.py:146 | change "for device in ("cpu", "cuda")" to "for device in ("cpu", "mlu") " |
| 633 | test/test_lm.py:151 | change "if device == "cuda" and not torch.cuda.is_available():" to "if device == "mlu" and not torch.mlu.is_available(): " |
| 634 | test/test_lm.py:152 | change "pytest.skip("no cuda device is available")" to "pytest.skip("no mlu device is available") " |
| 635 | test/test_loss.py:6 | add "import torch_mlu" |
| 636 | test/test_multi_spkrs.py:12 | add "import torch_mlu" |
| 637 | test/test_optimizer.py:9 | add "import torch_mlu" |
| 638 | test/test_positional_encoding.py:2 | add "import torch_mlu" |
| 639 | test/test_positional_encoding.py:13 | change "[(dt, dv) for dt in ("float32", "float64") for dv in ("cpu", "cuda")]," to "[(dt, dv) for dt in ("float32", "float64") for dv in ("cpu", "mlu")], " |
| 640 | test/test_positional_encoding.py:16 | change "if device == "cuda" and not torch.cuda.is_available():" to "if device == "mlu" and not torch.mlu.is_available(): " |
| 641 | test/test_positional_encoding.py:17 | change "pytest.skip("no cuda device is available")" to "pytest.skip("no mlu device is available") " |
| 642 | test/test_positional_encoding.py:46 | change "for dv in ("cpu", "cuda")" to "for dv in ("cpu", "mlu") " |
| 643 | test/test_positional_encoding.py:52 | change "if device == "cuda" and not torch.cuda.is_available():" to "if device == "mlu" and not torch.mlu.is_available(): " |
| 644 | test/test_positional_encoding.py:53 | change "pytest.skip("no cuda device is available")" to "pytest.skip("no mlu device is available") " |
| 645 | test/test_positional_encoding.py:68 | change "[(dt, dv) for dt in ("float32", "float64") for dv in ("cpu", "cuda")]," to "[(dt, dv) for dt in ("float32", "float64") for dv in ("cpu", "mlu")], " |
| 646 | test/test_positional_encoding.py:71 | change "if device == "cuda" and not torch.cuda.is_available():" to "if device == "mlu" and not torch.mlu.is_available(): " |
| 647 | test/test_positional_encoding.py:72 | change "pytest.skip("no cuda device is available")" to "pytest.skip("no mlu device is available") " |
| 648 | test/test_recog.py:10 | add "import torch_mlu" |
| 649 | test/test_scheduler.py:4 | add "import torch_mlu" |
| 650 | test/test_torch.py:3 | add "import torch_mlu" |
| 651 | test/test_train_dtype.py:2 | add "import torch_mlu" |
| 652 | test/test_train_dtype.py:70 | change "for device in ("cpu", "cuda")" to "for device in ("cpu", "mlu") " |
| 653 | test/test_train_dtype.py:74 | change "if device == "cuda" and not torch.cuda.is_available():" to "if device == "mlu" and not torch.mlu.is_available(): " |
| 654 | test/test_train_dtype.py:75 | change "pytest.skip("no cuda device is available")" to "pytest.skip("no mlu device is available") " |
| 655 | test/test_transformer_decode.py:3 | add "import torch_mlu" |
| 656 | test/espnet2/asr/test_ctc.py:2 | add "import torch_mlu" |
| 657 | test/espnet2/asr/test_maskctc_model.py:2 | add "import torch_mlu" |
| 658 | test/espnet2/asr/decoder/test_mlm_decoder.py:2 | add "import torch_mlu" |
| 659 | test/espnet2/asr/decoder/test_rnn_decoder.py:2 | add "import torch_mlu" |
| 660 | test/espnet2/asr/decoder/test_transformer_decoder.py:2 | add "import torch_mlu" |
| 661 | test/espnet2/asr/encoder/test_conformer_encoder.py:2 | add "import torch_mlu" |
| 662 | test/espnet2/asr/encoder/test_contextual_block_transformer_encoder.py:2 | add "import torch_mlu" |
| 663 | test/espnet2/asr/encoder/test_longformer_encoder.py:2 | add "import torch_mlu" |
| 664 | test/espnet2/asr/encoder/test_rnn_encoder.py:2 | add "import torch_mlu" |
| 665 | test/espnet2/asr/encoder/test_transformer_encoder.py:2 | add "import torch_mlu" |
| 666 | test/espnet2/asr/encoder/test_vgg_rnn_encoder.py:2 | add "import torch_mlu" |
| 667 | test/espnet2/asr/frontend/test_frontend.py:2 | add "import torch_mlu" |
| 668 | test/espnet2/asr/frontend/test_fused.py:1 | add "import torch_mlu" |
| 669 | test/espnet2/asr/frontend/test_s3prl.py:1 | add "import torch_mlu" |
| 670 | test/espnet2/asr/frontend/test_windowing.py:1 | add "import torch_mlu" |
| 671 | test/espnet2/asr/postencoder/test_hugging_face_transformers_postencoder.py:2 | add "import torch_mlu" |
| 672 | test/espnet2/asr/preencoder/test_linear.py:1 | add "import torch_mlu" |
| 673 | test/espnet2/asr/preencoder/test_sinc.py:1 | add "import torch_mlu" |
| 674 | test/espnet2/asr/specaug/test_specaug.py:2 | add "import torch_mlu" |
| 675 | test/espnet2/asr/transducer/test_beam_search_transducer.py:2 | add "import torch_mlu" |
| 676 | test/espnet2/asr/transducer/test_error_calculator_transducer.py:2 | add "import torch_mlu" |
| 677 | test/espnet2/asr/transducer/test_transducer_decoder.py:2 | add "import torch_mlu" |
| 678 | test/espnet2/bin/test_diar_inference.py:5 | add "import torch_mlu" |
| 679 | test/espnet2/bin/test_enh_inference.py:6 | add "import torch_mlu" |
| 680 | test/espnet2/diar/test_espnet_model.py:2 | add "import torch_mlu" |
| 681 | test/espnet2/diar/attractor/test_rnn_attractor.py:2 | add "import torch_mlu" |
| 682 | test/espnet2/diar/decoder/test_linear_decoder.py:2 | add "import torch_mlu" |
| 683 | test/espnet2/enh/test_espnet_enh_s2t_model.py:2 | add "import torch_mlu" |
| 684 | test/espnet2/enh/test_espnet_model.py:2 | add "import torch_mlu" |
| 685 | test/espnet2/enh/decoder/test_conv_decoder.py:2 | add "import torch_mlu" |
| 686 | test/espnet2/enh/decoder/test_stft_decoder.py:2 | add "import torch_mlu" |
| 687 | test/espnet2/enh/encoder/test_conv_encoder.py:2 | add "import torch_mlu" |
| 688 | test/espnet2/enh/encoder/test_stft_encoder.py:2 | add "import torch_mlu" |
| 689 | test/espnet2/enh/layers/test_complex_utils.py:3 | add "import torch_mlu" |
| 690 | test/espnet2/enh/layers/test_conv_utils.py:2 | add "import torch_mlu" |
| 691 | test/espnet2/enh/layers/test_enh_layers.py:3 | add "import torch_mlu" |
| 692 | test/espnet2/enh/loss/criterions/test_tf_domain.py:2 | add "import torch_mlu" |
| 693 | test/espnet2/enh/loss/criterions/test_time_domain.py:2 | add "import torch_mlu" |
| 694 | test/espnet2/enh/loss/wrappers/test_dpcl_solver.py:2 | add "import torch_mlu" |
| 695 | test/espnet2/enh/loss/wrappers/test_fixed_order_solver.py:2 | add "import torch_mlu" |
| 696 | test/espnet2/enh/loss/wrappers/test_multilayer_pit_solver.py:2 | add "import torch_mlu" |
| 697 | test/espnet2/enh/loss/wrappers/test_pit_solver.py:2 | add "import torch_mlu" |
| 698 | test/espnet2/enh/separator/test_beamformer.py:2 | add "import torch_mlu" |
| 699 | test/espnet2/enh/separator/test_conformer_separator.py:2 | add "import torch_mlu" |
| 700 | test/espnet2/enh/separator/test_dan_separator.py:2 | add "import torch_mlu" |
| 701 | test/espnet2/enh/separator/test_dc_crn_separator.py:2 | add "import torch_mlu" |
| 702 | test/espnet2/enh/separator/test_dccrn_separator.py:2 | add "import torch_mlu" |
| 703 | test/espnet2/enh/separator/test_dpcl_e2e_separator.py:2 | add "import torch_mlu" |
| 704 | test/espnet2/enh/separator/test_dpcl_separator.py:2 | add "import torch_mlu" |
| 705 | test/espnet2/enh/separator/test_dprnn_separator.py:2 | add "import torch_mlu" |
| 706 | test/espnet2/enh/separator/test_dptnet_separator.py:2 | add "import torch_mlu" |
| 707 | test/espnet2/enh/separator/test_fasnet_separator.py:2 | add "import torch_mlu" |
| 708 | test/espnet2/enh/separator/test_rnn_separator.py:2 | add "import torch_mlu" |
| 709 | test/espnet2/enh/separator/test_skim_separator.py:2 | add "import torch_mlu" |
| 710 | test/espnet2/enh/separator/test_svoice_separator.py:2 | add "import torch_mlu" |
| 711 | test/espnet2/enh/separator/test_tcn_separator.py:2 | add "import torch_mlu" |
| 712 | test/espnet2/enh/separator/test_transformer_separator.py:2 | add "import torch_mlu" |
| 713 | test/espnet2/gan_tts/hifigan/test_hifigan.py:8 | add "import torch_mlu" |
| 714 | test/espnet2/gan_tts/jets/test_jets.py:7 | add "import torch_mlu" |
| 715 | test/espnet2/gan_tts/jets/test_jets.py:522 | change "not torch.cuda.is_available()," to "not torch.mlu.is_available(), " |
| 716 | test/espnet2/gan_tts/jets/test_jets.py:683 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 717 | test/espnet2/gan_tts/jets/test_jets.py:722 | change "not torch.cuda.is_available()," to "not torch.mlu.is_available(), " |
| 718 | test/espnet2/gan_tts/jets/test_jets.py:897 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 719 | test/espnet2/gan_tts/joint/test_joint_text2wav.py:7 | add "import torch_mlu" |
| 720 | test/espnet2/gan_tts/melgan/test_melgan.py:8 | add "import torch_mlu" |
| 721 | test/espnet2/gan_tts/parallel_wavegan/test_parallel_wavegan.py:8 | add "import torch_mlu" |
| 722 | test/espnet2/gan_tts/style_melgan/test_style_melgan.py:8 | add "import torch_mlu" |
| 723 | test/espnet2/gan_tts/vits/test_generator.py:7 | add "import torch_mlu" |
| 724 | test/espnet2/gan_tts/vits/test_vits.py:7 | add "import torch_mlu" |
| 725 | test/espnet2/gan_tts/vits/test_vits.py:578 | change "not torch.cuda.is_available()," to "not torch.mlu.is_available(), " |
| 726 | test/espnet2/gan_tts/vits/test_vits.py:735 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 727 | test/espnet2/gan_tts/vits/test_vits.py:785 | change "not torch.cuda.is_available()," to "not torch.mlu.is_available(), " |
| 728 | test/espnet2/gan_tts/vits/test_vits.py:958 | change "device = torch.device("cuda")" to "device = torch.device("mlu") " |
| 729 | test/espnet2/gan_tts/wavenet/test_wavenet.py:7 | add "import torch_mlu" |
| 730 | test/espnet2/hubert/test_hubert_loss.py:2 | add "import torch_mlu" |
| 731 | test/espnet2/iterators/test_sequence_iter_factory.py:2 | add "import torch_mlu" |
| 732 | test/espnet2/layers/test_global_mvn.py:5 | add "import torch_mlu" |
| 733 | test/espnet2/layers/test_label_aggregation.py:2 | add "import torch_mlu" |
| 734 | test/espnet2/layers/test_log_mel.py:1 | add "import torch_mlu" |
| 735 | test/espnet2/layers/test_mask_along_axis.py:2 | add "import torch_mlu" |
| 736 | test/espnet2/layers/test_sinc_filters.py:1 | add "import torch_mlu" |
| 737 | test/espnet2/layers/test_stft.py:1 | add "import torch_mlu" |
| 738 | test/espnet2/layers/test_time_warp.py:2 | add "import torch_mlu" |
| 739 | test/espnet2/layers/test_utterance_mvn.py:2 | add "import torch_mlu" |
| 740 | test/espnet2/lm/test_seq_rnn_lm.py:2 | add "import torch_mlu" |
| 741 | test/espnet2/lm/test_transformer_lm.py:2 | add "import torch_mlu" |
| 742 | test/espnet2/main_funcs/test_average_nbest_models.py:2 | add "import torch_mlu" |
| 743 | test/espnet2/main_funcs/test_calculate_all_attentions.py:5 | add "import torch_mlu" |
| 744 | test/espnet2/optimizers/test_sgd.py:1 | add "import torch_mlu" |
| 745 | test/espnet2/schedulers/test_noam_lr.py:1 | add "import torch_mlu" |
| 746 | test/espnet2/schedulers/test_warmup_lr.py:2 | add "import torch_mlu" |
| 747 | test/espnet2/tasks/test_abs_task.py:3 | add "import torch_mlu" |
| 748 | test/espnet2/torch_utils/test_add_gradient_noise.py:1 | add "import torch_mlu" |
| 749 | test/espnet2/torch_utils/test_device_funcs.py:5 | add "import torch_mlu" |
| 750 | test/espnet2/torch_utils/test_device_funcs.py:29 | change "@pytest.mark.skipif(not torch.cuda.is_available(), reason="Require cuda")" to "@pytest.mark.skipif(not torch.mlu.is_available(), reason="Require mlu") " |
| 751 | test/espnet2/torch_utils/test_device_funcs.py:30 | change "def test_to_device_cuda():" to "def test_to_device_mlu(): " |
| 752 | test/espnet2/torch_utils/test_device_funcs.py:32 | change "obj2 = to_device(obj, "cuda")" to "obj2 = to_device(obj, "mlu") " |
| 753 | test/espnet2/torch_utils/test_device_funcs.py:33 | change "assert obj2["a"][0].device == torch.device("cuda:0")" to "assert obj2["a"][0].device == torch.device("mlu:0") " |
| 754 | test/espnet2/torch_utils/test_device_funcs.py:50 | change "@pytest.mark.skipif(not torch.cuda.is_available(), reason="Require cuda")" to "@pytest.mark.skipif(not torch.mlu.is_available(), reason="Require mlu") " |
| 755 | test/espnet2/torch_utils/test_device_funcs.py:51 | change "def test_force_gatherable_cuda():" to "def test_force_gatherable_mlu(): " |
| 756 | test/espnet2/torch_utils/test_device_funcs.py:53 | change "obj2 = force_gatherable(obj, "cuda")" to "obj2 = force_gatherable(obj, "mlu") " |
| 757 | test/espnet2/torch_utils/test_device_funcs.py:54 | change "assert obj2["a"][0].device == torch.device("cuda:0")" to "assert obj2["a"][0].device == torch.device("mlu:0") " |
| 758 | test/espnet2/torch_utils/test_forward_adaptor.py:2 | add "import torch_mlu" |
| 759 | test/espnet2/torch_utils/test_initialize.py:2 | add "import torch_mlu" |
| 760 | test/espnet2/torch_utils/test_load_pretrained_model.py:2 | add "import torch_mlu" |
| 761 | test/espnet2/torch_utils/test_model_summary.py:1 | add "import torch_mlu" |
| 762 | test/espnet2/train/test_distributed_utils.py:44 | change "dist_backend="nccl"," to "dist_backend="cncl", " |
| 763 | test/espnet2/train/test_distributed_utils.py:61 | change "dist_backend="nccl"," to "dist_backend="cncl", " |
| 764 | test/espnet2/train/test_distributed_utils.py:78 | change "dist_backend="nccl"," to "dist_backend="cncl", " |
| 765 | test/espnet2/train/test_distributed_utils.py:94 | change "dist_backend="nccl"," to "dist_backend="cncl", " |
| 766 | test/espnet2/train/test_distributed_utils.py:111 | change "dist_backend="nccl"," to "dist_backend="cncl", " |
| 767 | test/espnet2/train/test_distributed_utils.py:128 | change "dist_backend="nccl"," to "dist_backend="cncl", " |
| 768 | test/espnet2/train/test_distributed_utils.py:145 | change "dist_backend="nccl"," to "dist_backend="cncl", " |
| 769 | test/espnet2/train/test_distributed_utils.py:163 | change "dist_backend="nccl"," to "dist_backend="cncl", " |
| 770 | test/espnet2/train/test_distributed_utils.py:181 | change "dist_backend="nccl"," to "dist_backend="cncl", " |
| 771 | test/espnet2/train/test_distributed_utils.py:331 | change "dist_backend="nccl"," to "dist_backend="cncl", " |
| 772 | test/espnet2/train/test_distributed_utils.py:359 | change "dist_backend="nccl"," to "dist_backend="cncl", " |
| 773 | test/espnet2/train/test_distributed_utils.py:388 | change "dist_backend="nccl"," to "dist_backend="cncl", " |
| 774 | test/espnet2/train/test_reporter.py:7 | add "import torch_mlu" |
| 775 | test/espnet2/tts/test_fastspeech.py:2 | add "import torch_mlu" |
| 776 | test/espnet2/tts/test_fastspeech2.py:2 | add "import torch_mlu" |
| 777 | test/espnet2/tts/test_tacotron2.py:2 | add "import torch_mlu" |
| 778 | test/espnet2/tts/test_transformer.py:2 | add "import torch_mlu" |
| 779 | test/espnet2/tts/feats_extract/test_dio.py:2 | add "import torch_mlu" |
| 780 | test/espnet2/tts/feats_extract/test_energy.py:2 | add "import torch_mlu" |
| 781 | test/espnet2/tts/feats_extract/test_linear_spectrogram.py:2 | add "import torch_mlu" |
| 782 | test/espnet2/tts/feats_extract/test_log_mel_fbank.py:2 | add "import torch_mlu" |
| 783 | test/espnet2/tts/feats_extract/test_log_spectrogram.py:2 | add "import torch_mlu" |
| 784 | tools/check_install.py:54 | add "import torch_mlu" |
| 785 | tools/check_install.py:58 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 786 | tools/check_install.py:59 | change "print(f"[x] torch cuda={torch.version.cuda}")" to "print(f"[x] torch mlu={torch.version.mlu}") " |
| 787 | tools/check_install.py:61 | change "print("[ ] torch cuda")" to "print("[ ] torch mlu") " |
| 788 | tools/check_install.py:68 | change "if torch.distributed.is_nccl_available():" to "if torch.distributed.is_cncl_available(): " |
| 789 | tools/check_install.py:69 | change "print("[x] torch nccl")" to "print("[x] torch cncl") " |
| 790 | tools/check_install.py:71 | change "print("[ ] torch nccl")" to "print("[ ] torch cncl") " |
| 791 | tools/check_install.py:86 | change "if chainer.backends.cuda.available:" to "if chainer.backends.mlu.available: " |
| 792 | tools/check_install.py:87 | change "print("[x] chainer cuda")" to "print("[x] chainer mlu") " |
| 793 | tools/check_install.py:89 | change "print("[ ] chainer cuda")" to "print("[ ] chainer mlu") " |
| 794 | tools/check_install.py:91 | change "if chainer.backends.cuda.cudnn_enabled:" to "if chainer.backends.mlu.cudnn_enabled: " |
| 795 | tools/check_install.py:104 | change "from cupy.cuda import nccl  # NOQA" to "from cupy.mlu import cncl  # NOQA " |
| 796 | tools/check_install.py:106 | change "print("[x] cupy nccl")" to "print("[x] cupy cncl") " |
| 797 | tools/check_install.py:108 | change "print("[ ] cupy nccl")" to "print("[ ] cupy cncl") " |
| 798 | utils/average_checkpoints.py:69 | add "import torch_mlu" |
| 799 | utils/generate_wav_from_fbank.py:17 | add "import torch_mlu" |
| 800 | utils/generate_wav_from_fbank.py:133 | change "device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")" to "device = torch.device("mlu") if torch.mlu.is_available() else torch.device("cpu") " |
