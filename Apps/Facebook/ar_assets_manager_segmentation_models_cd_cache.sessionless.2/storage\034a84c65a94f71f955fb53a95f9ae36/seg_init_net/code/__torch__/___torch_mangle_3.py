class SegmentationWrapper(Module):
  __parameters__ = []
  __annotations__ = []
  __annotations__["prepack_folding._jit_pass_packed_weight_0"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  __annotations__["prepack_folding._jit_pass_packed_weight_1"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  __annotations__["prepack_folding._jit_pass_packed_weight_2"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  __annotations__["prepack_folding._jit_pass_packed_weight_3"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  __annotations__["prepack_folding._jit_pass_packed_weight_4"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  __annotations__["prepack_folding._jit_pass_packed_weight_5"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  __annotations__["prepack_folding._jit_pass_packed_weight_6"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  __annotations__["prepack_folding._jit_pass_packed_weight_7"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  __annotations__["prepack_folding._jit_pass_packed_weight_8"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  __annotations__["prepack_folding._jit_pass_packed_weight_9"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  __annotations__["prepack_folding._jit_pass_packed_weight_10"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  __annotations__["prepack_folding._jit_pass_packed_weight_11"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  __annotations__["prepack_folding._jit_pass_packed_weight_12"] = __torch__.torch.classes.metal_unet.UnetConv2dOpContext
  optimized_for_metal : bool
  def get_metadata(self: __torch__.___torch_mangle_3.SegmentationWrapper) -> Tuple[int, int, str, str, int]:
    self_metadata = (192, 192, "segmentation", "1001", 1)
    return self_metadata
  def forward(self: __torch__.___torch_mangle_3.SegmentationWrapper,
    input: Tensor) -> Tensor:
    _0 = torch.sub(input, CONSTANTS.c0, alpha=1)
    input0 = torch.mul(_0, CONSTANTS.c1)
    _1 = getattr(self, "prepack_folding._jit_pass_packed_weight_0")
    _2 = ops.metal_prepack_unet.conv2d_run(input0, _1)
    input1 = torch.max_pool2d(_2, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input2 = torch.relu(input1)
    _3 = getattr(self, "prepack_folding._jit_pass_packed_weight_1")
    _4 = ops.metal_prepack_unet.conv2d_run(input2, _3)
    input3 = torch.max_pool2d(_4, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input4 = torch.relu(input3)
    _5 = getattr(self, "prepack_folding._jit_pass_packed_weight_2")
    _6 = ops.metal_prepack_unet.conv2d_run(input4, _5)
    input5 = torch.max_pool2d(_6, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input6 = torch.relu(input5)
    _7 = getattr(self, "prepack_folding._jit_pass_packed_weight_3")
    _8 = ops.metal_prepack_unet.conv2d_run(input6, _7)
    input7 = torch.max_pool2d(_8, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input8 = torch.relu(input7)
    _9 = getattr(self, "prepack_folding._jit_pass_packed_weight_4")
    _10 = ops.metal_prepack_unet.conv2d_run(input8, _9)
    input9 = torch.max_pool2d(_10, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input10 = torch.relu(input9)
    _11 = getattr(self, "prepack_folding._jit_pass_packed_weight_5")
    _12 = ops.metal_prepack_unet.conv2d_run(input10, _11)
    _13 = getattr(self, "prepack_folding._jit_pass_packed_weight_6")
    _14 = ops.metal_prepack_unet.conv2d_run(_12, _13)
    input11 = torch.upsample_nearest2d(_14, None, [2., 2.])
    input12 = torch.add(input11, _10, alpha=1)
    input13 = torch.relu(input12)
    _15 = getattr(self, "prepack_folding._jit_pass_packed_weight_7")
    _16 = ops.metal_prepack_unet.conv2d_run(input13, _15)
    input112 = torch.upsample_nearest2d(_16, None, [2., 2.])
    input14 = torch.add(input112, _8, alpha=1)
    input15 = torch.relu(input14)
    _17 = getattr(self, "prepack_folding._jit_pass_packed_weight_8")
    _18 = ops.metal_prepack_unet.conv2d_run(input15, _17)
    input116 = torch.upsample_nearest2d(_18, None, [2., 2.])
    input17 = torch.add(input116, _6, alpha=1)
    input16 = torch.relu(input17)
    _19 = getattr(self, "prepack_folding._jit_pass_packed_weight_9")
    _20 = ops.metal_prepack_unet.conv2d_run(input16, _19)
    input118 = torch.upsample_nearest2d(_20, None, [2., 2.])
    input19 = torch.add(input118, _4, alpha=1)
    input18 = torch.relu(input19)
    _21 = getattr(self, "prepack_folding._jit_pass_packed_weight_10")
    _22 = ops.metal_prepack_unet.conv2d_run(input18, _21)
    input120 = torch.upsample_nearest2d(_22, None, [2., 2.])
    input21 = torch.add(input120, _2, alpha=1)
    input20 = torch.relu(input21)
    _23 = getattr(self, "prepack_folding._jit_pass_packed_weight_11")
    _24 = ops.metal_prepack_unet.conv2d_run(input20, _23)
    input22 = torch.sigmoid(_24)
    _25 = getattr(self, "prepack_folding._jit_pass_packed_weight_12")
    _26 = ops.metal_prepack_unet.conv2d_run(input22, _25)
    return ops.metal.copy_to_host(_26)
