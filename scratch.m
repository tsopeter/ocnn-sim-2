checkLayer(CustomPropagationLayer('no_name', 32, 32, 1, randn(16)+1i*randn(16)), {[32 32], [32 32]});
checkLayer(CustomDLPropagationLayer('no_name', 32, 32, 16, 16, 1, 1, 75e-2, 1550e-9), {[32, 32], [32, 32]});
