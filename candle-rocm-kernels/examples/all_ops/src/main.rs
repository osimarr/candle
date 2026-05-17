use candle::{
    quantized::{GgmlDType, QMatMul, QStorage, QTensor},
    DType, Device, Module, Result, Shape, Tensor,
};

macro_rules! run_kernel_ops_for_dtype {
    ($device:expr, $dtype:expr, $ggml_dtype:expr) => {{
        run_dtype_kernel_ops($device, $dtype, $ggml_dtype)?;
    }};
}

fn main() -> Result<()> {
    let device = Device::new_rocm(0)?;
    println!("running Candle ROCm kernel-op example on {device:?}");

    run_kernel_ops_for_dtype!(&device, DType::F32, Some(GgmlDType::F32));
    run_kernel_ops_for_dtype!(&device, DType::F16, Some(GgmlDType::F16));
    run_kernel_ops_for_dtype!(&device, DType::BF16, Some(GgmlDType::BF16));
    run_kernel_ops_for_dtype!(&device, DType::F8E4M3, None);

    device.synchronize()?;
    Ok(())
}

fn run_dtype_kernel_ops(
    device: &Device,
    dtype: DType,
    ggml_dtype: Option<GgmlDType>,
) -> Result<()> {
    println!("dtype {dtype:?}");
    run_device_ops(device, dtype)?;
    run_tensor_ops(device, dtype)?;
    run_copy_ops(device, dtype)?;
    run_quantized_ops(device, dtype, ggml_dtype)?;

    if supports_random(dtype) {
        run_random_ops(device, dtype)?;
    }
    if supports_const_set(dtype) {
        run_const_set_op(device, dtype)?;
    }
    if supports_nn_callbacks(dtype) {
        run_nn_ops(device, dtype)?;
    }
    Ok(())
}

fn run_device_ops(device: &Device, dtype: DType) -> Result<()> {
    device.set_seed(42)?;
    let _ = device.get_current_seed()?;

    let loaded = tensor(
        &[1., 2., 3., 4.],
        (2, 2),
        device,
        dtype,
        "storage_from_slice/to_dtype",
    )?;
    consume("storage_from_slice", &loaded)?;

    let zeros = Tensor::zeros((2, 2), dtype, device)?;
    consume("zeros", &zeros)?;
    Ok(())
}

fn run_random_ops(device: &Device, dtype: DType) -> Result<()> {
    let base = tensor(&[1., 2., 3., 4.], (2, 2), device, dtype, "random base")?;
    let uniform = base.rand_like(0., 1.)?;
    consume("rand_uniform", &uniform)?;
    let normal = base.randn_like(0., 1.)?;
    consume("rand_normal", &normal)?;
    Ok(())
}

fn run_const_set_op(device: &Device, dtype: DType) -> Result<()> {
    let filled = Tensor::zeros((2, 2), dtype, device)?;
    filled.one_set()?;
    consume("const_set", &filled)
}

fn run_tensor_ops(device: &Device, dtype: DType) -> Result<()> {
    let x = tensor(&[1., 2., 3., 4.], (2, 2), device, dtype, "x")?;

    consume("try_clone/contiguous", &x.t()?.contiguous()?)?;
    consume("affine", &x.affine(2., -1.)?)?;
    consume("powf", &x.powf(2.)?)?;
    consume(
        "elu",
        &tensor(&[-1., 0.5, 1., 2.], (2, 2), device, dtype, "elu input")?.elu(1.)?,
    )?;
    consume("unary", &x.sqr()?.sqrt()?)?;
    consume("binary", &(&x + &x)?)?;

    let cmp_rhs = tensor(&[2., 2., 2., 2.], (2, 2), device, dtype, "cmp rhs")?;
    consume("cmp", &x.gt(&cmp_rhs)?)?;

    consume("reduce_sum", &x.sum_keepdim(1)?)?;
    consume("reduce_max", &x.max_keepdim(0)?)?;
    consume("argsort", &x.arg_sort_last_dim(true)?)?;

    consume("to_dtype_f32", &x.to_dtype(DType::F32)?)?;
    if dtype != DType::F32 {
        consume(
            "to_dtype_roundtrip",
            &x.to_dtype(DType::F32)?.to_dtype(dtype)?,
        )?;
    }

    let mask = Tensor::new(&[[1u8, 0], [0, 1]], device)?;
    let on_true = tensor(&[10., 20., 30., 40.], (2, 2), device, dtype, "where true")?;
    let on_false = tensor(&[1., 2., 3., 4.], (2, 2), device, dtype, "where false")?;
    consume("where", &mask.where_cond(&on_true, &on_false)?)?;

    let ids = Tensor::new(&[1u32, 0], device)?;
    consume("index_select", &x.index_select(&ids, 0)?)?;

    let gather_src = tensor(
        &[10., 20., 30., 40., 50., 60.],
        (2, 3),
        device,
        dtype,
        "gather src",
    )?;
    let gather_ids = Tensor::new(&[[2u32, 0], [1, 2]], device)?;
    consume("gather", &gather_src.gather(&gather_ids, 1)?)?;

    let add_src = tensor(&[2., 3., 4., 5.], (2, 2), device, dtype, "index add src")?;
    consume("index_add", &x.index_add(&ids, &add_src, 1)?)?;

    let scatter_ids = Tensor::new(&[[1u32, 0], [0, 1]], device)?;
    consume(
        "scatter_set",
        &Tensor::zeros((2, 2), dtype, device)?.scatter(&scatter_ids, &add_src, 1)?,
    )?;
    consume(
        "scatter_add",
        &Tensor::zeros((2, 2), dtype, device)?.scatter_add(&scatter_ids, &add_src, 1)?,
    )?;

    let lhs = tensor(
        &[1., 2., 3., 4., 5., 6.],
        (2, 3),
        device,
        dtype,
        "matmul lhs",
    )?;
    let rhs = tensor(
        &[1., 2., 3., 4., 5., 6.],
        (3, 2),
        device,
        dtype,
        "matmul rhs",
    )?;
    consume("matmul", &lhs.matmul(&rhs)?)?;

    run_image_and_conv_ops(device, dtype)?;
    Ok(())
}

fn run_copy_ops(device: &Device, dtype: DType) -> Result<()> {
    let left = tensor(&[1., 2., 3., 4.], (2, 2), device, dtype, "cat left")?;
    let right = tensor(&[5., 6., 7., 8.], (2, 2), device, dtype, "cat right")?;
    consume("copy2d/cat", &Tensor::cat(&[left, right], 0)?)?;

    let dst = tensor(
        &[0., 0., 0., 0., 0., 0.],
        (3, 2),
        device,
        dtype,
        "slice dst",
    )?;
    let src = tensor(&[9., 8.], (1, 2), device, dtype, "slice src")?;
    consume(
        "copy_strided_src/slice_scatter",
        &dst.slice_scatter0(&src, 1)?,
    )?;
    Ok(())
}

fn run_image_and_conv_ops(device: &Device, dtype: DType) -> Result<()> {
    let image = tensor(&[1., 2., 3., 4.], (1, 1, 2, 2), device, dtype, "pool image")?;
    consume("avg_pool2d", &image.avg_pool2d((2, 2))?)?;
    consume("max_pool2d", &image.max_pool2d((2, 2))?)?;
    consume("upsample_nearest2d", &image.upsample_nearest2d(4, 4)?)?;
    consume(
        "upsample_bilinear2d",
        &image.upsample_bilinear2d(4, 4, false)?,
    )?;

    let sequence = tensor(&[1., 2.], (1, 1, 2), device, dtype, "upsample 1d")?;
    consume("upsample_nearest1d", &sequence.upsample_nearest1d(4)?)?;

    let conv1_input = tensor(&[1., 2., 3.], (1, 1, 3), device, dtype, "conv1d input")?;
    let conv1_kernel = tensor(&[1., 2.], (1, 1, 2), device, dtype, "conv1d kernel")?;
    consume("conv1d", &conv1_input.conv1d(&conv1_kernel, 0, 1, 1, 1)?)?;
    consume(
        "conv_transpose1d",
        &conv1_input.conv_transpose1d(&conv1_kernel, 0, 0, 1, 1, 1)?,
    )?;

    let conv2_input = tensor(
        &[1., 2., 3., 4.],
        (1, 1, 2, 2),
        device,
        dtype,
        "conv2d input",
    )?;
    let conv2_kernel = tensor(
        &[1., 2., 3., 4.],
        (1, 1, 2, 2),
        device,
        dtype,
        "conv2d kernel",
    )?;
    consume("conv2d", &conv2_input.conv2d(&conv2_kernel, 0, 1, 1, 1)?)?;
    consume(
        "conv_transpose2d",
        &conv2_input.conv_transpose2d(&conv2_kernel, 0, 0, 1, 1)?,
    )?;
    Ok(())
}

fn run_nn_ops(device: &Device, dtype: DType) -> Result<()> {
    let tensor_f32 = Tensor::new(
        &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]],
        device,
    )?;
    let tensor = tensor_f32.to_dtype(dtype)?;

    consume("nn_sigmoid", &candle_nn::ops::sigmoid(&tensor)?)?;

    let logits = tensor_f32.log()?.to_dtype(dtype)?;
    consume(
        "nn_softmax_last_dim",
        &candle_nn::ops::softmax_last_dim(&logits)?,
    )?;

    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?.to_dtype(dtype)?;
    let beta = Tensor::new(&[0.5f32, 0f32, -0.2f32], device)?.to_dtype(dtype)?;
    consume(
        "nn_rms_norm",
        &candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?,
    )?;
    consume(
        "nn_layer_norm",
        &candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?,
    )?;

    let (b_size, num_head, seq_len, head_dim) = (2, 2, 3, 4);
    let src_values = (0..b_size * num_head * seq_len * head_dim)
        .map(|v| v as f32 / 17.0 - 1.0)
        .collect::<Vec<_>>();
    let cos_values = (0..seq_len * head_dim / 2)
        .map(|v| (v as f32 / 7.0).cos())
        .collect::<Vec<_>>();
    let sin_values = (0..seq_len * head_dim / 2)
        .map(|v| (v as f32 / 7.0).sin())
        .collect::<Vec<_>>();
    let src = Tensor::from_vec(src_values, (b_size, num_head, seq_len, head_dim), device)?
        .to_dtype(dtype)?;
    let cos = Tensor::from_vec(cos_values, (seq_len, head_dim / 2), device)?.to_dtype(dtype)?;
    let sin = Tensor::from_vec(sin_values, (seq_len, head_dim / 2), device)?.to_dtype(dtype)?;

    consume(
        "nn_rope_i",
        &candle_nn::rotary_emb::rope_i(&src, &cos, &sin)?,
    )?;
    consume("nn_rope", &candle_nn::rotary_emb::rope(&src, &cos, &sin)?)?;
    let rope_thd =
        candle_nn::rotary_emb::rope_thd(&src.transpose(1, 2)?.contiguous()?, &cos, &sin)?;
    consume("nn_rope_thd", &rope_thd)?;
    Ok(())
}

fn run_quantized_ops(device: &Device, dtype: DType, ggml_dtype: Option<GgmlDType>) -> Result<()> {
    if let Some(ggml_dtype) = ggml_dtype {
        let src = tensor(&[1., -2., 3.5, 0.25], (2, 2), device, dtype, "quant src")?;
        let qtensor = QTensor::quantize(&src, ggml_dtype)?;
        let _ = qtensor.data()?;
        consume("quantize/dequantize/data", &qtensor.dequantize(device)?)?;

        if matches!(ggml_dtype, GgmlDType::F16 | GgmlDType::BF16) {
            run_quantized_from_data(device, ggml_dtype)?;
        }
    }

    if matches!(dtype, DType::F32 | DType::F16) {
        let activation_values = (0..64).map(|v| v as f32 / 64.0).collect::<Vec<_>>();
        let weight_values = (0..64).map(|v| v as f32 / 32.0).collect::<Vec<_>>();
        let activation = tensor(
            &activation_values,
            (2, 32),
            device,
            dtype,
            "qmatmul activation",
        )?;
        let weight = Tensor::from_slice(&weight_values, (32, 2), device)?;
        let qweight = QTensor::quantize(&weight.t()?, GgmlDType::Q4_0)?;
        consume(
            "quantized_matmul_t",
            &QMatMul::from_qtensor(qweight)?.forward(&activation)?,
        )?;
    }
    Ok(())
}

fn run_quantized_from_data(device: &Device, ggml_dtype: GgmlDType) -> Result<()> {
    let values = [1f32, -2., 3.5, 0.25];
    let raw = match ggml_dtype {
        GgmlDType::F16 => values
            .iter()
            .flat_map(|value| half::f16::from_f32(*value).to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
        GgmlDType::BF16 => values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
        _ => return Ok(()),
    };
    let storage = QStorage::from_data(
        std::borrow::Cow::Borrowed(raw.as_slice()),
        device,
        ggml_dtype,
    )?;
    let qtensor = QTensor::new(storage, (2, 2))?;
    consume("load_quantized/from_data", &qtensor.dequantize(device)?)
}

fn tensor<S: Into<Shape>>(
    data: &[f32],
    shape: S,
    device: &Device,
    dtype: DType,
    name: &'static str,
) -> Result<Tensor> {
    let tensor = Tensor::from_slice(data, shape, device)?;
    let tensor = if dtype == DType::F32 {
        tensor
    } else {
        tensor.to_dtype(dtype)?
    };
    consume(name, &tensor)?;
    Ok(tensor)
}

fn consume(name: &str, tensor: &Tensor) -> Result<()> {
    if !tensor.device().is_rocm() {
        candle::bail!("{name} left the ROCm device: {:?}", tensor.device())
    }
    match tensor.dtype() {
        DType::U8 => {
            let _ = tensor.flatten_all()?.to_vec1::<u8>()?;
        }
        DType::U32 => {
            let _ = tensor.flatten_all()?.to_vec1::<u32>()?;
        }
        DType::I64 => {
            let _ = tensor.flatten_all()?.to_vec1::<i64>()?;
        }
        _ => {
            let _ = tensor
                .to_dtype(DType::F32)?
                .flatten_all()?
                .sum_all()?
                .to_vec0::<f32>()?;
        }
    }
    println!("  {name}");
    Ok(())
}

fn supports_random(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::BF16 | DType::F8E4M3)
}

fn supports_const_set(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::BF16 | DType::F8E4M3)
}

fn supports_nn_callbacks(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::BF16 | DType::F8E4M3)
}
