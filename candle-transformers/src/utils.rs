//! Shared utilities: repeat_kv, repeat_penalty, causal mask.

use candle::{CpuStorage, Device, Layout, Result, Shape, Tensor};

/// Build a causal attention mask of shape `(seq_len, kv_len)` where
/// `kv_len = index_pos + seq_len`.
///
/// `mask[i][j] = 1` means query `i` must **not** attend to key `j`.
///
/// - `index_pos == 0`: classic square `(seq_len, seq_len)` mask.
/// - `index_pos > 0`: rectangular mask for prefix KV caching — the first
///   `index_pos` columns are all-zero (every query attends to all cached prefix
///   keys) and the last `seq_len` columns form the standard causal triangle.
///
/// All models that maintain a KV cache should use this function so that
/// batched user-turn prefill works correctly after prefix restoration.
pub fn build_causal_mask(seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
    let kv_len = index_pos + seq_len;
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > index_pos + i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, kv_len), device)
}

pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    let token_ids = unique_token_ids(context);
    #[cfg(feature = "rocm")]
    if logits.device().is_rocm() {
        let logits = logits.to_dtype(candle::DType::F32)?;
        if token_ids.is_empty() {
            return Ok(logits);
        }
        let token_ids_len = token_ids.len();
        let token_ids = Tensor::from_vec(token_ids, token_ids_len, logits.device())?;
        return logits.apply_op2_no_bwd(&token_ids, &RepeatPenalty { penalty });
    }

    let device = logits.device();
    let mut logits = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
    for token_id in token_ids {
        if let Some(logit) = logits.get_mut(token_id as usize) {
            if *logit >= 0. {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
    let logits_len = logits.len();
    Tensor::from_vec(logits, logits_len, device)
}

fn unique_token_ids(context: &[u32]) -> Vec<u32> {
    let mut already_seen = std::collections::HashSet::new();
    context
        .iter()
        .copied()
        .filter(|token_id| already_seen.insert(*token_id))
        .collect()
}

struct RepeatPenalty {
    penalty: f32,
}

impl candle::CustomOp2 for RepeatPenalty {
    fn name(&self) -> &'static str {
        "repeat-penalty"
    }

    fn cpu_fwd(
        &self,
        logits: &CpuStorage,
        logits_layout: &Layout,
        token_ids: &CpuStorage,
        token_ids_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if logits_layout.dims().len() != 1 {
            candle::bail!(
                "repeat-penalty expects rank-1 logits, got {:?}",
                logits_layout.shape()
            )
        }
        if token_ids_layout.dims().len() != 1 {
            candle::bail!(
                "repeat-penalty expects rank-1 token ids, got {:?}",
                token_ids_layout.shape()
            )
        }
        let logits = logits.as_slice::<f32>()?;
        let token_ids = token_ids.as_slice::<u32>()?;
        let mut logits = (0..logits_layout.shape().elem_count())
            .map(|index| logits[logits_layout.start_offset() + index * logits_layout.stride()[0]])
            .collect::<Vec<_>>();
        for token_id_index in 0..token_ids_layout.shape().elem_count() {
            let token_id = token_ids
                [token_ids_layout.start_offset() + token_id_index * token_ids_layout.stride()[0]]
                as usize;
            if let Some(logit) = logits.get_mut(token_id) {
                if *logit >= 0. {
                    *logit /= self.penalty
                } else {
                    *logit *= self.penalty
                }
            }
        }
        Ok((CpuStorage::F32(logits), logits_layout.shape().clone()))
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        logits: &candle::RocmStorage,
        logits_layout: &Layout,
        token_ids: &candle::RocmStorage,
        token_ids_layout: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        logits.repeat_penalty(logits_layout, token_ids, token_ids_layout, self.penalty)
    }
}

/// Repeats a key or value tensor for grouped query attention
/// The input tensor should have a shape `(batch, num_kv_heads, seq_len, head_dim)`,
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}
