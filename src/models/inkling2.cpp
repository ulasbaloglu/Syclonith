#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP, hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP, hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE, hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE, hparams.n_shortconv_l_cache);
    ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);
    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA, hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);
    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }
    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta,
                       hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }
    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32 = true;
    hparams.expert_weights_norm_eps = 1e-20f;
    hparams.rope_pattern.fill(false);
    uint64_t max_channels = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }
        max_channels = std::max<uint64_t>(max_channels, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }
    if (max_channels * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }
    hparams.n_sconv_channels = max_channels;
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, hparams.n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, hparams.n_vocab_padded}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);
    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & l = layers[i];
        const int64_t nh = hparams.n_head(i);
        const int64_t nk = hparams.n_head_kv(i);
        const int64_t hd = hparams.n_embd_head_k(i);
        const int64_t nr = hparams.n_relative;
        const int64_t kernel = hparams.n_shortconv_l_cache;
        const int64_t extent = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;
        l.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, nh*hd}, 0);
        l.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, nk*hd}, 0);
        l.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, nk*hd}, 0);
        l.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {nh*hd, n_embd}, 0);
        l.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, nh*hd}, 0);
        l.attn_rel = create_tensor(tn(LLM_TENSOR_ATTN_REL, "weight", i), {n_embd, nh*nr}, 0);
        l.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {nr, extent}, 0);
        l.attn_q_clip = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP, "weight", i), {hd, nh}, 0);
        l.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {nr, nh}, 0);
        l.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, nh}, 0);
        l.attn_k_conv = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV, "weight", i), {kernel, nk*hd}, 0);
        l.attn_v_conv = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV, "weight", i), {kernel, nk*hd}, 0);
        l.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {kernel, n_embd}, 0);
        l.ffn_post_conv = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV, "weight", i), {kernel, n_embd}, 0);
        if (hparams.is_swa(i)) {
            l.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            l.attn_q_inp_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM, "weight", i), {n_embd}, 0);
            l.attn_k_inp_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM, "weight", i), {n_embd}, 0);
            l.attn_v_inp_norm = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM, "weight", i), {n_embd}, 0);
            l.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, kernel*nk}, 0);
            l.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, kernel*nk}, 0);
        }
        if (i > 0) {
            l.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        l.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);
        l.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        l.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        l.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
        l.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*hparams.n_ff_exp(i), n_expert}, 0);
        l.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {hparams.n_ff_exp(i), n_embd, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }
    // x*min(1, t/|x|) = t*x/max(|x|, t), and l2_norm(x, eps) = x/max(|x|, eps)
    return ggml_scale(ctx0, ggml_l2_norm(ctx0, cur, threshold), threshold);
}

// The normalized states of a list, last first: [norm(l_n-1), ..., norm(l_0)].
// Cache each prefix so blocks shared by multiple mix() calls are normalized and
// concatenated only once while this graph is being built.
ggml_tensor * llama_model_inkling2::graph::mix_stack(const std::vector<ggml_tensor *> & list) {
    GGML_ASSERT(!list.empty());

    const auto it = mix_stacks.find(list);
    if (it != mix_stacks.end()) {
        return it->second;
    }

    auto * res = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, list.back(), n_embd, 1, n_tokens), 1e-6f);
    if (list.size() > 1) {
        res = ggml_concat(ctx0, res, mix_stack(std::vector<ggml_tensor *>(list.begin(), list.end() - 1)), 1);
    }

    mix_stacks.emplace(list, res);
    return res;
}

// Attention over depth: mix the earlier block outputs and cur with learned queries.
ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    // [norm(b_n-1), ..., norm(b_0), norm(cur)]
    auto * states = mix_stack({cur});
    if (!blocks.empty()) {
        states = ggml_concat(ctx0, mix_stack(blocks), states, 1);
    }

    // With one token, query already has the target shape; avoid an identity repeat.
    auto * queries = n_tokens == 1 ? query :
        ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));
    auto * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);
    return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
}

ggml_tensor * llama_model_inkling2::graph::convolve(ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * dynamic,
        ggml_tensor * state, ggml_tensor * history, llm_graph_input_rs * inp, int64_t offset) {
    const int64_t kernel = hparams.n_shortconv_l_cache;
    const int64_t channels = cur->ne[0];
    const int64_t nt = ubatch.n_seq_tokens;
    const int64_t ns = ubatch.n_seqs;
    auto * past = ggml_view_3d(ctx0, history, kernel - 1, channels, ns, history->nb[1], history->nb[2], offset*history->nb[1]);
    auto * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, channels, nt, ns));
    x = ggml_concat(ctx0, past, x, 0);
    const auto * mctx_cur = inp->mctx;
    const int64_t state_cols = kernel - 1;
    const int64_t row_count = state_cols * channels;
    const size_t row_size = ggml_row_size(state->type, hparams.n_embd_r());
    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;
    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, x->ne[0] - state_cols - slot);
        ggml_tensor * tail = ggml_view_3d(ctx0, x, state_cols, channels, ns,
                x->nb[1], x->nb[2], ggml_row_size(x->type, s_idx));
        ggml_tensor * dst = ggml_view_2d(ctx0, state, row_count, ns, state->nb[1],
                (slot * mctx_cur->get_size() + mctx_cur->get_head()) * row_size + offset*state_cols*ggml_element_size(state));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));
    }
    auto * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, channels, n_tokens);
    if (dynamic) {
        const int64_t heads = dynamic->ne[0]/kernel;
        dynamic = ggml_reshape_3d(ctx0, dynamic, kernel, heads, n_tokens);
        for (int64_t lag = 0; lag < kernel; ++lag) {
            auto * tap = ggml_view_3d(ctx0, dynamic, 1, heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            auto * window = ggml_view_3d(ctx0, x, nt, channels, ns, x->nb[1], x->nb[2], (kernel - 1 - lag)*x->nb[0]);
            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, channels/heads, heads, n_tokens);
            auto * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), channels, n_tokens);
            out = ggml_add(ctx0, out, term);
        }
    }
    return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*kernel + 1));
}

ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * message, const loop_layer & loop, int ctrl) {
    // Normalize cur and message as separate rows of one 3D tensor. RMS norm
    // operates along ne[0], so this is equivalent to two independent norms.
    auto * gate = ggml_concat(ctx0,
            ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens),
            ggml_reshape_3d(ctx0, message, n_embd, 1, n_tokens), 1);
    gate = ggml_reshape_2d(ctx0, ggml_rms_norm(ctx0, gate, 1e-6f), 2*n_embd, n_tokens);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);
    auto * gain = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl], gate));
    auto * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));
    // (1 - forget)*cur + gain*message
    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    auto * inp = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    attn = inp->get_attn();
    recr = inp->get_recr();
    distances[0] = inp->self_rel_pos;
    distances[1] = inp->self_rel_pos_swa;
    auto * positions = build_inp_pos();
    out_ids = build_inp_out_ids();
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, positions, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);
    auto * cur = build_inp_embd(model.tok_embd);
    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }
    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, const loop_layer * loop) {
    GGML_ASSERT(!messages == !loop);
    auto message = [&](int ctrl) {
        return ggml_cont(ctx0, ggml_view_2d(ctx0, messages, n_embd, n_tokens, messages->nb[2], ctrl*messages->nb[1]));
    };
    const auto & l = model.layers[il];
    const bool swa = hparams.is_swa(il);
    const int64_t nh = hparams.n_head(il);
    const int64_t nk = hparams.n_head_kv(il);
    const int64_t hd = hparams.n_embd_head_k(il);
    const int64_t kv = nk*hd;
    auto * state = recr->mctx->get_r_l(il);
    auto * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);
    auto * residual = cur;
    ggml_tensor * inputs[3] = {cur, cur, cur};
    if (l.attn_res_query) {
        auto * mixed = mix(blocks, cur, l.attn_res_query);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], (swa ? 0 : j)*mixed->nb[1]));
        }
    }
    if (messages) {
        // the messages make the q, k and v inputs of a sliding layer differ
        for (int j = 0; j < 3; ++j) {
            inputs[j] = control(inputs[j], message(LOOP_CTRL_Q + j), *loop, LOOP_CTRL_Q + j);
        }
    }
    if (swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], l.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (swa) {
        inputs[0] = build_norm(inputs[0], l.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], l.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], l.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], l.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }
    auto * q = build_lora_mm(l.wq, inputs[0]);
    auto * k = convolve(build_lora_mm(l.wk, inputs[1]), l.attn_k_conv,
            swa ? nullptr : build_lora_mm(l.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
    auto * v = convolve(build_lora_mm(l.wv, inputs[2]), l.attn_v_conv,
            swa ? nullptr : build_lora_mm(l.attn_v_conv_proj, inputs[2]), state, history, recr, kv);
    auto * r = build_lora_mm(l.attn_rel, inputs[0]);
    q = clip(ggml_reshape_3d(ctx0, q, hd, nh, n_tokens), l.attn_q_clip, hparams.f_qk_clip);
    k = clip(ggml_reshape_3d(ctx0, k, hd, nk, n_tokens), nullptr, hparams.f_qk_clip);
    v = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, v, hd, nk, n_tokens), hparams.f_norm_rms_eps);
    r = clip(ggml_reshape_3d(ctx0, r, hparams.n_relative, nh, n_tokens), l.attn_rel_clip, hparams.f_relative_clip);
    if (!swa) {
        q = ggml_mul(ctx0, q, tau);
        r = ggml_mul(ctx0, r, tau);
    }
    auto * mask = swa ? attn->get_kq_mask_swa() : attn->get_kq_mask();
    const int64_t streams = mask->ne[3];
    const int64_t n_kv = distances[swa]->ne[0];
    auto * sinks = build_lora_mm(l.attn_sink_proj, inputs[0]);

    // Project relative features onto the bias table, then gather by query-to-cell distance.
    auto * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, l.attn_rel_bias, 0, 1, 0, 0), r);
    // Fold each query's sink logit into the bias: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))).
    table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, nh, n_tokens));
    if (!cparams.flash_attn) {
        // The dense mask is added after QK scaling in the non-flash path.
        table = ggml_scale(ctx0, table, sqrtf(hd));
    }
    table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));
    auto * bias = ggml_get_rows(ctx0, table, distances[swa]);
    bias = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_4d(ctx0, bias, nh, n_kv, n_tokens/streams, streams), 2, 0, 1, 3));

    // The sink is now represented in the bias; retain a zero-valued sink slot for softmax.
    // Reuse one zero tensor per head count across all attention layers in this graph.
    auto *& zero_sink = zero_sinks[nh];
    if (zero_sink == nullptr) {
        zero_sink = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, nh), 0.0f);
    }
    sinks = zero_sink;

    // Flash attention requires an F16 mask; both paths consume the bias as part of the mask.
    auto * kq_mask = ggml_cast(ctx0, mask, GGML_TYPE_F32);
    kq_mask = ggml_add(ctx0, bias, kq_mask);
    if (cparams.flash_attn) {
        kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
    }
    cur = build_attn(attn, nullptr, nullptr, nullptr, q, k, v, nullptr, sinks, nullptr, 1.0f/sqrtf(hd), il, kq_mask);
    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, build_lora_mm(l.wqkv_gate, inputs[0])));
    cur = convolve(build_lora_mm(l.wo, cur), l.attn_post_conv, nullptr, state, history, recr, 2*kv);
    if (ip % hparams.attn_res_block_size == 0) {
        blocks.push_back(residual);
    } else {
        cur = ggml_add(ctx0, residual, cur);
    }
    residual = cur;
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, l.ffn_res_query), n_embd, n_tokens);
    if (messages) {
        cur = control(cur, message(LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN);
    }
    cur = build_norm(cur, l.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cur = build_moe_ffn(cur, l.ffn_gate_inp, nullptr, nullptr, l.ffn_down_exps, l.ffn_exp_probs_b,
            hparams.n_expert, hparams.n_expert_used(il), LLM_FFN_SITU, true, hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, il, nullptr, l.ffn_gate_up_exps);
    cur = convolve(cur, l.ffn_post_conv, nullptr, state, history, recr, 2*kv + n_embd);
    cur = ggml_add(ctx0, residual, cur);
    cb(cur, "l_out", il);
    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    if (out_ids) {
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;
        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));
    res->t_logits = cur;
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
}

llama_model_inkling2::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params), model(model) {
    auto * cur = build_inputs();
    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < n_layer; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }
    build_output(cur, blocks);
}
