/// \file manifest_test.cpp
/// \brief OPEN-MANIFEST: the manifest parser reads the recipe's output, and a
///        model whose config.json disagrees with it is refused with the key named.
///        No XRT, no hardware: `manifest_test <fixtures/manifest_qwen36.json>`.
// Traces: OPEN-MANIFEST (canonical spec: specs/open-engine/spec.md)
#include <cstdio>
#include <stdexcept>
#include <string>

#include "open_qwen36/manifest.hpp"

using open_qwen36::Manifest;
using nlohmann::json;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    failures += !ok;
}

/// Expects check_model to throw with `needle` in the message.
void refused(const Manifest& m, const json& cfg, const std::string& needle, const std::string& what) {
    try {
        m.check_model(cfg, "test");
        check(false, what + " (accepted)");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        check(msg.find(needle) != std::string::npos, what + ": " + msg);
    }
}

json matching_config(const Manifest& m) {
    json cfg = m.hf_config_check;
    cfg["model_type"] = m.hf_config_check["model_type"][0];
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: manifest_test <manifest.json>\n");
        return 2;
    }
    Manifest m;
    try {
        m = Manifest::load(argv[1]);
    } catch (const std::exception& e) {
        std::printf("FAIL  load: %s\n", e.what());
        return 1;
    }
    // ---- what the recipe wrote for the 27B
    check(m.version == 1 && m.family == "qwen36moe", "version 1, family qwen36moe");
    check(m.layers.size() == 40 && m.layers[3] == "full_attention" && m.layers[0] == "linear_attention", "40 layers, attention every 4th");
    check(m.hidden == 2048 && m.vocab == 248320 && m.real_vocab == 248070, "hidden / vocab / real vocab");
    check(m.pool_bytes == 536870912 && m.lmhead_pool_bytes == 542113792 && m.chunk_bytes == 5120, "pool sizes");
    check(m.kv_row == 2048 && m.ptab_row == 1024 && m.rout_idx_off == 1024 && m.rotary_dim == 64, "kv row, ptab row, router idx, rotary dim");
    check(m.moe.stripe == 163840 && m.moe.up_bytes == 655360 && m.moe.down_core == 81920 && m.moe.pool_down == 335544320 &&
          m.moe.share_up == 503316480 && m.moe.share_gate == 503971840 && m.moe.share_down == 504627200 && m.moe.topk == 8,
          "MoE pool geometry");
    check(m.contexts.count("lx") && m.contexts.count("ax") && m.contexts.count("ln") && m.contexts.count("lm"), "four contexts");
    check(m.kernels.at("ax0").patch == "attnpos" && m.kernels.at("lx1").patch == "moeroute2" && m.kernels.at("ln").patch.empty(), "kernel patch kinds");
    const auto& lin = m.layer_types.at("linear_attention");
    const auto& full = m.layer_types.at("full_attention");
    check(lin.consts_bytes == 11882496 && lin.act_bytes == 190464 && lin.state_kind == "linear" && lin.state_bytes == 2342912, "linear layer buffers");
    check(full.consts_bytes == 1062912 && full.act_bytes == 98304 && full.state_kind == "kv" && full.state_row == 2048, "attention layer buffers");
    check(lin.program.size() == 3 && lin.program[0].op == "run" && lin.program[0].kernel == "lx0" && lin.program[0].args.size() == 5 &&
          lin.program[1].op == "moeroute2" && lin.program[1].act_off == 176128 && lin.program[2].kernel == "lx1", "linear program");
    check(full.program.size() == 3 && full.program[0].args.size() == 6 && full.program[0].args[5] == "ptab" &&
          full.program[1].act_off == 83968, "attention program");
    check(lin.pool.size() == 7 && full.pool.size() == 10 && lin.consts.size() == 11 && full.consts.size() == 6, "packing plans");
    check(lin.pool[0].op == "expert_stripes" && lin.pool[0].experts == 256 && lin.pool[1].op == "expert_down" &&
          lin.pool[5].op == "std_perm" && lin.pool[5].dst == 505282560 && lin.pool[5].nch == 2048, "pool plan ops");
    check(full.pool[8].op == "std_perm" && full.pool[8].chunk0 == 1024 && full.pool[8].dst == 511836160, "the fused q|gate split");
    check(m.tail.size() == 2 && m.tail[0].kernel == "ln" && m.tail[1].kernel == "lm" && m.tail[1].args.size() == 3, "tail program");
    check(m.globals.at("logits") == 248320 * 4 && m.globals.at("lmpool") == 542113792 && m.per_row_globals.at("ptab") == 1024, "globals");
    check(m.embed_tensor == "model.embed_tokens.weight" && m.norm_tensor == "model.norm.weight" && m.lmhead_tensor == "lm_head.weight", "tensor names");
    check(m.files().size() == 10, "10 files named (4 xclbin + 6 insts)");

    // ---- the model check
    json ok = matching_config(m);
    try {
        m.check_model(ok, "test");
        check(true, "a matching config.json is accepted");
    } catch (const std::exception& e) {
        check(false, std::string("a matching config.json is accepted: ") + e.what());
    }
    json interval = ok;
    interval.erase("layer_types");
    interval["full_attention_interval"] = 4;
    try {
        m.check_model(interval, "test");
        check(true, "full_attention_interval in place of layer_types is accepted");
    } catch (const std::exception& e) {
        check(false, std::string("full_attention_interval: ") + e.what());
    }
    json bad = ok; bad["hidden_size"] = 2560;
    refused(m, bad, "hidden_size", "hidden_size 2560 is refused by name");
    bad = ok; bad["model_type"] = "llama";
    refused(m, bad, "model_type", "model_type llama is refused");
    bad = ok; bad.erase("num_experts");
    refused(m, bad, "lacks 'num_experts'", "a missing key is named");
    bad = interval; bad["full_attention_interval"] = 5;
    refused(m, bad, "layer_types", "a different attention interval is refused");
    bad = ok; bad["num_hidden_layers"] = 24;
    refused(m, bad, "num_hidden_layers", "a 24-layer slice config is refused (the manifest is the 40-layer set)");

    // ---- a broken manifest
    json j = json::parse(std::string("{\"manifest_version\": 2}"));
    try {
        Manifest::parse(j, "broken");
        check(false, "manifest_version 2 is refused");
    } catch (const std::runtime_error& e) {
        check(std::string(e.what()).find("manifest_version 2") != std::string::npos, std::string("manifest_version 2 is refused: ") + e.what());
    }
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
