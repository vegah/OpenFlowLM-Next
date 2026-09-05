/// \file manifest.cpp
/// \brief manifest.json parsing and the model check (see manifest.hpp).
#include "open_qwen36/manifest.hpp"

#include <fstream>
#include <stdexcept>

namespace open_qwen36 {

using nlohmann::json;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& what) {
    throw std::runtime_error("open_qwen36: " + where + ": " + what);
}

const json& need(const json& j, const char* key, const std::string& where) {
    if (!j.is_object() || !j.contains(key)) fail(where, std::string("lacks '") + key + "'");
    return j[key];
}

template <class T>
T get(const json& j, const char* key, const std::string& where) {
    try {
        return need(j, key, where).get<T>();
    } catch (const json::exception& e) {
        fail(where, std::string("'") + key + "': " + e.what());
    }
}

PackOp parse_op(const json& j, const std::string& where) {
    PackOp p;
    p.op = get<std::string>(j, "op", where);
    p.tensor = j.value("tensor", "");
    p.up = j.value("up", "");
    p.gate = j.value("gate", "");
    p.dst = j.value("dst", 0ull);
    p.cap = j.value("cap", 0ull);
    p.nch = j.value("nch", 0ull);
    p.in_dim = j.value("in_dim", 0ull);
    p.chunk0 = j.value("chunk0", 0ull);
    p.experts = j.value("experts", 0ull);
    p.stripes = j.value("stripes", 0ull);
    p.stripe_bytes = j.value("stripe_bytes", 0ull);
    p.expert_bytes = j.value("expert_bytes", 0ull);
    p.taps = j.value("taps", 0ull);
    p.groups = j.value("groups", 0ull);
    p.width = j.value("width", 0ull);
    if (p.op == "std_perm" || p.op == "put" || p.op == "expert_down" || p.op == "conv_transpose") {
        if (p.tensor.empty()) fail(where, p.op + " without a tensor");
    } else if (p.op == "expert_stripes") {
        if (p.up.empty() || p.gate.empty()) fail(where, "expert_stripes without up / gate");
    } else {
        fail(where, "unknown pack op '" + p.op + "'");
    }
    return p;
}

std::vector<Step> parse_program(const json& j, const std::string& where) {
    std::vector<Step> out;
    if (!j.is_array()) fail(where, "program is not a list");
    for (const auto& s : j) {
        Step st;
        st.op = get<std::string>(s, "op", where);
        st.kernel = get<std::string>(s, "kernel", where);
        if (st.op == "run") {
            st.args = get<std::vector<std::string>>(s, "args", where);
            if (st.args.empty() || st.args.size() > 8) fail(where, "run " + st.kernel + ": " + std::to_string(st.args.size()) + " buffer args (1..8)");
        } else if (st.op == "moeroute2") {
            st.act_off = get<uint64_t>(s, "act_off", where);
        } else {
            fail(where, "unknown program op '" + st.op + "'");
        }
        out.push_back(std::move(st));
    }
    return out;
}

}  // namespace

Manifest Manifest::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("open_qwen36: no manifest at " + path);
    json j = json::parse(f, nullptr, false);
    if (j.is_discarded()) throw std::runtime_error("open_qwen36: " + path + " is not JSON");
    return parse(j, path);
}

Manifest Manifest::parse(const json& j, const std::string& where) {
    Manifest m;
    m.version = get<int>(j, "manifest_version", where);
    if (m.version != 1) fail(where, "manifest_version " + std::to_string(m.version) + " (this engine reads 1)");
    m.family = get<std::string>(j, "family", where);
    m.spec_hash = j.value("spec_hash", "");
    m.build_key = j.value("build_key", "");
    m.max_ctx_default = j.value("max_ctx_default", 4096ull);

    const json& lay = need(j, "layout", where);
    const std::string lw = where + " layout";
    m.hidden = get<size_t>(lay, "hidden", lw);
    m.vocab = get<size_t>(lay, "vocab", lw);
    m.real_vocab = lay.value("real_vocab", m.vocab);
    m.chunk_bytes = get<size_t>(lay, "chunk_bytes", lw);
    m.pool_bytes = get<size_t>(lay, "pool_bytes", lw);
    m.lmhead_pool_bytes = get<size_t>(lay, "lmhead_pool_bytes", lw);
    m.lmhead_chunk_bytes = get<size_t>(lay, "lmhead_chunk_bytes", lw);
    m.kv_row = get<size_t>(lay, "kv_row", lw);
    m.ptab_row = get<size_t>(lay, "ptab_row", lw);
    m.rotary_dim = get<size_t>(lay, "rotary_dim", lw);
    m.rope_theta = get<double>(lay, "rope_theta", lw);
    m.rout_idx_off = get<size_t>(lay, "rout_idx_off", lw);
    const json& moe = need(lay, "moe", lw);
    const std::string mw = lw + ".moe";
    m.moe.experts = get<unsigned>(moe, "experts", mw);
    m.moe.topk = get<unsigned>(moe, "topk", mw);
    m.moe.stripe = get<uint64_t>(moe, "stripe", mw);
    m.moe.up_bytes = get<uint64_t>(moe, "up_bytes", mw);
    m.moe.down_core = get<uint64_t>(moe, "down_core", mw);
    m.moe.pool_down = get<uint64_t>(moe, "pool_down", mw);
    m.moe.share_up = get<uint64_t>(moe, "share_up", mw);
    m.moe.share_gate = get<uint64_t>(moe, "share_gate", mw);
    m.moe.share_down = get<uint64_t>(moe, "share_down", mw);
    if (m.moe.topk > 8 || m.moe.topk == 0) fail(mw, "topk " + std::to_string(m.moe.topk) + " (the router record holds 8)");
    m.attn.kv_row = m.kv_row;
    m.attn.ptab_row = m.ptab_row;

    m.layers = get<std::vector<std::string>>(j, "layers", where);
    if (m.layers.empty()) fail(where, "no layers");
    for (const auto& [k, v] : need(j, "contexts", where).items()) m.contexts[k] = v.get<std::string>();
    for (const auto& [k, v] : need(j, "kernels", where).items()) {
        KernelDesc d;
        d.context = get<std::string>(v, "context", where + " kernel " + k);
        d.insts = get<std::string>(v, "insts", where + " kernel " + k);
        d.patch = v.value("patch", "");
        if (!m.contexts.count(d.context)) fail(where, "kernel " + k + " names unknown context " + d.context);
        if (!d.patch.empty() && d.patch != "moeroute2" && d.patch != "attnpos") fail(where, "kernel " + k + ": unknown patch " + d.patch);
        m.kernels[k] = d;
    }
    for (const auto& [name, v] : need(j, "layer_types", where).items()) {
        const std::string tw = where + " layer type " + name;
        LayerType t;
        t.name = name;
        const json& b = need(v, "buffers", tw);
        t.consts_bytes = get<uint64_t>(b, "consts", tw);
        t.act_bytes = get<uint64_t>(b, "act", tw);
        const json& st = need(b, "state", tw);
        t.state_kind = get<std::string>(st, "kind", tw);
        if (t.state_kind == "linear") t.state_bytes = get<uint64_t>(st, "bytes", tw);
        else if (t.state_kind == "kv") t.state_row = get<uint64_t>(st, "row", tw);
        else fail(tw, "unknown state kind " + t.state_kind);
        t.program = parse_program(need(v, "program", tw), tw);
        for (const auto& s : t.program)
            if (!m.kernels.count(s.kernel)) fail(tw, "program names unknown kernel " + s.kernel);
        const json& pk = need(v, "pack", tw);
        for (const auto& o : need(pk, "pool", tw)) t.pool.push_back(parse_op(o, tw + " pack.pool"));
        for (const auto& o : need(pk, "consts", tw)) t.consts.push_back(parse_op(o, tw + " pack.consts"));
        m.layer_types[name] = std::move(t);
    }
    for (const auto& l : m.layers)
        if (!m.layer_types.count(l)) fail(where, "layers names unknown layer type " + l);
    m.tail = parse_program(need(j, "tail", where), where + " tail");
    for (const auto& s : m.tail)
        if (!m.kernels.count(s.kernel)) fail(where, "tail names unknown kernel " + s.kernel);
    for (const auto& [k, v] : need(j, "globals", where).items()) {
        if (v.is_number()) m.globals[k] = v.get<uint64_t>();
        else if (v.is_object() && v.contains("per_row")) m.per_row_globals[k] = v["per_row"].get<uint64_t>();
        else fail(where, "global " + k + " is neither a size nor {per_row}");
    }
    const json& pack = need(j, "pack", where);
    m.embed_tensor = get<std::string>(need(pack, "embed", where), "tensor", where + " pack.embed");
    m.norm_tensor = get<std::string>(need(pack, "norm", where), "tensor", where + " pack.norm");
    m.norm_bytes = get<size_t>(need(pack, "norm", where), "bytes", where + " pack.norm");
    m.lmhead_tensor = get<std::string>(need(pack, "lm_head", where), "tensor", where + " pack.lm_head");
    m.hf_config_check = need(j, "hf_config_check", where);
    return m;
}

const LayerType& Manifest::layer_type(size_t layer) const {
    if (layer >= layers.size()) throw std::runtime_error("open_qwen36: layer " + std::to_string(layer) + " beyond the manifest");
    return layer_types.at(layers[layer]);
}

std::vector<std::string> Manifest::files() const {
    std::vector<std::string> f;
    for (const auto& [k, v] : contexts) f.push_back(v);
    for (const auto& [k, v] : kernels) f.push_back(v.insts);
    return f;
}

void Manifest::check_model(const json& config, const std::string& where) const {
    if (!config.is_object()) fail(where, "config.json is not an object");
    auto lacks = [&](const std::string& key) { fail(where, "config.json lacks '" + key + "'"); };
    for (const auto& [key, want] : hf_config_check.items()) {
        if (key == "model_type") {
            if (!config.contains(key)) lacks(key);
            bool ok = false;
            for (const auto& t : want) ok |= (config[key] == t);
            if (!ok) fail(where, "config.json model_type " + config[key].dump() + " is not one this kernel set serves (" + want.dump() + ")");
        } else if (key == "layer_types") {
            std::vector<std::string> got;
            if (config.contains("layer_types")) {
                got = config["layer_types"].get<std::vector<std::string>>();
            } else if (config.contains("full_attention_interval") && config.contains("num_hidden_layers")) {
                int iv = config["full_attention_interval"].get<int>(), n = config["num_hidden_layers"].get<int>();
                if (iv <= 0) fail(where, "config.json full_attention_interval must be positive");
                for (int l = 0; l < n; ++l) got.push_back((l + 1) % iv == 0 ? "full_attention" : "linear_attention");
            } else {
                lacks("layer_types");
            }
            if (got != want.get<std::vector<std::string>>())
                fail(where, "config.json layer_types differ from the kernel set's (" + std::to_string(got.size()) + " layers)");
        } else {
            if (!config.contains(key)) lacks(key);
            if (config[key] != want)
                fail(where, "config.json " + key + " = " + config[key].dump() + ", the kernel set was built for " + want.dump());
        }
    }
}

}  // namespace open_qwen36
