/// \file pools.cpp
/// \brief The packing-plan interpreter (see pools.hpp).
#include "open_qwen36/pools.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace open_qwen36 {
namespace pools {

namespace {

std::string with_layer(const std::string& name, int layer) {
    std::string s = name;
    const std::string key = "{l}";
    for (size_t p = s.find(key); p != std::string::npos; p = s.find(key, p)) s.replace(p, key.size(), std::to_string(layer));
    return s;
}

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("pools: " + what); }

void bounds(const PackOp& op, uint64_t nbytes, size_t dst_bytes) {
    if (op.dst + nbytes > dst_bytes)
        fail(op.op + " " + (op.tensor.empty() ? op.up : op.tensor) + " writes " + std::to_string(nbytes) + " B at " +
             std::to_string(op.dst) + ", past the " + std::to_string(dst_bytes) + " B buffer");
}

/// pool chunk index -> file chunk index for a standard [out, in] matmul tensor:
/// pool chunk c covers rows 64*(c/per_band) + 32*(c%2) and cols
/// 1024*((c/8) % (in/1024)) + 256*((c/2)%4); file chunk f covers rows
/// 32*(f/ncol), cols 256*(f%ncol).
std::vector<size_t> std_perm(size_t nch, size_t in_dim) {
    size_t ncol = in_dim / 256, per_band = in_dim / 128, kgroups = in_dim >= 1024 ? in_dim / 1024 : 1;
    std::vector<size_t> perm(nch);
    for (size_t c = 0; c < nch; ++c) {
        size_t rows0 = 64 * (c / per_band) + 32 * (c % 2);
        size_t cols0 = (1024 * ((c / 8) % kgroups) + 256 * ((c / 2) % 4)) % in_dim;
        perm[c] = (rows0 / 32) * ncol + cols0 / 256;
    }
    return perm;
}

const uint8_t* raw(const Q4nxFile& m, const std::string& name, size_t need, size_t* got = nullptr) {
    size_t n = 0;
    const uint8_t* p = m.raw(name, &n);
    if (n < need) fail(name + " is " + std::to_string(n) + " B, the plan needs " + std::to_string(need));
    if (got) *got = n;
    return p;
}

}  // namespace

void apply(const PackOp& op, const Q4nxFile& m, int layer, uint8_t* dst, size_t dst_bytes, size_t ch) {
    if (op.op == "std_perm") {
        const std::string name = with_layer(op.tensor, layer);
        if (op.nch == 0 || op.in_dim == 0) fail("std_perm " + name + " without nch / in_dim");
        bounds(op, op.nch * ch, dst_bytes);
        const uint8_t* src = raw(m, name, (op.chunk0 + op.nch) * ch) + op.chunk0 * ch;
        auto perm = std_perm(op.nch, op.in_dim);
        for (size_t c = 0; c < op.nch; ++c) std::memcpy(dst + op.dst + c * ch, src + perm[c] * ch, ch);
    } else if (op.op == "expert_stripes") {
        // up / gate as interleaved [up_k | gate_k] stripes per expert, each stripe's chunks
        // transposed (pool chunk c <- file chunk ncol*(c%4) + c/4).
        const std::string un = with_layer(op.up, layer), gn = with_layer(op.gate, layer);
        const uint64_t S = op.stripe_bytes, ns = op.stripes, E = op.experts;
        if (!S || !ns || !E || !op.in_dim) fail("expert_stripes without stripe_bytes / stripes / experts / in_dim");
        bounds(op, E * 2 * ns * S, dst_bytes);
        const uint8_t* up = raw(m, un, E * ns * S);
        const uint8_t* gt = raw(m, gn, E * ns * S);
        const size_t ncol = op.in_dim / 256, nchs = S / ch;
        std::vector<size_t> tp(nchs);
        for (size_t c = 0; c < nchs; ++c) tp[c] = ncol * (c % 4) + c / 4;
        for (uint64_t e = 0; e < E; ++e) {
            for (uint64_t k = 0; k < ns; ++k) {
                const uint8_t* us = up + (ns * e + k) * S;
                const uint8_t* gs = gt + (ns * e + k) * S;
                uint8_t* ud = dst + op.dst + (2 * ns * e + 2 * k) * S;
                uint8_t* gd = ud + S;
                for (size_t c = 0; c < nchs; ++c) {
                    std::memcpy(ud + c * ch, us + tp[c] * ch, ch);
                    std::memcpy(gd + c * ch, gs + tp[c] * ch, ch);
                }
            }
        }
    } else if (op.op == "expert_down") {
        // down slices: pool chunk c <- file chunk 2*rt + cg, rt = 4*(c/8) + c%4, cg = (c/4)%2
        const std::string name = with_layer(op.tensor, layer);
        const uint64_t B = op.expert_bytes, E = op.experts;
        if (!B || !E) fail("expert_down without expert_bytes / experts");
        bounds(op, E * B, dst_bytes);
        const uint8_t* dn = raw(m, name, E * B);
        const size_t nchs = B / ch;
        for (uint64_t e = 0; e < E; ++e) {
            const uint8_t* ds = dn + e * B;
            uint8_t* dd = dst + op.dst + e * B;
            for (size_t c = 0; c < nchs; ++c) {
                size_t rt = 4 * (c / 8) + (c % 4), cg = (c / 4) % 2;
                std::memcpy(dd + c * ch, ds + (2 * rt + cg) * ch, ch);
            }
        }
    } else if (op.op == "put") {
        const std::string name = with_layer(op.tensor, layer);
        size_t n = 0;
        const uint8_t* src = raw(m, name, 0, &n);
        if (n > op.cap) fail(name + " is " + std::to_string(n) + " B, its slot holds " + std::to_string(op.cap));
        bounds(op, n, dst_bytes);
        std::memcpy(dst + op.dst, src, n);
    } else if (op.op == "conv_transpose") {
        // conv1d bf16 [taps][groups*width] -> [groups][taps][width]
        const std::string name = with_layer(op.tensor, layer);
        const uint64_t taps = op.taps, groups = op.groups, width = op.width;
        if (!taps || !groups || !width) fail("conv_transpose without taps / groups / width");
        size_t n = 0;
        const uint8_t* src = raw(m, name, taps * groups * width * 2, &n);
        if (n != taps * groups * width * 2) fail(name + " is not bf16[" + std::to_string(taps) + ", " + std::to_string(groups * width) + "]");
        bounds(op, n, dst_bytes);
        for (uint64_t g = 0; g < groups; ++g)
            for (uint64_t t = 0; t < taps; ++t)
                std::memcpy(dst + op.dst + (g * taps + t) * width * 2, src + (t * groups * width + g * width) * 2, width * 2);
    } else {
        fail("unknown pack op " + op.op);
    }
}

void pack_pool(const Manifest& m, const LayerType& lt, const Q4nxFile& f, int layer, uint8_t* dst) {
    std::memset(dst, 0, m.pool_bytes);
    for (const auto& op : lt.pool) apply(op, f, layer, dst, m.pool_bytes, m.chunk_bytes);
}

void pack_consts(const Manifest& m, const LayerType& lt, const Q4nxFile& f, int layer, uint8_t* dst) {
    std::memset(dst, 0, lt.consts_bytes);
    for (const auto& op : lt.consts) apply(op, f, layer, dst, lt.consts_bytes, m.chunk_bytes);
}

void pack_lmhead(const Manifest& m, const Q4nxFile& f, uint8_t* out) {
    // 128-row supertile order: pool chunk k <- file chunk (4*(k/32) + (k%4))*8 + ((k%32)/4)
    const size_t CH8 = m.lmhead_chunk_bytes;
    size_t n = 0;
    const uint8_t* src = f.raw(m.lmhead_tensor, &n);
    size_t nch = n / CH8;
    if (nch * CH8 > m.lmhead_pool_bytes) fail(m.lmhead_tensor + " is larger than its pool");
    std::memset(out, 0, m.lmhead_pool_bytes);
    for (size_t k = 0; k < nch; ++k) {
        size_t s = k / 32, r = k % 32;
        size_t fch = (4 * s + r % 4) * 8 + r / 4;
        std::memcpy(out + k * CH8, src + fch * CH8, CH8);
    }
}

void build_ptab(const Manifest& m, size_t rows, uint8_t* t) {
    // partial RoPE: rotary_dim of head_dim, half-split pairs (i, i + rot/2), the recipe's theta
    const size_t half = m.rotary_dim / 2;
    if (512 + 4 * half > 640 || 640 + 4 * half > m.ptab_row) fail("the rotary dim does not fit the position record");
    std::memset(t, 0, rows * m.ptab_row);
    for (size_t p = 0; p < rows; ++p) {
        uint8_t* r = t + p * m.ptab_row;
        int32_t pos = static_cast<int32_t>(p), nf = static_cast<int32_t>(p ? p : 1);
        std::memcpy(r, &pos, 4);
        std::memcpy(r + 4, &nf, 4);
        for (size_t i = 0; i < half; ++i) {
            double ang = static_cast<double>(p) * std::pow(m.rope_theta, -static_cast<double>(i) / static_cast<double>(half));
            float c = static_cast<float>(std::cos(ang)), s = static_cast<float>(std::sin(ang));
            std::memcpy(r + 512 + 4 * i, &c, 4);
            std::memcpy(r + 640 + 4 * i, &s, 4);
        }
    }
}

}  // namespace pools
}  // namespace open_qwen36
