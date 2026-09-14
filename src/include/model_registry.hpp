/// \file model_registry.hpp
/// \brief merging the built-in model registry with the user-level ones (#30)
///
/// A registry FILE is not one entry. Picking the first model_list.json found
/// would lose a user's whole registry the moment a newer user directory gained
/// one, so the files are merged instead:
///
///   built-in, then each user file oldest first (utils::model_list_layers()).
///
/// Two rules decide every collision:
///   - a user file never replaces a built-in tag. The built-ins are what the
///     application ships and tests; a user entry under the same tag would make
///     `oflm run qwen3:8b` mean something different on one machine;
///   - between user files, the later one wins, so the newer directory
///     (.oflm) overrides the pre-rename one (.flm).
///
/// Only `models` is merged. Every other top-level key -- `model_path` among
/// them -- comes from the built-in file, because one process has one layout.
///
/// Nothing here touches the filesystem, so it is tested without one
/// (src/common/user_dirs_test.cpp).
#pragma once
#include "nlohmann/json.hpp"
#include <fstream>
#include <map>
#include <utility>
#include <string>
#include <vector>

namespace model_registry {

/// One registry to apply on top of the base, and where it came from.
struct Layer {
    std::string path;
    nlohmann::json doc;
};

/// What the merge did that a user should hear about, one line per fact.
struct Report {
    std::vector<std::string> notes;
    /// every user file that contributed at least one entry, and the tags it added. The
    /// tags are named, not just counted: an old full-copy registry also carries entries
    /// that have since left the built-in registry, and those merge as user models.
    struct Source {
        std::string path;
        std::vector<std::string> tags;
    };
    std::vector<Source> sources;
};

namespace detail {

inline std::string sample(const std::vector<std::string>& tags) {
    std::string s;
    for (size_t i = 0; i < tags.size() && i < 3; ++i) s += (i ? ", " : "") + tags[i];
    if (tags.size() > 3) s += ", ...";
    return s;
}

}  // namespace detail

/// \brief the first size under each model type of the base registry
/// \note A bare tag (`qwen3`) is itself a built-in tag. model_list resolves it to
///       the first size of its type; a user entry sorting before the built-in
///       sizes would otherwise change what the bare tag runs. Recorded before
///       the merge, used by model_list::rectify_model_tag.
inline std::map<std::string, std::string> default_sizes(const nlohmann::json& base) {
    std::map<std::string, std::string> out;
    if (!base.is_object() || !base.contains("models") || !base["models"].is_object()) return out;
    for (const auto& [type, sizes] : base["models"].items()) {
        if (sizes.is_object() && !sizes.empty()) out[type] = sizes.begin().key();
    }
    return out;
}

/// \brief merge user registries over a built-in one
/// \param base the built-in registry; returned with the user entries added
/// \param layers user registries in merge order (oldest first)
/// \param report receives one note per ignored or replaced group of entries
inline nlohmann::json merge(nlohmann::json base, const std::vector<Layer>& layers, Report* report) {
    auto note = [report](const std::string& s) { if (report) report->notes.push_back(s); };
    if (!base.contains("models") || !base["models"].is_object()) base["models"] = nlohmann::json::object();
    const nlohmann::json builtin = base["models"];
    std::map<std::string, std::string> owner;   // tag -> the user file that set it

    for (const Layer& layer : layers) {
        if (!layer.doc.is_object()) {
            note(layer.path + ": not a JSON object; ignored");
            continue;
        }
        if (!layer.doc.contains("models")) continue;
        if (!layer.doc["models"].is_object()) {
            note(layer.path + ": \"models\" is not an object; ignored");
            continue;
        }
        std::vector<std::string> shadowed, malformed;
        std::map<std::string, std::vector<std::string>> replaced;   // earlier file -> tags
        std::vector<std::string> added;
        for (const auto& [type, sizes] : layer.doc["models"].items()) {
            if (!sizes.is_object()) {
                malformed.push_back(type);
                continue;
            }
            for (const auto& [size, entry] : sizes.items()) {
                const std::string tag = type + ":" + size;
                if (!entry.is_object()) {
                    malformed.push_back(tag);
                    continue;
                }
                if (builtin.contains(type) && builtin[type].is_object() && builtin[type].contains(size)) {
                    // A copy of the built-in entry is what oflm-add used to seed a user
                    // file with; it is harmless and not worth a word. A DIFFERENT entry
                    // under a built-in tag is somebody's intention being ignored.
                    if (builtin[type][size] != entry) shadowed.push_back(tag);
                    continue;
                }
                auto prev = owner.find(tag);
                if (prev != owner.end() && base["models"][type][size] != entry) {
                    replaced[prev->second].push_back(tag);
                }
                base["models"][type][size] = entry;
                owner[tag] = layer.path;
                added.push_back(tag);
            }
        }
        if (!shadowed.empty()) {
            note(layer.path + ": " + std::to_string(shadowed.size()) +
                 " entr" + (shadowed.size() == 1 ? "y uses" : "ies use") +
                 " a built-in tag with different contents and " +
                 (shadowed.size() == 1 ? "was" : "were") +
                 " ignored; a user registry cannot replace a built-in model (" +
                 detail::sample(shadowed) + "). Older oflm-add versions copied every built-in "
                 "entry into this file; removing them silences this");
        }
        for (const auto& [earlier, tags] : replaced) {
            note(layer.path + ": replaces " + std::to_string(tags.size()) + " entr" +
                 (tags.size() == 1 ? "y" : "ies") + " from " + earlier + " (" + detail::sample(tags) + ")");
        }
        if (!malformed.empty()) {
            note(layer.path + ": " + std::to_string(malformed.size()) +
                 " malformed entr" + (malformed.size() == 1 ? "y" : "ies") + " ignored (" +
                 detail::sample(malformed) + ")");
        }
        if (!added.empty() && report) report->sources.push_back({layer.path, added});
    }
    return base;
}

/// \brief merge user model_info.json files over the shipped one (#30)
/// \note model_info.json is keyed by tag at the top level: each value lists a model's files
///       with their sizes and hashes. Same two rules as merge(): a user file never
///       replaces a shipped tag -- those hashes are what `oflm pull` checks downloads
///       against -- and a later user file wins over an earlier one.
inline nlohmann::json merge_model_info(nlohmann::json base, const std::vector<Layer>& layers) {
    if (!base.is_object()) base = nlohmann::json::object();
    const nlohmann::json shipped = base;
    for (const Layer& layer : layers) {
        if (!layer.doc.is_object()) continue;
        for (const auto& [tag, files] : layer.doc.items()) {
            if (shipped.contains(tag)) continue;
            base[tag] = files;
        }
    }
    return base;
}

/// \brief read and merge the files utils::find_model_infos() returns
/// \param paths the shipped file (may be an empty string: none), then user files in merge order
/// \note Silent, unlike the model list: this is read once per model on every `oflm list`,
///       and a user file that cannot be read only means its models go unverified.
/// \throw nlohmann::json::exception when the SHIPPED file exists but does not parse
inline nlohmann::json load_model_info(const std::vector<std::string>& paths) {
    nlohmann::json base = nlohmann::json::object();
    std::vector<Layer> layers;
    for (size_t i = 0; i < paths.size(); ++i) {
        if (paths[i].empty()) continue;
        std::ifstream f(paths[i]);
        if (i == 0) {
            if (f.is_open()) base = nlohmann::json::parse(f);
            continue;
        }
        try {
            if (f.is_open()) layers.push_back({paths[i], nlohmann::json::parse(f)});
        } catch (const std::exception&) {
            // A broken user file must not take the shipped entries down with it.
        }
    }
    return merge_model_info(std::move(base), layers);
}

}  // namespace model_registry
