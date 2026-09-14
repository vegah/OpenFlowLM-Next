/// \file user_dirs_test.cpp
/// \brief Unit tests for the user-directory search (#30): which registries are read
///        and merged, where an installed model is found, and which model_info.json
///        files verify it.
///
/// Every defect here is a well-formed wrong answer. A registry that is not read
/// shows fewer models; a model looked up in the wrong directory shows as "not
/// downloaded" and is pulled again; a model_info.json nobody merges leaves an added
/// model unverifiable. None of it crashes.
///
/// No device, no weights, no network. Everything runs in a temporary directory,
/// through the same functions the application calls: the functions that normally
/// read the user's profile take it as a parameter, and the ones that read
/// environment variables get them set here.
///
///   ctest --test-dir build -R user_dirs
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "model_list.hpp"
#include "model_registry.hpp"
#include "utils/utils.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

static int failures = 0;
static int checks = 0;

static void ok(bool cond, const std::string& what) {
    ++checks;
    if (cond) {
        std::printf("ok    %s\n", what.c_str());
    } else {
        ++failures;
        std::printf("FAIL  %s\n", what.c_str());
    }
}

static void eq(const std::string& got, const std::string& want, const std::string& what) {
    ok(got == want, what + (got == want ? "" : "  (got \"" + got + "\", want \"" + want + "\")"));
}

static void eqi(long long got, long long want, const std::string& what) {
    ok(got == want, what + (got == want ? "" : "  (got " + std::to_string(got) +
                                              ", want " + std::to_string(want) + ")"));
}

static bool same(const std::string& a, const std::string& b) {
    std::error_code ec;
    return fs::equivalent(a, b, ec);
}

static void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    if (value.empty()) unsetenv(name); else setenv(name, value.c_str(), 1);
#endif
}

static void write_json(const fs::path& p, const json& j) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << j.dump(2);
}

static bool has_note(const model_registry::Report& r, const std::string& needle) {
    for (const std::string& n : r.notes) if (n.find(needle) != std::string::npos) return true;
    return false;
}

static json entry(const std::string& name, const std::string& family = "qwen3") {
    return {{"name", name}, {"files", {"config.json", "model.q4nx"}}, {"details", {{"family", family}}}};
}

/// A model directory holding `files` (default: everything entry() lists).
static void install(const fs::path& dir, const std::vector<std::string>& files = {"config.json", "model.q4nx"}) {
    fs::create_directories(dir);
    for (const std::string& f : files) std::ofstream(dir / f) << "x";
}

static json builtin_registry() {
    return {{"model_path", "models"},
            {"models", {{"qwen3", {{"4b", entry("Qwen3-4B-NPU2")}, {"8b", entry("Qwen3-8B-NPU2")}}},
                        {"llama3.2", {{"1b", entry("Llama-3.2-1B-NPU2", "llama")}}}}}};
}

// ---------------------------------------------------------------------------
// USERDIR-ORDER: the user directories, newest name first.
// ---------------------------------------------------------------------------
static void test_user_directories() {
    std::printf("\n-- user_directories --\n");
    const std::vector<std::string> d = utils::user_directories("U");
#ifdef _WIN32
    eqi((long long)d.size(), 4, "four directories on Windows");
    if (d.size() == 4) {
        eq(d[0], (fs::path("U") / ".oflm").string(), "1st: .oflm (what #30 asks for)");
        eq(d[1], (fs::path("U") / ".config" / "oflm").string(), "2nd: .config/oflm (where oflm-add writes)");
        eq(d[2], (fs::path("U") / ".flm").string(), "3rd: the pre-rename .flm");
        eq(d[3], (fs::path("U") / ".config" / "flm").string(), "4th: the pre-rename .config/flm");
    }
#else
    eqi((long long)d.size(), 2, "two directories on POSIX");
    if (d.size() == 2) {
        eq(d[0], "U/oflm", "1st: oflm");
        eq(d[1], "U/flm", "2nd: the pre-rename flm");
    }
#endif
}

// ---------------------------------------------------------------------------
// USERDIR-REGISTRY-LAYERS: which registry files are read, in which order.
// ---------------------------------------------------------------------------
static void test_model_list_layers(const fs::path& tmp) {
    std::printf("\n-- model_list_layers --\n");
    const fs::path home = tmp / "layers_home";
    const std::vector<std::string> dirs = utils::user_directories(home.string());
    const fs::path builtin = tmp / "layers_share" / "model_list.json";
    write_json(builtin, builtin_registry());

    std::vector<std::string> l = utils::model_list_layers(builtin.string(), dirs);
    eqi((long long)l.size(), 1, "no user files: the built-in registry alone");

    // A registry in the newest and in the oldest directory.
    write_json(fs::path(dirs.front()) / "model_list.json", json::object());
    write_json(fs::path(dirs.back()) / "model_list.json", json::object());
    l = utils::model_list_layers(builtin.string(), dirs);
    eqi((long long)l.size(), 3, "two user files: three layers");
    if (l.size() == 3) {
        ok(same(l[0], builtin.string()), "the built-in registry is the base");
        ok(same(l[1], (fs::path(dirs.back()) / "model_list.json").string()),
           "the pre-rename file is applied first");
        ok(same(l[2], (fs::path(dirs.front()) / "model_list.json").string()),
           "the newest file is applied last, so it wins");
    }

    // A user directory that IS where the built-in registry lives is not read twice.
    const std::vector<std::string> self{builtin.parent_path().string()};
    eqi((long long)utils::model_list_layers(builtin.string(), self).size(), 1,
        "a user file equivalent to the base is not a second layer");

    // registry_layers: an explicit registry is the whole registry -- unless it IS the built-in one.
    const fs::path custom = tmp / "layers_custom" / "model_list.json";
    write_json(custom, builtin_registry());
    l = utils::registry_layers(custom.string(), builtin.string(), dirs);
    ok(l.size() == 1 && l[0] == custom.string(), "an explicit custom registry is read alone");
    l = utils::registry_layers(builtin.string(), builtin.string(), dirs);
    eqi((long long)l.size(), 3, "an explicit path naming the built-in list still merges the user files (oflm_env.sh)");
    eqi((long long)utils::registry_layers("", builtin.string(), dirs).size(), 3, "no explicit path: merged");
    l = utils::registry_layers(custom.string(), "", dirs);
    ok(l.size() == 1 && l[0] == custom.string(), "an explicit registry with no built-in one is read alone");

    // first_builtin_model_list: a user registry is never the base.
    const std::string user_file = (fs::path(dirs.front()) / "model_list.json").string();
    eq(utils::first_builtin_model_list({user_file, builtin.string()}, dirs), builtin.string(),
       "a candidate that is a user registry (`cd ~/.config/oflm && oflm list`) is skipped");
    eq(utils::first_builtin_model_list({(tmp / "missing.json").string()}, dirs), "",
       "no candidate exists: empty");
    const auto bc = utils::builtin_model_list_candidates("E", "P");
    ok(bc.size() == 4 && bc[1] == "model_list.json", "built-in candidates: exe dir, CWD, bundle, prefix");
}

// ---------------------------------------------------------------------------
// USERDIR-REGISTRY-MERGE: the merge rules.
// ---------------------------------------------------------------------------
static void test_merge() {
    std::printf("\n-- model_registry::merge --\n");
    json legacy = builtin_registry();                       // the old full-copy seeding
    legacy["models"]["qwen3"]["8b"]["name"] = "Stale-8B";  // ...gone stale for one entry
    legacy["models"]["mine"]["1b"] = entry("Mine-Old");
    legacy["models"]["mine"]["2b"] = entry("Mine-2B");
    json newest = {{"model_path", "elsewhere"},
                   {"models", {{"mine", {{"1b", entry("Mine-New")}}},
                               {"qwen3", {{"0.5b", entry("Qwen3-Custom")}}},
                               {"bad", {{"x", "not an object"}}}}}};

    model_registry::Report r;
    json m = model_registry::merge(builtin_registry(), {{"legacy.json", legacy}, {"new.json", newest}}, &r);

    eq(m["models"]["qwen3"]["4b"]["name"].get<std::string>(), "Qwen3-4B-NPU2", "an identical copy of a built-in changes nothing");
    eq(m["models"]["qwen3"]["8b"]["name"].get<std::string>(), "Qwen3-8B-NPU2", "a user file cannot replace a built-in entry");
    ok(has_note(r, "legacy.json: 1 entry uses a built-in tag"), "...and says so, naming the file");
    ok(!has_note(r, "qwen3:4b"), "...but not about the identical copy");
    eq(m["models"]["mine"]["2b"]["name"].get<std::string>(), "Mine-2B", "a user entry from the pre-rename file is kept");
    eq(m["models"]["mine"]["1b"]["name"].get<std::string>(), "Mine-New", "the newer user file wins over the older one");
    ok(has_note(r, "new.json: replaces 1 entry from legacy.json (mine:1b)"), "...and says which");
    eq(m["models"]["qwen3"]["0.5b"]["name"].get<std::string>(), "Qwen3-Custom", "a new size under a built-in type is added");
    eq(m["model_path"].get<std::string>(), "models", "model_path comes from the built-in registry");
    ok(!m["models"].contains("bad") || !m["models"]["bad"].contains("x"), "a malformed entry is not added");
    ok(has_note(r, "malformed"), "...and is reported");
    eqi((long long)r.sources.size(), 2, "both user files are reported as sources");
    if (r.sources.size() == 2) {
        ok(r.sources[0].tags == std::vector<std::string>{"mine:1b", "mine:2b"},
           "...naming the tags each one added, not just a count");
    }

    model_registry::Report r2;
    json m2 = model_registry::merge(builtin_registry(), {{"broken.json", json::array()}}, &r2);
    eqi((long long)m2["models"].size(), 2, "a user file that is not an object leaves the built-ins intact");
    ok(has_note(r2, "broken.json: not a JSON object"), "...and is reported");

    const auto defaults = model_registry::default_sizes(builtin_registry());
    eq(defaults.at("qwen3"), "4b", "default_sizes records the built-in first size");
}

// ---------------------------------------------------------------------------
// USERDIR-MODEL-PER-ENTRY: model_list on real files.
// ---------------------------------------------------------------------------
static void test_model_list(const fs::path& tmp) {
    std::printf("\n-- model_list over merged registries and several model roots --\n");
    const fs::path home = tmp / "ml_home";
    const std::vector<std::string> roots = utils::user_directories(home.string());
    const std::string newest = roots.front(), legacy = roots.back();
    // Where oflm-add writes by default: <user>\.config\oflm on Windows, the one oflm
    // directory on POSIX. Either way a different directory from `legacy`.
    const std::string added = roots.size() > 2 ? roots[1] : roots[0];

    const fs::path builtin = tmp / "ml_share" / "model_list.json";
    write_json(builtin, builtin_registry());
    json user = {{"model_path", "models"},
                 {"models", {{"mine", {{"1b", entry("Mine-1B")}}},
                             {"qwen3", {{"0.5b", entry("Qwen3-Tiny")}}}}}};
    write_json(fs::path(added) / "model_list.json", user);
    fs::path broken = fs::path(legacy) / "model_list.json";
    fs::create_directories(broken.parent_path());
    std::ofstream(broken) << "{ not json";

    // Qwen3-8B is complete in the pre-rename directory and PARTIAL in the new one;
    // Qwen3-4B is in the new one; Llama is only a partial copy in the pre-rename one.
    install(fs::path(legacy) / "models" / "Qwen3-8B-NPU2");
    install(fs::path(newest) / "models" / "Qwen3-8B-NPU2", {"config.json"});
    install(fs::path(newest) / "models" / "Qwen3-4B-NPU2");
    install(fs::path(added) / "models" / "Mine-1B");
    install(fs::path(legacy) / "models" / "Llama-3.2-1B-NPU2", {"config.json"});

    const std::vector<std::string> layers = utils::model_list_layers(builtin.string(), roots);
    model_list ml(layers, newest, roots);

    ok(ml.is_model_supported("qwen3:4b"), "a built-in tag is listed");
    ok(ml.is_model_supported("mine:1b"), "a user tag is listed without OFLM_CONFIG_PATH");
    ok(!ml.is_model_supported("nope:1b"), "an unknown tag is not");
    ok(same(ml.get_model_path("qwen3:4b"), (fs::path(newest) / "models" / "Qwen3-4B-NPU2").string()),
       "a model in the new directory is found there");
    ok(same(ml.get_model_path("qwen3:8b"), (fs::path(legacy) / "models" / "Qwen3-8B-NPU2").string()),
       "a complete copy in the PRE-RENAME directory is found, even past a partial one in the new");
    ok(same(ml.get_model_path("mine:1b"), (fs::path(added) / "models" / "Mine-1B").string()),
       "a user model in oflm-add's default directory is found");
    eq(ml.get_model_path("llama3.2:1b"), (fs::path(newest) / "models" / "Llama-3.2-1B-NPU2").string(),
       "a model that is only a partial copy elsewhere resolves to the default directory, where pull writes");
    fs::remove_all(fs::path(legacy) / "models" / "Llama-3.2-1B-NPU2");
    eq(ml.get_model_path("llama3.2:1b"), (fs::path(newest) / "models" / "Llama-3.2-1B-NPU2").string(),
       "a model installed nowhere resolves to the default directory");
    eq(ml.rectify_model_tag("qwen3"), "qwen3:4b",
       "a bare built-in tag keeps its built-in size when a user size sorts first");
    ok(ml.is_model_supported("qwen3:0.5b"), "...while the user size is still listed");

    std::string one = builtin.string(), dir = newest;
    model_list single(one, dir);
    ok(!single.is_model_supported("mine:1b"), "the one-file constructor reads one file, as before");
}

// ---------------------------------------------------------------------------
// USERDIR-MODEL-ROOTS: an explicit OFLM_MODEL_PATH is the one store.
// ---------------------------------------------------------------------------
static void test_models_search_roots() {
    std::printf("\n-- models_search_roots --\n");
    const auto e = utils::models_search_roots("X", "U");
    eqi((long long)e.size(), 1, "explicit path: exactly one root");
    if (!e.empty()) eq(e[0], "X", "...and it is the explicit one");
    const auto u = utils::models_search_roots("", "U");
    ok(u == utils::user_directories("U"), "no explicit path: every user directory, in order");
}

// ---------------------------------------------------------------------------
// USERDIR-REGISTRY-DIRS: where user registries are looked for.
// ---------------------------------------------------------------------------
static void test_registry_directories() {
    std::printf("\n-- registry_directories --\n");
    ok(utils::registry_directories("", "U") == utils::user_directories("U"),
       "no OFLM_MODEL_PATH: the user directories");
    const auto e = utils::registry_directories("M", "U");
    ok(!e.empty() && e[0] == "M" && e.size() == utils::user_directories("U").size() + 1,
       "OFLM_MODEL_PATH (where `oflm add` writes its registry) first, then the user directories");
}

// ---------------------------------------------------------------------------
// USERDIR-REGISTRY-LAYERS with an explicit OFLM_CONFIG_PATH, through the real function.
// ---------------------------------------------------------------------------
static void test_find_model_lists_explicit(const fs::path& tmp) {
    std::printf("\n-- find_model_lists with OFLM_CONFIG_PATH --\n");
    const fs::path custom = tmp / "explicit" / "model_list.json";
    write_json(custom, builtin_registry());
    set_env("FLM_CONFIG_PATH", "");
    set_env("OFLM_CONFIG_PATH", custom.string());
    const auto lists = utils::find_model_lists();
    ok(lists.size() == 1 && same(lists[0], custom.string()),
       "an explicit custom OFLM_CONFIG_PATH is the whole registry");
    set_env("OFLM_CONFIG_PATH", "");
}

// ---------------------------------------------------------------------------
// USERDIR-MODEL-INFO: the shipped model_info.json, then the user ones merged by tag.
// ---------------------------------------------------------------------------
static void test_model_info(const fs::path& tmp) {
    std::printf("\n-- model_info.json --\n");
    const auto c = utils::model_info_candidates("E", "P");
    eqi((long long)c.size(), 3, "three installed locations");
    if (c.size() == 3) {
        eq(c[0], (fs::path("E") / "model_info.json").string(), "next to the executable");
        eq(c[1], (fs::path("E") / ".." / "share" / "oflm" / "model_info.json").string(),
           "the relocatable bundle (the Windows branch used to stop before this)");
        eq(c[2], (fs::path("P") / "share" / "oflm" / "model_info.json").string(), "the configured prefix");
    }

    const json shipped = {{"qwen3:8b", json::array({{{"path", "model.q4nx"}, {"size", 1}}})}};
    const json older = {{"mine:1b", json::array({{{"path", "old"}}})},
                        {"qwen3:8b", json::array({{{"path", "model.q4nx"}, {"size", 999}}})}};
    const json newer = {{"mine:1b", json::array({{{"path", "new"}}})},
                        {"mine:2b", json::array({{{"path", "x"}}})}};
    const json m = model_registry::merge_model_info(shipped, {{"older", older}, {"newer", newer}});
    eqi(m["qwen3:8b"][0]["size"].get<long long>(), 1, "a user model_info.json cannot replace a shipped tag's hashes");
    eq(m["mine:1b"][0]["path"].get<std::string>(), "new", "a newer user file wins over an older one");
    ok(m.contains("mine:2b"), "a user tag is added");

    // On files: the layer order, and a missing shipped file.
    const fs::path home = tmp / "info_home";
    const auto dirs = utils::user_directories(home.string());
    const fs::path base = tmp / "info_share" / "model_info.json";
    write_json(base, shipped);
    write_json(fs::path(dirs.back()) / "model_info.json", older);
    write_json(fs::path(dirs.front()) / "model_info.json", newer);
    auto layers = utils::registry_file_layers("model_info.json", base.string(), dirs);
    eqi((long long)layers.size(), 3, "shipped file, then the user files");
    const json loaded = model_registry::load_model_info(layers);
    ok(loaded.contains("qwen3:8b") && loaded.contains("mine:2b") &&
           loaded["mine:1b"][0]["path"] == "new",
       "load_model_info reads and merges them");
    layers = utils::registry_file_layers("model_info.json", "", dirs);
    const json no_base = model_registry::load_model_info(layers);
    ok(no_base.contains("mine:2b") && no_base.contains("qwen3:8b") && no_base["qwen3:8b"][0]["size"] == 999,
       "with no shipped file every user entry loads, even under a tag the shipped file would own");
    std::ofstream(fs::path(dirs.front()) / "model_info.json") << "{ not json";
    const json broken = model_registry::load_model_info(utils::registry_file_layers("model_info.json", base.string(), dirs));
    ok(broken.contains("qwen3:8b") && broken.contains("mine:1b"),
       "a broken user file is skipped; the shipped and other user entries remain");

    // find_model_infos() on this process: a model_info.json in OFLM_MODEL_PATH is the newest layer.
    const fs::path store = tmp / "info_store";
    write_json(store / "model_info.json", newer);
    set_env("FLM_MODEL_PATH", "");
    set_env("FLM_MODELINFO_PATH", "");
    set_env("OFLM_MODEL_PATH", store.string());
    try {
        const auto infos = utils::find_model_infos();
        ok(!infos.empty() && same(infos.back(), (store / "model_info.json").string()),
           "find_model_infos(): the model_info.json `oflm add` writes into OFLM_MODEL_PATH is read");
    } catch (const std::exception& e) {
        ok(false, std::string("find_model_infos() threw: ") + e.what());
    }
    const fs::path only = tmp / "info_explicit" / "model_info.json";
    write_json(only, shipped);
    set_env("OFLM_MODELINFO_PATH", only.string());
    const auto explicit_infos = utils::find_model_infos();
    ok(explicit_infos.size() == 1 && same(explicit_infos[0], only.string()),
       "an explicit OFLM_MODELINFO_PATH is the whole file");
    set_env("OFLM_MODELINFO_PATH", "");
    set_env("OFLM_MODEL_PATH", "");
}

int main() {
    const fs::path tmp = fs::temp_directory_path() / ("oflm_user_dirs_test_" + std::to_string(std::rand()));
    std::error_code ec;
    fs::remove_all(tmp, ec);
    fs::create_directories(tmp);

    test_user_directories();
    test_registry_directories();
    test_model_list_layers(tmp);
    test_find_model_lists_explicit(tmp);
    test_merge();
    test_model_list(tmp);
    test_models_search_roots();
    test_model_info(tmp);

    fs::remove_all(tmp, ec);
    std::printf("\n%s (%d checks, %d failures)\n", failures ? "FAILED" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
