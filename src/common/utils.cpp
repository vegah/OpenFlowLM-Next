/// \file utils.cpp
/// \brief utils class
/// \author OpenFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// 
/// \note This file contains some utility functions for the OpenFlowLM project.
#include "utils/utils.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <cstdlib>
#include <set>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

namespace utils {

std::string getenv_oflm(const char* oflm_name) {
    if (const char* v = std::getenv(oflm_name)) {
        if (*v) return std::string(v);
    }
    // "OFLM_FOO" -> "FLM_FOO". The rename dropped nothing else, so the inverse is
    // to remove the leading 'O'; a name that is not OFLM_-prefixed has no legacy form.
    if (std::strncmp(oflm_name, "OFLM_", 5) != 0) return std::string();
    const std::string legacy = std::string("FLM_") + (oflm_name + 5);
    const char* v = std::getenv(legacy.c_str());
    if (!v || !*v) return std::string();
    static std::set<std::string> warned;          // once per variable, not once per read
    if (warned.insert(legacy).second) {
        std::cerr << "[OFLM]  " << legacy << " is the name used before the oflm rename. "
                  << "It still works; set " << oflm_name << " instead." << std::endl;
    }
    return std::string(v);
}

namespace {

/// The user-level directories for THIS project, newest first (#30). Returned as
/// a list rather than a single path so that a directory made either way is
/// found: #30 asks for %USERPROFILE%\.oflm on Windows, while oflm-add's own
/// `Path.home() / ".config" / <name>` lands in <profile>/.config/oflm. On POSIX
/// `user_dir` already ends in .config, so one entry covers it. Scanning them
/// costs one stat each.
std::vector<std::string> user_oflm_directories(const std::string& user_dir) {
    std::vector<std::string> v;
#ifdef _WIN32
    v.push_back((std::filesystem::path(user_dir) / ".oflm").string());
    v.push_back((std::filesystem::path(user_dir) / ".config" / "oflm").string());
#else
    v.push_back(user_dir + "/oflm");
#endif
    return v;
}

/// The user-level directories the pre-rename releases used, for the same reason
/// user_oflm_directories() exists: a model store is gigabytes and must not have to
/// move for an upgrade. Searched AFTER every oflm location, never before.
std::vector<std::string> legacy_flm_directories(const std::string& user_dir) {
    std::vector<std::string> v;
#ifdef _WIN32
    v.push_back((std::filesystem::path(user_dir) / ".flm").string());
    v.push_back((std::filesystem::path(user_dir) / ".config" / "flm").string());
#else
    v.push_back(user_dir + "/flm");
#endif
    return v;
}

} // namespace

std::vector<std::string> user_directories(const std::string& user_dir) {
    std::vector<std::string> v = user_oflm_directories(user_dir);
    for (const std::string& d : legacy_flm_directories(user_dir)) v.push_back(d);
    return v;
}

std::vector<std::string> user_directories() {
    return user_directories(get_user_directory());
}

std::vector<std::string> registry_directories(const std::string& explicit_model_path, const std::string& user_dir) {
    std::vector<std::string> v;
    // `oflm add` writes model_list.json and model_info.json into the models directory,
    // which is an explicit OFLM_MODEL_PATH when one is set -- a place the user
    // directories below do not cover. It is the newest, so it goes first.
    if (!explicit_model_path.empty()) v.push_back(explicit_model_path);
    for (const std::string& d : user_directories(user_dir)) v.push_back(d);
    return v;
}

std::vector<std::string> registry_directories() {
    return registry_directories(getenv_oflm("OFLM_MODEL_PATH"), get_user_directory());
}

std::string find_model_list() {
    // 1. Check OFLM_CONFIG_PATH environment variable
    const std::string env_path = getenv_oflm("OFLM_CONFIG_PATH");
    if (!env_path.empty()) {
        if (std::filesystem::exists(env_path)) {
            std::cerr << "[OFLM]  Using custom model list path: " << env_path << std::endl;
            return env_path;
        }
    }
    return find_builtin_model_list();
}

std::vector<std::string> registry_file_layers(const std::string& filename, const std::string& builtin_path,
                                              const std::vector<std::string>& dirs) {
    namespace fs = std::filesystem;
    std::vector<std::string> layers{builtin_path};
    // dirs is newest first; a merge lets the LATER layer win, so the newest file has
    // to be applied last.
    for (auto it = dirs.rbegin(); it != dirs.rend(); ++it) {
        const fs::path cand = fs::path(*it) / filename;
        std::error_code ec;
        if (!fs::is_regular_file(cand, ec)) continue;
        bool seen = false;
        for (const std::string& l : layers) {
            std::error_code eq;
            if (!l.empty() && fs::equivalent(l, cand, eq)) { seen = true; break; }
        }
        if (!seen) layers.push_back(cand.string());
    }
    return layers;
}

std::vector<std::string> model_list_layers(const std::string& builtin_path,
                                           const std::vector<std::string>& user_dirs) {
    return registry_file_layers("model_list.json", builtin_path, user_dirs);
}

std::vector<std::string> registry_layers(const std::string& explicit_path, const std::string& builtin_path,
                                         const std::vector<std::string>& user_dirs) {
    std::error_code ec;
    // An explicit registry is the WHOLE registry, exactly as before #30: an install
    // that exported OFLM_CONFIG_PATH at a full copy keeps the behaviour it had. The
    // exception is a variable naming the built-in list itself -- home_install.sh's
    // oflm_env.sh exports exactly that -- which is not a custom registry at all.
    if (!explicit_path.empty() &&
        (builtin_path.empty() || !std::filesystem::equivalent(explicit_path, builtin_path, ec))) {
        return {explicit_path};
    }
    return model_list_layers(builtin_path, user_dirs);
}

std::vector<std::string> find_model_lists() {
    std::string builtin;
    try {
        builtin = find_builtin_model_list();
    } catch (const std::exception&) {
        builtin.clear();
    }
    std::string env_path = getenv_oflm("OFLM_CONFIG_PATH");
    if (!env_path.empty() && !std::filesystem::exists(env_path)) {
        // find_model_list() falls through here in silence; a variable the user set
        // and we did not use is worth one line.
        std::cerr << "[OFLM]  OFLM_CONFIG_PATH=" << env_path << " does not exist; ignoring it." << std::endl;
        env_path.clear();
    }
    if (env_path.empty() && builtin.empty()) {
        throw std::runtime_error("model_list.json not found. Please set OFLM_CONFIG_PATH or place it next to the executable.");
    }
    std::vector<std::string> layers = registry_layers(env_path, builtin, registry_directories());
    if (!env_path.empty()) {
        if (layers.size() == 1 && layers[0] == env_path) {
            std::cerr << "[OFLM]  Using custom model list path: " << env_path << std::endl;
        } else {
            std::cerr << "[OFLM]  OFLM_CONFIG_PATH names the built-in model list; user registries are merged over it."
                      << std::endl;
        }
    }
    return layers;
}

std::vector<std::string> builtin_model_list_candidates(const std::string& exe_dir, const std::string& install_prefix) {
    namespace fs = std::filesystem;
    return {
        (fs::path(exe_dir) / "model_list.json").string(),                               // portable tree
        "model_list.json",                                                               // the CWD
        (fs::path(exe_dir) / ".." / "share" / "oflm" / "model_list.json").string(),     // relocatable bundle
        (fs::path(install_prefix) / "share" / "oflm" / "model_list.json").string(),     // configured prefix
    };
}

namespace {

/// The first candidate that exists and is not <dir>/<filename> for any registry directory.
std::string first_non_user_file(const std::vector<std::string>& candidates, const std::string& filename,
                                const std::vector<std::string>& dirs) {
    namespace fs = std::filesystem;
    for (const std::string& cand : candidates) {
        std::error_code ec;
        if (cand.empty() || !fs::is_regular_file(cand, ec)) continue;
        bool user_file = false;
        for (const std::string& d : dirs) {
            std::error_code eq;
            if (fs::equivalent(cand, fs::path(d) / filename, eq)) { user_file = true; break; }
        }
        if (!user_file) return cand;
    }
    return std::string();
}

} // namespace

std::string first_builtin_model_list(const std::vector<std::string>& candidates,
                                     const std::vector<std::string>& user_dirs) {
    // A user registry is never the base (#30). The CWD candidate makes this reachable:
    // `cd ~/.config/oflm && oflm list` would otherwise read a user file that holds only
    // what was added, and lose every built-in model.
    return first_non_user_file(candidates, "model_list.json", user_dirs);
}

std::string find_builtin_model_list() {
    const std::string found = first_builtin_model_list(
        builtin_model_list_candidates(get_executable_directory(), CMAKE_INSTALL_PREFIX), registry_directories());
    if (found.empty()) {
        throw std::runtime_error("model_list.json not found. Please set OFLM_CONFIG_PATH or place it next to the executable.");
    }
    return found;
}

std::string find_model_info() {
    std::string install_prefix = CMAKE_INSTALL_PREFIX;

    // 1. Check OFLM_MODELINFO_PATH environment variable
    const std::string env_path = getenv_oflm("OFLM_MODELINFO_PATH");
    if (!env_path.empty()) {
        if (std::filesystem::exists(env_path)) {
            std::cerr << "[OFLM]  Using custom model info path: " << env_path << std::endl;
            return env_path;
        }
    }

    // 2. Stay next to an explicitly configured model_list.json. A relocated
    // install points OFLM_CONFIG_PATH at its own share/oflm; without this the
    // lookup falls through to the baked-in prefix below and we end up sizing
    // and hash-checking downloads against a different (stale) revision.
    const std::string config_path = getenv_oflm("OFLM_CONFIG_PATH");
    if (!config_path.empty()) {
        std::filesystem::path sibling =
            std::filesystem::path(config_path).parent_path() / "model_info.json";
        if (std::filesystem::exists(sibling)) {
            return sibling.string();
        }
    }

    // 3. The installed locations -- the same on every platform (#30: the Windows
    // branch used to stop at the executable's directory).
    for (const std::string& cand : model_info_candidates(get_executable_directory(), install_prefix)) {
        if (std::filesystem::exists(cand)) {
            return cand;
        }
    }

    // If not found, throw an error
    throw std::runtime_error("model_info.json not found. Please set OFLM_MODELINFO_PATH or place it next to the executable.");
}

std::vector<std::string> model_info_candidates(const std::string& exe_dir, const std::string& install_prefix) {
    namespace fs = std::filesystem;
    return {
        (fs::path(exe_dir) / "model_info.json").string(),                               // portable tree
        (fs::path(exe_dir) / ".." / "share" / "oflm" / "model_info.json").string(),     // relocatable bundle
        (fs::path(install_prefix) / "share" / "oflm" / "model_info.json").string(),     // configured prefix
    };
}

std::vector<std::string> find_model_infos() {
    namespace fs = std::filesystem;
    // An explicit OFLM_MODELINFO_PATH is the whole file, as it is for find_model_info().
    const std::string env_path = getenv_oflm("OFLM_MODELINFO_PATH");
    if (!env_path.empty() && fs::exists(env_path)) return {env_path};

    // The shipped file: beside OFLM_CONFIG_PATH (a relocated install points that at its
    // own share/oflm), then the installed locations -- but never a user registry's
    // model_info.json, which is a layer below and holds only what was added.
    std::vector<std::string> candidates;
    const std::string config_path = getenv_oflm("OFLM_CONFIG_PATH");
    if (!config_path.empty()) {
        candidates.push_back((fs::path(config_path).parent_path() / "model_info.json").string());
    }
    for (const std::string& c : model_info_candidates(get_executable_directory(), CMAKE_INSTALL_PREFIX)) {
        candidates.push_back(c);
    }
    const std::vector<std::string> dirs = registry_directories();
    const std::string builtin = first_non_user_file(candidates, "model_info.json", dirs);

    // `oflm add` writes a model_info.json beside the user registry it writes; without
    // this, a model it added is found in the merged model list and then missing from
    // the file verification that reads model_info.json by tag.
    std::vector<std::string> layers = registry_file_layers("model_info.json", builtin, dirs);
    if (builtin.empty() && layers.size() == 1) {
        throw std::runtime_error("model_info.json not found. Please set OFLM_MODELINFO_PATH or place it next to the executable.");
    }
    return layers;
}

namespace {

/// A configured root may be given with or without its trailing "xclbins"
/// component; the callers of `find_xclbin_path` always append it themselves.
std::string strip_xclbins(std::string path) {
    std::filesystem::path p(path);
    if (p.filename().empty()) p = p.parent_path();   // a trailing separator
    if (p.filename() == "xclbins") return p.parent_path().string();
    return path;
}

/// The roots `find_xclbin_path` has always walked, in its order. Kept separate so that
/// widening the OPEN path's search (xclbin_roots below) cannot move which root the CLOSED
/// path picks: it returns exactly one, and every closed kernel is loaded relative to it.
std::vector<std::string> closed_path_roots() {
    std::vector<std::string> c;
    const std::string env_path = getenv_oflm("OFLM_XCLBIN_PATH");
    if (!env_path.empty()) c.push_back(strip_xclbins(env_path));
    std::string exe_dir = get_executable_directory();
    c.push_back(exe_dir);                       // portable development tree
    c.push_back(".");                           // then the CWD
    c.push_back(exe_dir + "/../share/oflm");     // relocatable installed bundle
    c.push_back(CMAKE_XCLBIN_PREFIX);           // legacy configured prefix
    return c;
}

} // namespace

std::vector<std::string> xclbin_roots() {
    std::vector<std::string> candidates;

    // The user-level roots first: oflm-add installs a model's kernels under one of these,
    // and the shipped sets live in the install tree below. A lookup that stops at the
    // first root (find_xclbin_path) can only ever see one of the two.
    const std::string env_path = getenv_oflm("OFLM_XCLBIN_PATH");
    if (!env_path.empty()) candidates.push_back(strip_xclbins(env_path));
    // Beside an explicitly configured model_list.json, the way find_model_info stays
    // beside it: a user registry and its kernels live in one directory.
    const std::string config_path = getenv_oflm("OFLM_CONFIG_PATH");
    if (!config_path.empty()) {
        candidates.push_back(std::filesystem::path(config_path).parent_path().string());
    }
    // The directories oflm-add uses when neither variable is exported, new
    // before legacy (#30) -- an existing install keeps working with no
    // migration and no copying of multi-gigabyte weights.
    //
    // The second line used to be user_flm_directory(), i.e. the LEGACY root, and the
    // oflm rename turned it into user_oflm_directory() -- a strict subset of the list
    // above it. The line became dead and the legacy root stopped being searched, while
    // the comment went on promising it. legacy_flm_directories() restores what the
    // comment says (#41).
    for (const std::string& d : user_directories()) candidates.push_back(d);

    for (const std::string& c : closed_path_roots()) candidates.push_back(c);

    std::vector<std::string> roots;
    for (const std::string& c : candidates) {
        if (c.empty()) continue;
        std::error_code ec;
        if (!std::filesystem::exists(c + "/xclbins", ec)) continue;
        if (std::find(roots.begin(), roots.end(), c) == roots.end()) roots.push_back(c);
    }
    return roots;
}

std::string find_xclbin_path() {
    for (const std::string& c : closed_path_roots()) {
        if (!c.empty() && std::filesystem::exists(c + "/xclbins")) return c;
    }
    throw std::runtime_error("xclbins not found. Please set OFLM_XCLBIN_PATH or place it next to the executable.");
}

std::string get_executable_directory() {
#ifdef _WIN32
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::string exe_path(buffer);
    size_t last_slash = exe_path.find_last_of("\\");
    if (last_slash != std::string::npos) {
        return exe_path.substr(0, last_slash);
    }
    return ".";
#else
    char buffer[PATH_MAX] = {0};
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len > 0) {
        buffer[len] = '\0';
        std::string exe_path(buffer);
        size_t last_slash = exe_path.find_last_of("/");
        if (last_slash != std::string::npos) {
            return exe_path.substr(0, last_slash);
        }
    }
    return ".";
#endif
}

std::string get_user_directory() {
#ifdef _WIN32
    char buffer[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_PROFILE, NULL, 0, buffer))) {
        return std::string(buffer);
    }
    // Fallback to current directory if user folder cannot be found
    return ".";
#else
    const char* home = std::getenv("HOME");
    if (home && *home) {
        return std::string(home) + "/.config";
    }
    return ".";
#endif
}

///@brief get_server_port gets the server port from environment variable OFLM_SERVE_PORT
///@return the server port, default is 52625 if environment variable is not set
int get_server_port(int user_port) {
    if (user_port > 0 && user_port <= 65535) {
        return user_port;
    }
    else {
        // OFLM_SERVE_PORT, or the FLM_SERVE_PORT the pre-rename installer wrote (#41).
        const std::string port_env = getenv_oflm("OFLM_SERVE_PORT");
        if (!port_env.empty()) {
            try {
                int port = std::stoi(port_env);
                if (port > 0 && port <= 65535) {
                    return port;
                }
            }
            catch (const std::exception&) {
                // Invalid port number, use default
            }
        }
    }

    return 52625; // Default port
}

///@brief get_models_directory gets the models directory from environment variable or defaults to user/.oflm/models on Windows or ~/.config/oflm on Linux
///@return the models directory path
std::string get_models_directory() {
    // OFLM_MODEL_PATH, or the FLM_MODEL_PATH the pre-rename installer wrote (#41).
    const std::string custom_path = getenv_oflm("OFLM_MODEL_PATH");
    if (!custom_path.empty()) return custom_path;

    // No variable set: the oflm directory, then the pre-rename one if it exists and the
    // oflm one does not. A model store is gigabytes; an upgrade must not orphan it.
    std::string user_dir = get_user_directory();
#ifdef _WIN32
    const std::string oflm_dir = user_dir + "\\.oflm";
#else
    const std::string oflm_dir = user_dir + "/oflm";
#endif
    std::error_code ec;
    if (std::filesystem::is_directory(oflm_dir, ec)) return oflm_dir;
    for (const std::string& d : legacy_flm_directories(user_dir)) {
        if (std::filesystem::is_directory(d, ec)) {
            std::cerr << "[OFLM]  using the pre-rename model directory " << d
                      << "; move it to " << oflm_dir << " when convenient." << std::endl;
            return d;
        }
    }
    return oflm_dir;
}

std::vector<std::string> models_search_roots(const std::string& explicit_path, const std::string& user_dir) {
    // An explicit OFLM_MODEL_PATH is the one store, as it always was.
    if (!explicit_path.empty()) return {explicit_path};
    return user_directories(user_dir);
}

std::vector<std::string> models_directories() {
    return models_search_roots(getenv_oflm("OFLM_MODEL_PATH"), get_user_directory());
}

} // end of namespace utils
