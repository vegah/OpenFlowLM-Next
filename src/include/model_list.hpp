/// \file model_list.hpp
/// \brief model_list class
/// \author OpenFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class is used to manage the model list.
#pragma once
#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <any>
#include <unordered_set>
#include <filesystem>
#include <map>
#include "utils/utils.hpp"
#include "model_registry.hpp"

/// \note This class is used to manage the model list.
class model_list {
    public:
        std::unordered_set<std::string> all_tags;
        /// \brief constructor
        model_list(){}


        /// \brief constructor: one registry, one models directory
        /// \param list_path the path to the model list
        /// \param exe_dir the directory `model_path` is resolved against
        model_list(std::string& list_path, std::string& exe_dir)
            : model_list(std::vector<std::string>{list_path}, exe_dir, std::vector<std::string>{exe_dir}) {}

        /// \brief constructor: a built-in registry merged with the user ones (#30)
        /// \param list_paths the built-in registry, then user registries oldest first
        ///        (utils::find_model_lists()); the merge rules are in model_registry.hpp
        /// \param default_root where a model installed nowhere yet belongs
        ///        (utils::get_models_directory())
        /// \param search_roots every directory an installed model may already be in, in
        ///        order (utils::models_directories()). Searched per model, so a model in
        ///        the pre-rename directory stays found once the new one exists.
        model_list(const std::vector<std::string>& list_paths, const std::string& default_root,
                   const std::vector<std::string>& search_roots) {
            if (list_paths.empty()) {
                std::cerr << "No model list to read" << std::endl;
                exit(1);
            }
            this->list_path = list_paths.front();
            std::ifstream config_file(this->list_path);
            if (!config_file.is_open()) {
                std::cerr << "Failed to open config file: " << this->list_path << std::endl;
                exit(1);
            }
            this->config = nlohmann::json::parse(config_file);
            config_file.close();
            this->builtin_default_sizes = model_registry::default_sizes(this->config);

            std::vector<model_registry::Layer> layers;
            model_registry::Report report;
            for (size_t i = 1; i < list_paths.size(); ++i) {
                try {
                    std::ifstream f(list_paths[i]);
                    if (!f.is_open()) throw std::runtime_error("cannot be opened");
                    layers.push_back({list_paths[i], nlohmann::json::parse(f)});
                } catch (const std::exception& e) {
                    // A broken user file must not take the built-in models down with it,
                    // and must not disappear in silence either.
                    report.notes.push_back(list_paths[i] + ": could not be read (" + e.what() + "); ignored");
                }
            }
            if (!layers.empty()) this->config = model_registry::merge(std::move(this->config), layers, &report);
            for (const auto& src : report.sources) {
                const size_t n = src.tags.size();
                std::cerr << "[OFLM]  " << n << " user model" << (n == 1 ? "" : "s") << " from " << src.path
                          << " (" << model_registry::detail::sample(src.tags) << ")" << std::endl;
            }
            for (const std::string& n : report.notes) std::cerr << "[OFLM]  " << n << std::endl;

            // Resolve model_root_path relative to the default models directory
            std::string relative_model_path = this->config["model_path"];
            this->model_relative_path = relative_model_path;
            std::filesystem::path root_path = std::filesystem::path(default_root) / relative_model_path;
            this->model_root_path = root_path.string();
            this->model_search_roots = search_roots;

            // Populate all_tags set
            for (const auto& [model_type, sizes] : this->config["models"].items()) {
                // insert default tag without size
                all_tags.insert(model_type);
                for (const auto& [size, model_info] : sizes.items()) {
                    all_tags.insert(model_type + ":" + size);
                }
            }
        }

        /// \brief get the model info
        /// \param tag the tag of the model
        /// \return the model info
        std::pair<std::string, nlohmann::json> get_model_info(const std::string tag) const {
            static std::string last_error_tag = "";
            std::string new_tag = rectify_model_tag(tag);
            bool model_found = false;
            // get model type, the string before ':' in the tag
            std::string model_type;
            std::string model_size;

            if (new_tag.find(':') != std::string::npos) {
                model_type = new_tag.substr(0, new_tag.find(':'));
                model_size = new_tag.substr(new_tag.find(':') + 1);
            }
            else {
                model_type = new_tag;
                model_size = "";
            }
            
            // find the model subset first, compare with the key of the model
            bool model_subset_found = false;
            for (const auto& [key, model] : this->config["models"].items()) {
                if (key == model_type) {
                    model_subset_found = true;
                    break;
                }
            } 
            if (model_subset_found) {
                bool model_found = false;
                for (const auto& [key, model] : this->config["models"][model_type].items()) {
                    if (key == model_size) { // if the size is found, return the model
                        model_found = true;
                        return std::make_pair(new_tag, model);
                    }
                } 
                if (!model_found) {
                    if (last_error_tag != new_tag) {
                        last_error_tag = new_tag;
                        header_print_r("ERROR", "Model not found: " + model_size + " in subset " + model_type);
                        header_print_r("ERROR", "Using default model: llama3.2-1B");
                    }
                    return std::make_pair("llama3.2:1b", this->config["models"]["llama3.2"]["1b"]);
                }
            }
            else{
                if (last_error_tag != new_tag) {
                    last_error_tag = new_tag;
                    header_print_r("ERROR", "Model subset not found: " << model_type << "; using default model: llama3.2-1B");
                }
                return std::make_pair("llama3.2:1b", this->config["models"]["llama3.2"]["1b"]);
            }
            return std::make_pair("llama3.2:1b", this->config["models"]["llama3.2"]["1b"]);
        }

        /// \brief cut the tag, some program adds a prefix to the tag, like "Ollama/llama3.2-1B", we need to cut the prefix
        /// \param tag the tag of the model
        /// \return the model type, string
        std::string cut_tag(const std::string tag) const {
            std::string new_tag = tag;
            if (tag.find('/') != std::string::npos) {
                new_tag = tag.substr(tag.find('/') + 1);
            }
            return new_tag;
        }

        /// \brief rectify the model tag, remove / and replace with actuall tag if size is not specified
        /// \param original_tag the original tag of the model
        /// \return the rectified model tag, string
        std::string rectify_model_tag(const std::string original_tag) const {
            std::string new_tag = this->cut_tag(original_tag);
            // check if size is specified
            if (new_tag.find(':') == std::string::npos) {
                // get the first size in the subset
                std::string model_type = new_tag;
                // A bare built-in tag keeps meaning what the built-in registry says, however a
                // user entry under the same type happens to sort (#30).
                auto builtin = this->builtin_default_sizes.find(model_type);
                std::string model_size = builtin != this->builtin_default_sizes.end()
                    ? builtin->second
                    : this->config["models"][model_type].begin().key();
                new_tag = model_type + ":" + model_size;
            }
            return new_tag;
        }

        /// \brief get the model root path
        /// \return the model root path, string
        std::string get_model_root_path(){
            return this->model_root_path;
        }

        /// \brief get all the models
        /// \return all the models in json
        nlohmann::json get_all_models(){
            nlohmann::json response = {
                {"models", nlohmann::json::array()}
            };

            for (const auto& [model_type, model_subset] : this->config["models"].items()) {
                for (const auto& [size, model_info] : model_subset.items()) {
                    nlohmann::json model_entry = model_info;
                    model_entry["name"] = model_type + ":" + size;
                    model_entry["model"] = model_type + ":" + size;
                    response["models"].push_back(model_entry);
                }
            }
            return response;
        }

        /// \brief get all the models
        /// \return all the models in json
        nlohmann::json get_all_models_ollama() {
            nlohmann::json response = {
                {"models", nlohmann::json::array()}
            };

            for (const auto& [model_type, model_subset] : this->config["models"].items()) {
                if (model_type == "whisper-v3") continue;
                else if (model_type == "embed-gemma") continue;
                for (const auto& [size, model_info] : model_subset.items()) {
                    nlohmann::json model_entry = {
                        {"name", model_type + ":" + size},
                        {"model", model_type + ":" + size},
                        {"details", {
                            {"family", model_info["details"]["family"]},
                            {"parameter_size", model_info["details"]["parameter_size"]},
                            {"quantization_level", model_info["details"]["quantization_level"]}
                        }}
                    };
                    response["models"].push_back(model_entry);
                }
            }
            return response;
        }

        /// \brief get all the models
        /// \return all the models in json
        nlohmann::json get_all_models_openai() {
            nlohmann::json response = {
                {"object", "list"},
                {"data", nlohmann::json::array()},
                {"object", "list" }
            };

            std::time_t now = std::time(nullptr);

            for (const auto& [model_type, model_subset] : this->config["models"].items()) {
                if (model_type == "whisper-v3") continue;
                else if (model_type == "embed-gemma") continue;
                for (const auto& [size, model_info] : model_subset.items()) {
                    // id uses the same "type:size" convention; created uses current epoch seconds
                    nlohmann::json model_entry = {
                        {"id", model_type + ":" + size},
                        {"object", "model"},
                        {"created", static_cast<long long>(now)},
                        {"owned_by", "OpenFlowLM"}
                    };
                    response["data"].push_back(model_entry);
                }
            }

            return response;
        }

        /// \brief get the model path
        /// \param tag the tag of the model
        /// \return the model path, string
        std::string get_model_path(const std::string& tag){
            std::string new_tag = this->rectify_model_tag(tag);
            auto [new_tag_unused, model_info] = this->get_model_info(new_tag);
            std::string model_name = model_info["name"];
            std::filesystem::path full_path = std::filesystem::path(this->model_root_path) / model_name;
            // Per model, not per directory (#30): the first search root that holds a COMPLETE
            // copy, so one model pulled into .oflm does not hide every model still in .flm.
            // Complete, not merely present: this path is also where `oflm pull` writes and what
            // `oflm remove` deletes, so a partial or abandoned copy in another store must never
            // become the target -- anything short of complete belongs to the default directory.
            for (const std::string& root : this->model_search_roots) {
                const std::filesystem::path cand =
                    std::filesystem::path(root) / this->model_relative_path / model_name;
                if (is_complete_copy(cand, model_info)) return cand.string();
            }
            return full_path.string();
        }

        /// \brief true when `dir` holds every file the entry lists (#30)
        /// \note An entry without a `files` list counts as complete when the directory is not
        ///       empty. Sizes and hashes are the downloader's business, not this lookup's.
        static bool is_complete_copy(const std::filesystem::path& dir, const nlohmann::json& model_info) {
            std::error_code ec;
            if (!std::filesystem::is_directory(dir, ec)) return false;
            if (model_info.contains("files") && model_info["files"].is_array() && !model_info["files"].empty()) {
                for (const auto& f : model_info["files"]) {
                    if (!f.is_string() || !std::filesystem::is_regular_file(dir / f.get<std::string>(), ec)) return false;
                }
                return true;
            }
            return std::filesystem::directory_iterator(dir, ec) != std::filesystem::directory_iterator();
        }

        bool is_model_supported(const std::string& tag) {
            return all_tags.find(tag) != all_tags.end();
        }
        
    private:
        std::string list_path;
        nlohmann::json config;
        std::string model_root_path;
        std::string model_relative_path;
        std::vector<std::string> model_search_roots;
        std::map<std::string, std::string> builtin_default_sizes;

};
