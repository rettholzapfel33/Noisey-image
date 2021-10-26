#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }


    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
        catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";

    std::vector<torch::jit::IValue> inputs;
    at::Tensor input_tensor = torch::ones({1,3,416,416});
    input_tensor = input_tensor.cuda();
    inputs.push_back(input_tensor);

    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(1,0,5) << '\n';
}