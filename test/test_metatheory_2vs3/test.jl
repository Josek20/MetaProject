using Pkg
using BenchmarkTools


function benchmark_version(package_path, function_name, args...)
    # Pkg.activate(package_path)
    # using Metatheory

    # return @btime $(function_name)($(args...))
end

# version1_result = benchmark_version("path_to_env_version1", YourPackage.some_function, arg1, arg2)
# version2_result = benchmark_version("path_to_env_version2", YourPackage.some_function, arg1, arg2)

# println("Version 1 Benchmark Result: ", version1_result)
# println("Version 2 Benchmark Result: ", version2_result)

Pkg.activate("test/src/test_metatheory_2vs3")
using Metatheory

