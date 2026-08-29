target "runtime" {
    context = "."
    dockerfile = "docker/Dockerfile"
    target = "runtime"
    tags = ["ghcr.io/badrmoh/mlops-practitioner/prodml:runtime"]
}

target "builder" {
    context = "."
    dockerfile = "docker/Dockerfile"
    target = "builder"
    tags = ["ghcr.io/badrmoh/mlops-practitioner/prodml:builder"]
}
