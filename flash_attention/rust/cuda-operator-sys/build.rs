use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command, description: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to {description}: {error}"));
    assert!(status.success(), "failed to {description}: {status}");
}

fn cuda_root(nvcc: &Path) -> PathBuf {
    if let Some(root) = env::var_os("CUDA_HOME").or_else(|| env::var_os("CUDA_PATH")) {
        return PathBuf::from(root);
    }

    let resolved = nvcc.canonicalize().unwrap_or_else(|_| nvcc.to_path_buf());
    resolved
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"))
}

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let flash_attention_dir = manifest_dir.parent().unwrap().parent().unwrap();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let nvcc = PathBuf::from(env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()));
    let architecture = env::var("CUDA_ARCH").unwrap_or_else(|_| "86".to_owned());

    assert!(
        !architecture.is_empty() && architecture.bytes().all(|byte| byte.is_ascii_digit()),
        "CUDA_ARCH must contain only digits, for example CUDA_ARCH=86"
    );

    let sources = ["flash_attention_v0.cu", "flash_attention_ffi.cu"];
    let mut objects = Vec::with_capacity(sources.len());

    for source in sources {
        let object = out_dir.join(format!("{source}.o"));
        run(
            Command::new(&nvcc)
                .arg("-std=c++20")
                .arg("-O3")
                .arg("-lineinfo")
                .arg(format!("-arch=sm_{architecture}"))
                .arg("-Xcompiler=-fPIC")
                .arg("-I")
                .arg(flash_attention_dir)
                .arg("-c")
                .arg(flash_attention_dir.join(source))
                .arg("-o")
                .arg(&object),
            &format!("compile {source}"),
        );
        objects.push(object);
    }

    let library = out_dir.join("libcuda_operator_flash_attention.a");
    let mut archive = Command::new(&nvcc);
    archive.arg("--lib").args(&objects).arg("-o").arg(&library);
    run(&mut archive, "archive the CUDA objects");

    let root = cuda_root(&nvcc);
    let target_lib = root.join("targets/x86_64-linux/lib");
    let lib64 = root.join("lib64");
    let cuda_lib = if target_lib.is_dir() {
        target_lib
    } else {
        lib64
    };

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=static=cuda_operator_flash_attention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVCC");
    for source in sources {
        println!(
            "cargo:rerun-if-changed={}",
            flash_attention_dir.join(source).display()
        );
    }
    println!(
        "cargo:rerun-if-changed={}",
        flash_attention_dir
            .join("flash_attention_api.cuh")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        flash_attention_dir.join("flash_attention_ffi.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        flash_attention_dir
            .join("flash_attention_global.cuh")
            .display()
    );
}
