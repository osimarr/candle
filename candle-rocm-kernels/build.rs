use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(hip_runtime)");
    println!("cargo:rerun-if-env-changed=CANDLE_ROCM_HIP");
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-changed=build.rs");
    for source in HIP_SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }
    println!("cargo:rerun-if-changed=src/kernels/common.h");

    let hip_mode = env::var("CANDLE_ROCM_HIP").unwrap_or_else(|_| "auto".to_string());
    if matches!(
        hip_mode.as_str(),
        "0" | "false" | "False" | "FALSE" | "off" | "OFF"
    ) {
        println!("cargo:warning=building candle-rocm-kernels without HIP runtime support");
        return;
    }

    let required = matches!(
        hip_mode.as_str(),
        "1" | "true" | "True" | "TRUE" | "on" | "ON"
    );
    let Some(hipcc) = find_hipcc() else {
        if required {
            panic!("CANDLE_ROCM_HIP requested HIP support, but hipcc was not found");
        }
        println!("cargo:warning=hipcc not found; building candle-rocm-kernels host fallback only");
        return;
    };

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let lib = out_dir.join("libhip_kernels.a");
    let objects = compile_hip_sources(&hipcc, &out_dir);

    let ar = env::var_os("AR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ar"));
    let mut ar_cmd = Command::new(&ar);
    ar_cmd.arg("crus").arg(&lib);
    for obj in &objects {
        ar_cmd.arg(obj);
    }
    let output = ar_cmd
        .output()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", ar.display()));
    if !output.status.success() {
        panic!(
            "ar failed while creating {}:\nstdout:\n{}\nstderr:\n{}",
            lib.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let rocm_path = env::var_os("ROCM_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm"));
    println!("cargo:rustc-cfg=hip_runtime");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=hip_kernels");
    println!(
        "cargo:rustc-link-search=native={}",
        rocm_path.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

const HIP_SOURCES: &[&str] = &[
    "src/kernels/runtime.cpp",
    "src/kernels/unary.cpp",
    "src/kernels/binary.cpp",
    "src/kernels/cmp.cpp",
    "src/kernels/scalar.cpp",
    "src/kernels/cast.cpp",
    "src/kernels/reduce.cpp",
    "src/kernels/random.cpp",
    "src/kernels/conv.cpp",
    "src/kernels/matmul.cpp",
    "src/kernels/indexing.cpp",
    "src/kernels/ternary.cpp",
    "src/kernels/sort.cpp",
    "src/kernels/nn.cpp",
    "src/kernels/image.cpp",
    "src/kernels/fill.cpp",
    "src/kernels/copy.cpp",
];

fn compile_hip_sources(hipcc: &Path, out_dir: &Path) -> Vec<PathBuf> {
    HIP_SOURCES
        .iter()
        .map(|source| {
            let source = Path::new(source);
            let stem = source
                .file_stem()
                .and_then(|stem| stem.to_str())
                .expect("HIP source filenames have valid UTF-8 stems");
            let obj = out_dir.join(format!("hip_{stem}.o"));
            let output = Command::new(hipcc)
                .arg("-O2")
                .arg("-fPIC")
                .arg("-std=c++17")
                .arg("-c")
                .arg(source)
                .arg("-o")
                .arg(&obj)
                .output()
                .unwrap_or_else(|err| panic!("failed to run {}: {err}", hipcc.display()));
            if !output.status.success() {
                panic!(
                    "hipcc failed while compiling {}:\nstdout:\n{}\nstderr:\n{}",
                    source.display(),
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            obj
        })
        .collect()
}

fn find_hipcc() -> Option<PathBuf> {
    if let Some(path) = env::var_os("HIPCC").map(PathBuf::from) {
        if path.exists() {
            return Some(path);
        }
    }
    let rocm_path = env::var_os("ROCM_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm"));
    let rocm_hipcc = rocm_path.join("bin/hipcc");
    if rocm_hipcc.exists() {
        return Some(rocm_hipcc);
    }
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|dir| dir.join("hipcc"))
            .find(|candidate| candidate.exists())
    })
}
