// Copyright AGNTCY Contributors (https://github.com/agntcy)
// SPDX-License-Identifier: Apache-2.0

fn main() {
    // Prefer the local a2a-itk clone (created by run_itk.sh);
    // fall back to the path baked into the ITK Docker image at build time.
    let proto_dir = if std::path::Path::new("a2a-itk/protos").exists() {
        "a2a-itk/protos"
    } else {
        "/tmp/protos"
    };
    let proto_file = format!("{proto_dir}/instruction.proto");
    prost_build::compile_protos(&[proto_file.as_str()], &[proto_dir])
        .expect("failed to compile instruction.proto");
    println!("cargo:rerun-if-changed={proto_dir}/instruction.proto");
}
