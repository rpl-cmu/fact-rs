{ lib, rustPlatform }:
let
  cargoTOML = lib.trivial.importTOML ./Cargo.toml;
in
rustPlatform.buildRustPackage {
  pname = cargoTOML.package.name;
  version = cargoTOML.package.version;

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.gitTrackedWith {
      recurseSubmodules = true;
    } ./.;
  };

  cargoLock = {
    lockFile = ./Cargo.lock;
    outputHashes = {
      "sophus_autodiff-0.15.0" = "sha256-Vo+b98A3tx8apkFXtImY1JYwtm1B9kZHxQpg0Qib6LE=";
    };
  };
}
