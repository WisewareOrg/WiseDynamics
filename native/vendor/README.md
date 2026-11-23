# Vendored headers

The C++ headers the compiled integrator builds against, copied verbatim and never edited, so an
install builds offline against exactly the Basilisk build WiseDynamics pins. Nothing here is compiled
on its own.

- `basilisk/` — the Basilisk headers the integrator includes, at the source revision in
  `basilisk/REVISION`, the one the pinned bsk wheel was built from (`Basilisk/_buildInfoData.py`'s
  `sourceRevision`), under their `src/` paths, with Basilisk's ISC `LICENSE`.
- `eigen/` — Eigen's `Eigen/` directory at the version in `eigen/VERSION`, the one Basilisk's
  `conanfile.py` requires, with Eigen's `COPYING.*` files. The dense modules the integrator uses are
  MPL-2.0.

## Checking a tree against upstream

Each tree is checked by diffing it against its upstream, not by reading it. From the repository root:

```sh
rev=$(cat native/vendor/basilisk/REVISION)
git clone -q --filter=blob:none --no-checkout https://github.com/AVSLab/basilisk /tmp/bsk
for f in $(cd native/vendor/basilisk && find src -type f) LICENSE; do
  git -C /tmp/bsk show "$rev:$f" | cmp - "native/vendor/basilisk/$f"
done

curl -sSfL -o /tmp/eigen.tar.bz2 https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
echo "b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626  /tmp/eigen.tar.bz2" | sha256sum -c
tar xjf /tmp/eigen.tar.bz2 -C /tmp
diff -r /tmp/eigen-3.4.0/Eigen native/vendor/eigen/Eigen
```

The checksum is conan-center's for the same archive. A bsk upgrade replaces `basilisk/` with the new
revision's headers, found as the include closure of the integrator (`g++ -MM`), and updates `REVISION`.
