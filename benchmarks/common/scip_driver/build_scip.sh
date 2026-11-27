#!/usr/bin/env bash
# Builds the pinned SCIP for cbls_scip_traj (#442): scipopt/scip at SCIP_COMMIT
# beside this file, SoPlex, Ipopt over MUMPS with METIS 5.2.1, TPI=tny.
#
#   benchmarks/common/scip_driver/build_scip.sh PREFIX    # sources in PREFIX/src
#
# Follows scipoptsuite-deploy main e4f0441 (.github/workflows/scripts/linux.bash)
# except: shared libraries, the system's LAPACK/BLAS, no GCG, no Boost. Run it
# niced and pinned to the cores the machine's rules allow; JOBS caps make.
set -euo pipefail

[ $# -eq 1 ] || { echo "usage: $0 PREFIX" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
SCIP_COMMIT=$(head -n1 "$HERE/SCIP_COMMIT")
SOPLEX_VERSION=8.1.0
IPOPT_VERSION=3.14.19
METIS_VERSION=5.2.1
GKLIB_COMMIT=8bd6bad750b2b0d90800c632cf18e8ee93ad72d7    # deploy main's pin; GKlib has no 5.2.1 tag
MUMPS_TP_COMMIT=35e03ac76c20c2a8b70c0e6ad3604455590518d6 # coin-or-tools/ThirdParty-Mumps HEAD, 2026-11-27
MUMPS_VERSION=5.9.2                                      # that commit's get.Mumps
JOBS=${JOBS:-4}

mkdir -p "$1"
PREFIX=$(cd "$1" && pwd)
SRC=$PREFIX/src
mkdir -p "$SRC" "$PREFIX/share"
cd "$SRC"

# Autoconf (MUMPS, Ipopt) reads $prefix/share/config.site: METIS 5.2.1 and LAPACK.
cat > "$PREFIX/share/config.site" <<EOF
enable_java=no
enable_sipopt=no
with_pic=yes
with_metis_cflags="-I$PREFIX/include"
with_metis_lflags="-L$PREFIX/lib -lmetis -lGKlib -lm"
with_lapack_lflags="-llapack -lblas -lgfortran -lm"
EOF

# fetch URL FILE SHA256 DIR: download and unpack once; refuse an archive whose
# hash is not the one the verified build fetched (2026-11-27). Renamed only once
# complete, so a failed download is fetched again.
fetch() {
  [ -f "$2" ] || { wget -q "$1" -O "$2.part" && mv "$2.part" "$2"; }
  echo "$3  $2" | sha256sum -c --quiet - || { echo "$2: sha256 mismatch" >&2; exit 1; }
  [ -d "$4" ] || tar -xf "$2"
}

if [ ! -f "$PREFIX/lib/libmetis.a" ]; then
  fetch https://github.com/KarypisLab/GKlib/archive/$GKLIB_COMMIT.tar.gz $GKLIB_COMMIT.tar.gz \
    e1d59de12336731e6dde8465f05de7b907fae5d72c921cffcde217a89eaab654 GKlib-$GKLIB_COMMIT
  fetch https://github.com/KarypisLab/METIS/archive/refs/tags/v$METIS_VERSION.tar.gz v$METIS_VERSION.tar.gz \
    1a4665b2cd07edc2f734e30d7460afb19c1217c2547c2ac7bf6e1848d50aff7a METIS-$METIS_VERSION
  (cd GKlib-$GKLIB_COMMIT
   sed -i 's/^CONFIG_FLAGS =/CONFIG_FLAGS = -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_POSITION_INDEPENDENT_CODE=ON/' Makefile
   sed -i 's/-march=native//' GKlibSystem.cmake
   make config prefix="$PREFIX" && make -j"$JOBS" && make install)
  (cd METIS-$METIS_VERSION
   sed -i 's/^CONFIG_FLAGS =/CONFIG_FLAGS = -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_POSITION_INDEPENDENT_CODE=ON/' Makefile
   sed -i 's/-march=native//' conf/gkbuild.cmake
   make config prefix="$PREFIX" gklib_path="$PREFIX" && make -j"$JOBS" && make install)
fi

if [ ! -f "$PREFIX/lib/pkgconfig/coinmumps.pc" ]; then
  [ -d ThirdParty-Mumps ] || git clone -q https://github.com/coin-or-tools/ThirdParty-Mumps.git
  (cd ThirdParty-Mumps && git checkout -q $MUMPS_TP_COMMIT
   if [ ! -d MUMPS ]; then  # get.Mumps's steps from a checksummed archive; it checks none
     grep -qx "mumps_ver=$MUMPS_VERSION" get.Mumps || { echo "get.Mumps is not MUMPS $MUMPS_VERSION" >&2; exit 1; }
     fetch https://coin-or-tools.github.io/ThirdParty-Mumps/MUMPS_$MUMPS_VERSION.tar.gz MUMPS_$MUMPS_VERSION.tar.gz \
       652d28a91c7ff886b4a5ea122df4e5d014bdb2eba0545e581be4d8fa50c0ff19 MUMPS_$MUMPS_VERSION
     mv MUMPS_$MUMPS_VERSION MUMPS
     patch -p0 < mumps_mpi.patch
     mv MUMPS/libseq/mpi.h MUMPS/libseq/mumps_mpi.h
     patch -p0 < mumps_call.patch
   fi
   ./configure --prefix="$PREFIX"
   make -j1 && make install)   # serial: MUMPS's Fortran module dependencies race under parallel make
fi

if [ ! -f "$PREFIX/lib/pkgconfig/ipopt.pc" ]; then
  fetch https://github.com/coin-or/Ipopt/archive/refs/tags/releases/$IPOPT_VERSION.tar.gz ipopt-$IPOPT_VERSION.tar.gz \
    b3eb84a23812b53a3325bcd2c599de2b0f5df45a18ed251f9e3c1cd893136287 Ipopt-releases-$IPOPT_VERSION
  mkdir -p Ipopt-releases-$IPOPT_VERSION/build
  (cd Ipopt-releases-$IPOPT_VERSION/build && ../configure --prefix="$PREFIX" && make -j"$JOBS" && make install)
fi

if [ ! -d "$PREFIX/lib/cmake/soplex" ]; then
  fetch https://github.com/scipopt/soplex/archive/refs/tags/v$SOPLEX_VERSION.tar.gz soplex-$SOPLEX_VERSION.tar.gz \
    5aad34f158d251549e2aadbfbef706629f38652fcd027c12b0a8b2d3c8ff0719 soplex-$SOPLEX_VERSION
  cmake -S soplex-$SOPLEX_VERSION -B soplex-$SOPLEX_VERSION/build -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release -DGMP=false -DPAPILO=false -DBOOST=false -DMPFR=false
  cmake --build soplex-$SOPLEX_VERSION/build -j"$JOBS"
  cmake --install soplex-$SOPLEX_VERSION/build
fi

[ -d scip ] || git clone -q --filter=blob:none https://github.com/scipopt/scip.git
git -C scip fetch -q origin "$SCIP_COMMIT" 2>/dev/null || true
git -C scip checkout -q "$SCIP_COMMIT"
cmake -S scip -B scip/build --preset interface -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_BUILD_TYPE=Release -DSHARED=true -DLPS=spx -DSYM=snauty -DSOPLEX_DIR="$PREFIX" \
  -DPAPILO=false -DZIMPL=false -DGMP=false -DREADLINE=false -DIPOPT=true -DIPOPT_DIR="$PREFIX" \
  -DTPI=tny -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build scip/build -j"$JOBS"
cmake --install scip/build
"$PREFIX/bin/scip" -c quit | sed -n 1p   # sed reads it all: head would SIGPIPE under pipefail
